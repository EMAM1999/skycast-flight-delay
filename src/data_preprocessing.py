import os
import glob
from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, hour, when, lag, unix_timestamp
from pyspark.sql.window import Window


def readCSV(spark, path, fraction=None):
    """Read CSV(s). If 0 < fraction < 1, sample that fraction."""
    data = spark.read.csv(path, header=True, inferSchema=True)
    if fraction is not None and 0 < fraction < 1:
        return data.sample(fraction=fraction, seed=42)
    return data


def load_and_combine_flight_data(spark, folder_path, fraction):
    """
    Finds all CSV files in the given folder and combines them into a single DataFrame.
    Uses Spark's native wildcard reading for maximum performance.
    """
    print(f"Searching for CSV files in: {folder_path}")

    # Construct the wildcard path (e.g., "data/flights/*.csv")
    wildcard_path = os.path.join(folder_path, "*.csv")

    # Check if any files exist to prevent cryptic Spark errors
    if not glob.glob(wildcard_path):
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    print("Loading and combining files...")
    # Spark will automatically combine all files matching the wildcard into one DataFrame
    return readCSV(spark, wildcard_path, fraction)


# --- FALLBACK FUNCTION (Only use if the wildcard method fails due to schema mismatch) ---
def load_and_combine_flight_data_manual(spark, folder_path):
    """Use this ONLY if the files have different column orders or missing columns."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for file in all_files:
        df = spark.read.csv(file, header=True, inferSchema=True)
        dfs.append(df)
    # unionByName safely merges DataFrames even if column order differs
    return reduce(DataFrame.unionByName, dfs)


def is_directory(data_path: str):
    return os.path.isdir(data_path)


def run_preprocessing(input_data, processed_data_path):
    print("--- Starting Data Preprocessing ---")

    spark = (
        SparkSession.builder.appName("FlightDelay_Preprocessing")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    # Example: "C:/Data/Flights_CSVs/" or "/dbfs/FileStore/flights/"
    flight_data_path = input_data[0]

    weather_data_path = input_data[1]

    # 1. Load and Combine Flight Data (if multible csv files)
    print("Loading Flight Data...")
    df_flights = (
        load_and_combine_flight_data(spark, flight_data_path, 0.1)
        if is_directory(flight_data_path)
        else readCSV(spark, flight_data_path, 0.1)
    )

    # 2. Load Weather Data
    print("Loading Weather Data...")
    df_weather = readCSV(spark, weather_data_path)

    print(f"Total combined flight records: {df_flights.count()}")

    # a. Convert Time Columns to Timestamps for calculations
    df_flights = df_flights.withColumn(
        "ScheduledDep", col("CRS_DEP_TIME").cast("timestamp")
    )
    df_flights = df_flights.withColumn("ActualArr", col("ARR_TIME").cast("timestamp"))

    # b. Calculate Turnaround Time (Propagated Delays)
    # window_spec = Window.partitionBy("TAIL_NUM").orderBy("ScheduledDep")
    # df_flights = df_flights.withColumn(
    #     "PreviousArrTime", lag("ActualArr", 1).over(window_spec)
    # )
    # df_flights = df_flights.withColumn(
    #     "Turnaround_Time_Mins",
    #     (unix_timestamp("ScheduledDep") - unix_timestamp("PreviousArrTime")) / 60,
    # )

    # c. Time of Day Feature
    df_flights = df_flights.withColumn("Departure_Hour", hour("ScheduledDep"))
    df_flights = df_flights.withColumn(
        "Time_of_Day",
        when((col("Departure_Hour") >= 5) & (col("Departure_Hour") < 12), "Morning")
        .when((col("Departure_Hour") >= 12) & (col("Departure_Hour") < 18), "Afternoon")
        .when((col("Departure_Hour") >= 18) & (col("Departure_Hour") < 24), "Evening")
        .otherwise("Night"),
    )

    # d. Bin Delay into Hours (Multi-class Classification)
    df_flights = df_flights.withColumn(
        "Delay_Category",
        when(col("DEP_DELAY") < 15, 0)
        .when((col("DEP_DELAY") >= 15) & (col("DEP_DELAY") < 60), 1)
        .when((col("DEP_DELAY") >= 60) & (col("DEP_DELAY") < 120), 2)
        .otherwise(3),
    )

    # 4. Handle Missing Values
    df_flights = df_flights.dropna(
        # subset=["DEP_DELAY", "Turnaround_Time_Mins", "Time_of_Day"]
        subset=["DEP_DELAY", "Time_of_Day"]
    )

    df_flights = df_flights.withColumn("JoinDate", col("FL_DATE"))
    df_weather = df_weather.withColumn("JoinDate", col("date"))
    df_weather = df_weather.dropDuplicates(["JoinDate"])

    df_merged = df_flights.join(df_weather, on="JoinDate", how="left")

    print("Data Preprocessing Complete. Saving to Parquet for fast training...")

    df_merged.coalesce(10).write.mode("overwrite").parquet(processed_data_path)

    print("Preprocessing finished successfully!\n")
    spark.stop()
    return True


if __name__ == "__main__":
    run_preprocessing(
        input_data=[
            f"absolute/path/to/folder/flight_with_weather_2024.csv",
            f"absolute/path/to/folder/daily_weather.csv",
        ],
        processed_data_path="processed_flights.parquet",
    )
