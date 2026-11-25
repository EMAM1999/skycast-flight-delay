from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline



def run_training():
    print("--- Starting Model Training ---")
    spark = SparkSession.builder.appName("FlightDelay_Modeling").getOrCreate()

    # TODO: ENSURE THIS MATCHES THE SAVE PATH IN data_preprocessing.py
    df = spark.read.parquet("processed_flights.parquet")

    # TODO: UPDATE THESE COLUMN LISTS BASED ON YOUR ACTUAL DATA
    columns_to_keep = [
        "Turnaround_Time_Mins",
        "Departure_Hour",
        "Time_of_Day",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "Delay_Category",
    ]
    df = df.select(*columns_to_keep).dropna()

    categorical_cols = ["Time_of_Day", "OP_CARRIER", "ORIGIN", "DEST"]
    numeric_cols = ["Turnaround_Time_Mins", "Departure_Hour"]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec")
        for c in categorical_cols
    ]
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{c}_vec" for c in categorical_cols],
        outputCol="features",
    )

    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training Data Count: {train_data.count()}")
    print(f"Testing Data Count: {test_data.count()}")

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="Delay_Category",
        numTrees=100,
        maxDepth=10,
        seed=42,
    )
    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    print("Training Random Forest Model...")
    model = pipeline.fit(train_data)

    predictions = model.transform(test_data)
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="Delay_Category", predictionCol="prediction", metricName="f1"
    )

    f1_score = evaluator_f1.evaluate(predictions)
    print(f"Test F1-Score: {f1_score:.4f}")

    print("Confusion Matrix:")
    predictions.groupBy("Delay_Category", "prediction").count().orderBy(
        "Delay_Category", "prediction"
    ).show()

    # TODO: CHANGE THIS SAVE PATH TO YOUR DESIRED LOCATION
    model.write().overwrite().save("flight_delay_rf_model")
    print("Model saved successfully!\n")
    spark.stop()
    return True


if __name__ == "__main__":
    run_training()
