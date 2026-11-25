import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors


# 1. Initialize Spark and Load Model
# Note: Running Spark inside Streamlit is heavy. For a production app, you'd use a Flask API backend.
# For this project, we load the Spark model and use it.
@st.cache_resource
def load_model():
    spark = SparkSession.builder.appName("FlightDelay_Dashboard").getOrCreate()
    # TODO: CHANGE THIS TO MATCH THE SAVE PATH IN model_training.py
    model = PipelineModel.load("flight_delay_rf_model")
    return spark, model


spark, model = load_model()

# 2. Define the UI
st.title("✈️ SkyCast: Flight Delay Predictor")
st.write("Enter flight details below to predict the probability of a delay category.")

col1, col2 = st.columns(2)

with col1:
    origin = st.text_input("Origin Airport (e.g., JFK)", "JFK")
    airline = st.text_input("Airline Code (e.g., AA)", "AA")
    departure_hour = st.slider("Scheduled Departure Hour (0-23)", 0, 23, 14)

with col2:
    dest = st.text_input("Destination Airport (e.g., LAX)", "LAX")
    turnaround = st.number_input(
        "Aircraft Turnaround Time (Minutes)", min_value=0, max_value=600, value=60
    )
    # TODO: ADD THESE FIELDS TO THE UI IF USING WEATHER DATA
    precipitation = st.number_input(
        "Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0
    )
    wind_speed = st.number_input(
        "Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0
    )

# Map hour to Time_of_Day
if 5 <= departure_hour < 12:
    time_of_day = "Morning"
elif 12 <= departure_hour < 18:
    time_of_day = "Afternoon"
elif 18 <= departure_hour < 24:
    time_of_day = "Evening"
else:
    time_of_day = "Night"

# 3. Prediction Logic
if st.button("Predict Delay Category"):
    # Create a dataframe matching the model's expected input columns (BEFORE vector assembler)
    # Spark models expect a DataFrame with the original columns.
    # TODO: ENSURE THESE COLUMNS MATCH THE TRAINING SCRIPT EXACTLY
    # If you added weather columns, add them here and in the UI above.
    input_data = [
        (
            turnaround,
            departure_hour,
            time_of_day,
            airline,
            origin,
            dest,
            # temprature,
            precipitation,
            wind_speed,
        )
    ]

    columns = [
        "Turnaround_Time_Mins",
        "Departure_Hour",
        "Time_of_Day",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        # "O_TEMP",
        "O_PRCP",
        "O_WSPD",
    ]

    # Create Spark DataFrame
    df_input = spark.createDataFrame(input_data, columns)

    # Predict
    prediction = model.transform(df_input)

    # Extract probability
    prob_array = prediction.select("probability").collect()[0][0]

    # Map classes: 0: On Time, 1: Short, 2: Medium, 3: Long
    labels = [
        "On Time (<15m)",
        "Short Delay (15m-1h)",
        "Medium Delay (1-2h)",
        "Long Delay (>2h)",
    ]

    st.subheader("Prediction Results:")
    for i, label in enumerate(labels):
        st.write(f"**{label}**: {prob_array[i]*100:.2f}%")

    # Get the most likely outcome
    predicted_class_index = prob_array.argmax()
    st.success(f"Most Likely Outcome: **{labels[predicted_class_index]}**")
