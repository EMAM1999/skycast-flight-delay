import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors

MODEL_PATH = "flight_delay_rf_model"


# 1. Initialize Spark and Load Model
# Note: Running Spark inside Streamlit is heavy. For a production app, you'd use a Flask API backend.
# For this project, we load the Spark model and use it.
@st.cache_resource
def load_model(model_path):
    spark = SparkSession.builder.appName("FlightDelay_Dashboard").getOrCreate()
    # TODO: CHANGE THIS TO MATCH THE SAVE PATH IN model_training.py
    model = PipelineModel.load(model_path)
    return spark, model


spark, model = load_model(MODEL_PATH)

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
    # turnaround = st.number_input(
    #     "Aircraft Turnaround Time (Minutes)", min_value=0, max_value=600, value=60
    # )
    o_temp = st.number_input("Origin Temperature (°F)", -50.0, 130.0, 60.0)
    o_prcp = st.number_input("Origin Precipitation (in)", 0.0, 10.0, 0.0)
    o_wspd = st.number_input("Origin Wind Speed (mph)", 0.0, 100.0, 5.0)

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
    input_data = [
        (
            # turnaround,
            departure_hour,
            time_of_day,
            airline,
            origin,
            dest,
            o_temp,
            o_prcp,
            o_wspd,
        )
    ]

    columns = [
        # "Turnaround_Time_Mins",
        "Departure_Hour",
        "Time_of_Day",
        "OP_CARRIER",
        "ORIGIN",
        "DEST",
        "O_TEMP",
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
