# SkyCast ✈️

**A Distributed Machine Learning Approach to Real-Time Flight Delay Prediction and Visualization**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

**SkyCast** is a Big Data project that predicts US domestic flight delays using a distributed **Apache Spark** pipeline. Instead of predicting exact delay minutes (which is noisy), we bin delays into **four actionable categories** and train a distributed **Random Forest** classifier on **50+ million historical flights** enriched with weather features.

A **Streamlit dashboard** lets a user enter flight details (origin, destination, carrier, hour, weather) and receive a probability distribution over the four delay categories.

---

## 🎯 Problem Statement

Flight delays cost the aviation industry billions of dollars annually and disrupt millions of passengers. Today's ecosystem is *reactive* — passengers and airlines only learn about delays when it's too late to mitigate them.

We build a **proactive, distributed ML pipeline** that:
- Ingests tens of millions of flight records + weather data
- Engineers time-of-day and weather features
- Predicts whether a flight will be **On Time**, **Short**, **Medium**, or **Long** delayed
- Serves predictions via an interactive dashboard

### Why is this a Big Data problem?

| Dimension | Justification |
|---|---|
| **Volume** | 50M+ flight records (~44 GB raw) |
| **Variety** | Structured flight logs + semi-structured weather |
| **Velocity** | Extendable to real-time weather API streams |

A single-machine tool (e.g., Pandas) would OOM or take hours — hence Spark.

---

## 🎯 Target Variable (Multi-Class)

Per instructor guidance, delays are **binned into hours** rather than minutes to reduce noise:

| Class | Label | Range |
|---|---|---|
| `0` | On Time | `< 15 min` |
| `1` | Short Delay | `15 min – 1 hr` |
| `2` | Medium Delay | `1 – 2 hrs` |
| `3` | Long Delay | `> 2 hrs` |

---

## 🗂️ Repository Structure
skycast-flight-delay/
```bash
│
└── src/ 
    ├── data_preprocessing.py # Load, clean, engineer features, join weather, save Parquet
    ├── model_training.py # Train Random Forest pipeline, evaluate, save model
    ├── run.py # Orchestrates preprocessing → training
    ├── app.py # Streamlit dashboard (software deliverable)
├── requirements.txt # Python dependencies
├── README.md
│
└── data/ # (gitignored) place raw CSVs here
    ├── flight_with_weather_2024.csv
    └── daily_weather.csv
```
---

## 📊 Datasets

| Dataset | Description | Size | Source |
|---|---|---|---|
| **Aeolus: A Multi-structural Flight Delay Dataset** | NeurIPS 2025 benchmark; 50M+ flights, 306 columns | 44 GB | [Kaggle](https://www.kaggle.com/datasets/flnny123/mfddmulti-modal-flight-delay-dataset) |
| **~5 Years of Daily Weather NOAA** | 1,728 days of daily weather, 27 columns | 188 KB | [Kaggle](https://www.kaggle.com/datasets/davidliaos/daily-weather-noaa) |

> **Download steps:**
> 1. Sign in to Kaggle and accept the dataset licenses.
> 2. Download and place the CSVs in the `data/` directory.
> 3. Update the `flight_data_path` and `weather_data_path` variables in `data_preprocessing.py`.

**Key flight columns used:** `FL_DATE`, `OP_CARRIER`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `ARR_TIME`, `DEP_DELAY`, `O_TEMP`, `O_PRCP`, `O_WSPD`.

---

## ⚙️ Installation

### 1. Prerequisites
- Python **3.10+**
- Java **8 or 11** (required by PySpark)
- ~8 GB free RAM (for the 10% sample; more for full data)

### 2. Clone and install

```bash
git clone https://github.com/<your-username>/skycast-flight-delay.git
cd skycast-flight-delay

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. requirements.txt

```
pyspark==3.5.1
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.4
```

## 🚀 Usage

### Step 1 — Configure paths
Open data_preprocessing.py and set:
```bash
flight_data_path = "/absolute/path/to/flight_with_weather_2024.csv"
weather_data_path = "/absolute/path/to/daily_weather.csv"
```
Use absolute paths, not ~ or D:/.... Spark does not expand ~.

### Step 2 — Run the full pipeline
```bash
python run.py
```
This executes:

1. **Preprocessing** — samples 10%, bins delays, extracts `Departure_Hour` and `Time_of_Day`, joins weather by date, writes `processed_flights.parquet`.

2. **Training** — builds a Spark ML pipeline (StringIndexer → OneHotEncoder → VectorAssembler → RandomForest), evaluates on a 20% test split, prints F1-Score and a confusion matrix, saves the model to `flight_delay_rf_model/`.

#### Expected output:
```bash
[Step 1/2] Executing Data Preprocessing...
✅ Data Preprocessing completed.
[Step 2/2] Executing Model Training...
Test F1-Score: 0.xxxx
Confusion Matrix: ...
Model saved successfully!
🎉 PIPELINE COMPLETE! Time: x.xx mins
```

### Step 3 — Launch the dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501, enter flight details, and click Predict Delay Category.

## 🧠 Methodology
### Data Preprocessing
|Step |Technique
|---|---|
|Sampling	|10% random sample for local reproducibility
|Timestamp |parsing	Cast CRS_DEP_TIME / ARR_TIME (already timestamps)
|Binning	|DEP_DELAY → 4-category Delay_Category
|Time features	|Departure_Hour, Time_of_Day (Morning/Afternoon/Evening/Night)
|Missing values	|dropna on critical columns
|Weather join	|Coerce both sides to yyyy-MM-dd; dedupe weather by date

### Modeling Pipeline (Spark MLlib)
```
StringIndexer  →  OneHotEncoder  →  VectorAssembler  →  RandomForestClassifier
   (4 cats)       (4 cats)             (8 features)         (100 trees, depth 10)
```
Features used: Departure_Hour, Time_of_Day, OP_CARRIER, ORIGIN, DEST, O_TEMP, O_PRCP, O_WSPD.

## Evaluation

Because classes are highly imbalanced (most flights are on time), we report:
* Weighted F1-Score (primary)
* Confusion Matrix (to inspect per-class performance)

Accuracy is intentionally not used — it would be misleading.

## 📈 Results

Update this section once you run the full pipeline on your machine.

Metric	|Value
---|---
Training rows	|TBD
Test rows	|TBD
Weighted F1	|TBD
On-Time Recall	|TBD

Confusion matrix (classes 0–3): see run.py output.

## 🖥️ Streamlit Dashboard

The dashboard accepts:

* Origin / Destination airport
* Carrier code
* Scheduled departure hour
* Origin temperature, precipitation, wind speed

and returns the probability distribution across the 4 delay categories.

## ⚠️ Known Limitations

1. No aircraft tail number in the current CSV → the instructor's turnaround-time (propagated delay) feature is not computable. The Time_of_Day feature partially compensates. Re-export with TAIL_NUM to enable the full block (see the commented section in data_preprocessing.py).
2. Weather dataset is single-station → the join uses date only. For a geographically correct join, use a multi-station NOAA GSOD dataset and join on ORIGIN → nearest station.
3. 10% sample is used by default to keep local runs feasible. Increase the fraction in readCSV for the full 50M-row benchmark.
4. Streamlit + Spark is heavy for production. A production system would separate Spark (batch inference) from the frontend via an API.

## 🛠️ Troubleshooting
Error	|Fix
---|---
`PATH_NOT_FOUND`	|Use an absolute path; on clusters use echo $HOME to find your real home
`DATATYPE_MISMATCH` on timestamps	|Cast CRS_DEP_TIME directly instead of concatenating with FL_DATE
`cannot resolve Tail_Number`	|Your CSV lacks the column — keep the turnaround block commented
`OutOfMemoryError`	|Lower the fraction (e.g., 0.02) or increase spark.driver.memory
Streamlit can't find model	|Run python run.py first, and ensure flight_delay_rf_model/ exists

## 👥 Authors
Name	|Role
---|---
Mohamed Ashraf Hamza (EMAM)	| Data engineering, modeling, evaluation, Environment setup, Streamlit dashboard

## 📄 License

MIT License — see LICENSE for details.

```
Note: The datasets are governed by their respective Kaggle licenses. This repository only contains code.
```