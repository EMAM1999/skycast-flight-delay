import subprocess
import sys
import time
from data_preprocessing import run_preprocessing
from model_training import run_training


def run_pipeline():
    print("========================================")
    print("   Starting SkyCast ML Pipeline         ")
    print("========================================")

    start_time = time.time()

    # Step 1: Run Data Preprocessing
    print("\n[Step 1/2] Executing Data Preprocessing...")
    # result_prep = subprocess.run([sys.executable, "data_preprocessing.py"])

    if not run_preprocessing():
        print("\n❌ ERROR: Data preprocessing failed.")
        sys.exit(1)

    print("✅ Data Preprocessing completed.")

    # Step 2: Run Model Training
    print("\n[Step 2/2] Executing Model Training...")
    # result_train = subprocess.run([sys.executable, "model_training.py"])

    if not run_training():
        print("\n❌ ERROR: Model training failed.")
        sys.exit(1)

    print("✅ Model Training completed.")

    end_time = time.time()
    total_time = (end_time - start_time) / 60

    print("\n========================================")
    print(f" 🎉 PIPELINE COMPLETE! Time: {total_time:.2f} mins")
    print("========================================")
    print("Next step: Run 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    run_pipeline()
