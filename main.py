
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src import data_preprocessing  # noqa: E402
from src import data_training       # noqa: E402


def run_pipeline():
    print("=" * 60)
    print("STARTING FULL PIPELINE: preprocessing -> training")
    print("=" * 60)

    start = time.time()

    print("\n[1/2] Running data preprocessing...\n")
    data_preprocessing.main()

    print("\n[2/2] Running model training...\n")
    data_training.main()

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print("Model saved to models/fraud_model.pkl")
    print("Metrics saved to models/metrics.json")
    print("Run 'mlflow ui' to view the experiment run.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()