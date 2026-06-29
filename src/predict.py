
import yaml
import joblib
import numpy as np
import pandas as pd


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


class FraudPredictor:
    """
    Wraps the trained model so callers don't need to know about
    file paths, column order, or thresholds -- they just pass in
    transaction data and get a prediction back.
    """

    def __init__(self, params_path: str = "params.yaml"):
        params = load_params(params_path)
        self.model_path = params["paths"]["model_output"]
        self.threshold = params["api"]["prediction_threshold"]
        self.target_col = params["data"]["target_column"]
        self.model = self._load_model()
        # Column order the model was trained on (everything except the target)
        self.expected_columns = None

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            print(f"Loaded model from {self.model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No trained model found at '{self.model_path}'. "
                "Run 'python src/data_training.py' first."
            )

    def predict_one(self, transaction: dict) -> dict:
        """
        transaction: dict of feature_name -> value
                     e.g. {"Time": 0.5, "V1": -1.2, ..., "Amount": 0.3}

        Returns: {"is_fraud": bool, "fraud_probability": float}
        """
        df = pd.DataFrame([transaction])

        # Align column order to what the model expects, filling any
        # missing engineered columns with 0 rather than crashing.
        if hasattr(self.model, "feature_names_in_"):
            expected = list(self.model.feature_names_in_)
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected]

        proba = self.model.predict_proba(df)[0][1]
        is_fraud = bool(proba >= self.threshold)

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(float(proba), 6),
        }

    def predict_batch(self, transactions: list) -> list:
        """Same as predict_one but for a list of transactions at once."""
        return [self.predict_one(t) for t in transactions]


# Quick manual sanity check when running this file directly
if __name__ == "__main__":
    predictor = FraudPredictor()

    # A dummy all-zero transaction just to confirm the pipeline runs end to end.
    # Replace with a real row from data/processed/test.csv for a meaningful check.
    dummy_transaction = {f"V{i}": 0.0 for i in range(1, 29)}
    dummy_transaction["Time"] = 0.0
    dummy_transaction["Amount"] = 0.0

    result = predictor.predict_one(dummy_transaction)
    print("Sample prediction:", result)