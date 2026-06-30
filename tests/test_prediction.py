
import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import yaml
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import FraudPredictor  # noqa: E402

@pytest.fixture
def fake_predictor(tmp_path):
    """
    Builds a tiny model + a matching params.yaml inside a pytest temp
    directory, so this test is fully self-contained and never touches
    your real models/fraud_model.pkl.
    """
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({f"V{i}": np.random.randn(n) for i in range(1, 6)})
    X["Time"] = np.random.randint(0, 1000, n)
    X["Amount"] = np.random.uniform(1, 100, n)
    y = pd.Series([0] * 90 + [1] * 10)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "fraud_model.pkl"
    joblib.dump(model, model_path)

    params = {
        "paths": {"model_output": str(model_path)},
        "api": {"prediction_threshold": 0.5},
        "data": {"target_column": "Class"},
    }
    params_path = tmp_path / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(params, f)

    return FraudPredictor(params_path=str(params_path)), X.columns.tolist()


def test_predictor_loads_model(fake_predictor):
    predictor, _ = fake_predictor
    assert predictor.model is not None


def test_predict_one_returns_expected_keys(fake_predictor):
    predictor, columns = fake_predictor
    transaction = {col: 0.0 for col in columns}

    result = predictor.predict_one(transaction)

    assert "is_fraud" in result
    assert "fraud_probability" in result
    assert isinstance(result["is_fraud"], bool)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_batch_returns_list(fake_predictor):
    predictor, columns = fake_predictor
    transactions = [{col: 0.0 for col in columns} for _ in range(3)]

    results = predictor.predict_batch(transactions)

    assert len(results) == 3
    for r in results:
        assert "is_fraud" in r
        assert "fraud_probability" in r


def test_predict_one_missing_model_file_raises(tmp_path):
    params = {
        "paths": {"model_output": str(tmp_path / "does_not_exist.pkl")},
        "api": {"prediction_threshold": 0.5},
        "data": {"target_column": "Class"},
    }
    params_path = tmp_path / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(params, f)

    with pytest.raises(FileNotFoundError):
        FraudPredictor(params_path=str(params_path))
