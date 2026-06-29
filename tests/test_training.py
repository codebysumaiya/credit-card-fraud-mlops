import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from src.data_training import build_model, evaluate_model  # noqa: E402


FAKE_PARAMS = {
    "training": {
        "model_type": "random_forest",
        "random_forest": {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
        "logistic_regression": {
            "max_iter": 200,
            "C": 1.0,
            "random_state": 42,
        },
    }
}


def make_fake_train_test():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({f"feat_{i}": np.random.randn(n) for i in range(5)})
    y = pd.Series([0] * 150 + [1] * 50)

    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]
    return X_train, X_test, y_train, y_test


def test_build_model_random_forest():
    model, model_type = build_model(FAKE_PARAMS)

    assert model_type == "random_forest"
    assert model.n_estimators == 10


def test_build_model_logistic_regression():
    params = {"training": {**FAKE_PARAMS["training"], "model_type": "logistic_regression"}}
    model, model_type = build_model(params)

    assert model_type == "logistic_regression"
    assert model.C == 1.0


def test_build_model_invalid_type_raises():
    params = {"training": {**FAKE_PARAMS["training"], "model_type": "not_a_real_model"}}

    try:
        build_model(params)
        assert False, "Should have raised ValueError for unknown model_type"
    except ValueError:
        pass


def test_model_trains_and_evaluates_without_error():
    X_train, X_test, y_train, y_test = make_fake_train_test()
    model, _ = build_model(FAKE_PARAMS)

    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0, f"{key} should be between 0 and 1"
        assert not np.isnan(metrics[key]), f"{key} should not be NaN"