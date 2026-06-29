
import os
import json
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(train_path: str, test_path: str, target_col: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Loaded processed data: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_model(params: dict):
    """Construct the model defined in params.yaml (config-swappable)."""
    model_type = params["training"]["model_type"]

    if model_type == "random_forest":
        cfg = params["training"]["random_forest"]
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            random_state=cfg["random_state"],
            n_jobs=cfg["n_jobs"],
        )
    elif model_type == "logistic_regression":
        cfg = params["training"]["logistic_regression"]
        model = LogisticRegression(
            max_iter=cfg["max_iter"],
            C=cfg["C"],
            random_state=cfg["random_state"],
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}' in params.yaml")

    print(f"Built model: {model_type}")
    return model, model_type


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute metrics that matter for fraud detection specifically.

    Accuracy alone is misleading here -- a model that predicts
    'not fraud' every time still scores ~99.8% accuracy. Precision,
    recall, and ROC-AUC tell the real story.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)
    print("Metrics:", json.dumps(metrics, indent=2))

    return metrics


def main():
    params = load_params()

    train_path = params["data"]["processed_train_path"]
    test_path = params["data"]["processed_test_path"]
    target_col = params["data"]["target_column"]
    model_output = params["paths"]["model_output"]
    metrics_output = params["paths"]["metrics_output"]

    mlflow_tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    registered_model_name = params["mlflow"]["registered_model_name"]

    # 1. Load processed data
    X_train, X_test, y_train, y_test = load_processed_data(train_path, test_path, target_col)

    # 2. Configure MLflow (local folder-based tracking, no server needed)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # 3. Build and train model
        model, model_type = build_model(params)
        model.fit(X_train, y_train)

        # 4. Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # 5. Log params, metrics, and model to MLflow
        mlflow.log_param("model_type", model_type)
        if model_type == "random_forest":
            mlflow.log_params(params["training"]["random_forest"])
        else:
            mlflow.log_params(params["training"]["logistic_regression"])

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: run_id={run_id}")

    # 6. Also save the model + metrics locally so the API and Docker
    #    container don't need to depend on the MLflow store at serve time.
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(model, model_output)
    print(f"Saved model -> {model_output}")

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    with open(metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_output}")


if __name__ == "__main__":
    main()