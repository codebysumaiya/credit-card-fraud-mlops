import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from data_preprocessing import clean_data, scale_columns, handle_imbalance  # noqa: E402


def make_fake_df(n_rows=200, fraud_ratio=0.05):
    """Builds a small fake transactions dataset that mimics the real schema."""
    np.random.seed(42)
    n_fraud = max(1, int(n_rows * fraud_ratio))

    data = {f"V{i}": np.random.randn(n_rows) for i in range(1, 29)}
    data["Time"] = np.random.randint(0, 100000, n_rows)
    data["Amount"] = np.random.uniform(1, 500, n_rows)
    data["Class"] = [0] * (n_rows - n_fraud) + [1] * n_fraud

    return pd.DataFrame(data)


def test_clean_data_removes_duplicates():
    df = make_fake_df()
    df_with_dupes = pd.concat([df, df.iloc[:5]], ignore_index=True)

    cleaned = clean_data(df_with_dupes)

    assert len(cleaned) == len(df), "Duplicate rows should be removed"


def test_clean_data_removes_nulls():
    df = make_fake_df()
    df.loc[0, "Amount"] = np.nan

    cleaned = clean_data(df)

    assert cleaned.isnull().sum().sum() == 0, "No nulls should remain after cleaning"


def test_scale_columns_changes_distribution():
    df = make_fake_df()
    original_mean = df["Amount"].mean()

    scaled = scale_columns(df, ["Amount", "Time"])

    # After standard scaling, mean should be ~0 (not the original raw mean)
    assert abs(scaled["Amount"].mean()) < 1e-6
    assert abs(scaled["Amount"].mean() - original_mean) > 0.01


def test_handle_imbalance_smote_balances_classes():
    df = make_fake_df(n_rows=200, fraud_ratio=0.05)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_res, y_res = handle_imbalance(X, y, method="smote", random_state=42)

    counts = y_res.value_counts()
    assert counts[0] == counts[1], "SMOTE should fully balance both classes"


def test_handle_imbalance_none_leaves_data_unchanged():
    df = make_fake_df(n_rows=200, fraud_ratio=0.05)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_res, y_res = handle_imbalance(X, y, method="none", random_state=42)

    assert len(X_res) == len(X), "method='none' should not change row count"