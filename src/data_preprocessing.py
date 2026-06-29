
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(params_path: str = "params.yaml") -> dict:
    """Read the central config file."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load the raw Kaggle creditcard.csv into a DataFrame."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at '{raw_path}'. "
            "Download the Kaggle 'Credit Card Fraud Detection' dataset "
            "and place creditcard.csv in data/raw/."
        )
    df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows and any rows with missing values."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    after = len(df)
    print(f"Cleaning: removed {before - after} rows (duplicates/nulls)")
    return df


def scale_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Scale specified columns to have mean 0 and std 1.
    The other columns (V1-V28) are already PCA-transformed by Kaggle,
    so they don't need scaling — only Amount and Time do.
    """
    scaler = StandardScaler()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"Scaled columns: {columns}")
    return df


def handle_imbalance(X: pd.DataFrame, y: pd.Series, method: str, random_state: int):
    """
    Fraud datasets are extremely imbalanced (~0.17% fraud).
    Without balancing, a model can score 99.8% accuracy while
    catching zero fraud cases. This fixes that.
    """
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        print(f"Applied SMOTE: {len(y)} -> {len(y_res)} rows "
              f"(fraud cases balanced to {sum(y_res)} / {len(y_res)})")
        return X_res, y_res
    elif method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X, y)
        print(f"Applied undersampling: {len(y)} -> {len(y_res)} rows")
        return X_res, y_res
    else:
        print("No imbalance handling applied (method='none')")
        return X, y


def main():
    params = load_params()

    raw_path = params["data"]["raw_path"]
    train_out = params["data"]["processed_train_path"]
    test_out = params["data"]["processed_test_path"]
    target_col = params["data"]["target_column"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    scale_cols = params["preprocessing"]["scale_columns"]
    handle_imb = params["preprocessing"]["handle_imbalance"]
    imb_method = params["preprocessing"]["imbalance_method"]

    # 1. Load
    df = load_raw_data(raw_path)

    # 2. Clean
    df = clean_data(df)

    # 3. Scale Amount and Time
    df = scale_columns(df, scale_cols)

    # 4. Split features/target BEFORE balancing
    #    (balancing must only ever be applied to training data, never test data,
    #    otherwise we'd be evaluating on synthetic rows that aren't real transactions)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Split: {len(X_train)} train rows, {len(X_test)} test rows")

    # 5. Balance ONLY the training set
    if handle_imb:
        X_train, y_train = handle_imbalance(X_train, y_train, imb_method, random_state)

    # 6. Recombine and save
    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(test_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Saved processed train set -> {train_out}")
    print(f"Saved processed test set  -> {test_out}")


if __name__ == "__main__":
    main()