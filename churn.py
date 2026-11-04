"""Shared preprocessing for training and the Streamlit app.

Both use the same steps, so a prediction in the app sees exactly
the features the model was trained on.
"""

import pandas as pd

TARGET = "Churn"
DROP_COLUMNS = ["customerID", TARGET, "tenure"]
TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numbers and drop the 11 rows where it is blank."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df.dropna(how="any")


def add_tenure_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Group tenure (months) into 12-month bins."""
    df = df.copy()
    df["tenure_bin"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    )
    return df


def encode(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode features. Pass `columns` to align new data with training columns."""
    df = add_tenure_bin(df)
    features = df.drop(columns=[c for c in DROP_COLUMNS if c in df])
    if columns is None:
        # Training: drop the first category of each column to avoid redundant dummies
        return pd.get_dummies(features, drop_first=True).astype(float)
    # Prediction: never use drop_first here. On a single row each column has one
    # category, so drop_first would delete it and every input would encode as 0.
    # Reindexing to the training columns removes the baseline dummies instead.
    X = pd.get_dummies(features, drop_first=False)
    return X.reindex(columns=columns, fill_value=0).astype(float)
