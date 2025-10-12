"""
feature_engineering.py
Feature engineering helpers: leak detection, train/test split helpers.
"""
import pandas as pd
from typing import List, Tuple


def drop_leaky_features(df: pd.DataFrame, leak_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in leak_cols:
        if c in df.columns:
            print(f"Dropping leaky column: {c}")
            df = df.drop(columns=[c])
    return df

def detect_leakage(df, target, threshold=0.9):
    """
    Identify potential leaky columns based on high correlation with target.
    Only for numeric features.
    """
    numeric = df.select_dtypes('number')
    corr = numeric.corr()[target].drop(target)
    leaky = corr[abs(corr) > threshold].index.tolist()
    return leaky


def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
