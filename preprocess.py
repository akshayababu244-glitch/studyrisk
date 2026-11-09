"""
preprocess.py
-------------
Turns the raw student dataset into:
  1. A risk LABEL (Low / Medium / High), derived from the final grade G3.
  2. A FEATURE matrix usable at "early warning" time — i.e. we deliberately
     exclude G3 (the thing we're predicting) and only use G1, G2 (early
     checkpoint grades) plus behavioral/demographic features. This mirrors
     a real deployment: at week 10 you don't know the final grade yet, but
     you do know attendance, study habits, and the first two assessments.

Risk bands (grades are on a 0-20 scale, as in the source dataset):
    G3 < 10   -> High risk    (failing threshold)
    10-14     -> Medium risk
    >= 15     -> Low risk
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]

TARGET_COL = "risk_label"


def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def label(g3):
        if g3 < 10:
            return "High"
        elif g3 < 15:
            return "Medium"
        else:
            return "Low"

    df[TARGET_COL] = df["G3"].apply(label)
    return df


def build_features(df: pd.DataFrame):
    """
    Returns (X, y, encoders) where X is a numeric feature matrix ready for
    sklearn, y is the encoded risk label, and encoders lets you decode /
    re-encode new rows consistently (needed when scoring new students later).
    """
    df = add_risk_label(df)

    encoders = {}
    X = df[NUMERIC_COLS].copy()

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col])
        encoders[col] = le

    label_encoder = LabelEncoder()
    # Fix the order so High/Medium/Low map predictably
    label_encoder.classes_ = pd.array(["High", "Medium", "Low"])
    y = df[TARGET_COL].map({"High": 0, "Medium": 1, "Low": 2})

    feature_names = NUMERIC_COLS + CATEGORICAL_COLS
    return X[feature_names], y, encoders, feature_names


def risk_int_to_label(i: int) -> str:
    return {0: "High", 1: "Medium", 2: "Low"}[int(i)]
