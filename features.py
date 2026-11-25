"""Shared feature engineering, used by both train.py and main.py."""
import numpy as np
import pandas as pd

ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").replace(0, np.nan)

    if {"Glucose", "BMI"}.issubset(out.columns):
        out["Glucose_BMI_Index"] = (out["Glucose"] * out["BMI"]) / 100.0
    if {"Glucose", "Insulin"}.issubset(out.columns):
        out["Glucose_Insulin_Ratio"] = out["Glucose"] / (out["Insulin"].fillna(80.0) + 1.0)
    if {"Age", "BMI"}.issubset(out.columns):
        out["Age_BMI_Load"] = out["Age"] * out["BMI"]
    if {"Glucose", "DiabetesPedigreeFunction"}.issubset(out.columns):
        out["Metabolic_Genetic_Score"] = out["Glucose"] * (1.0 + out["DiabetesPedigreeFunction"])

    return out