"""Train the diabetes model ONCE and save it.

Run:  python train.py
Creates:  model.pkl  (the full pipeline)  and  metrics.json  (evaluation results)
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from features import engineer_features

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "diabetes.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
RANDOM_STATE = 42


def build_ensemble() -> Pipeline:
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    lgbm_clf = lgbm.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=15,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    knn_clf = KNeighborsClassifier(n_neighbors=9, weights="distance")
    ensemble = VotingClassifier(
        estimators=[("lgbm", lgbm_clf), ("rf", rf_clf), ("knn", knn_clf)],
        voting="soft",
        weights=[3, 2, 1],
    )
    return Pipeline([("prep", preprocess), ("clf", ensemble)])


def build_baseline() -> Pipeline:
    """Simple model to compare against. If the ensemble is not clearly better, say so."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def cross_validated_scores(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "recall", "precision", "f1", "roc_auc"],
    )
    return {
        name: {
            "mean": round(float(np.mean(scores[f"test_{name}"])), 4),
            "std": round(float(np.std(scores[f"test_{name}"])), 4),
        }
        for name in ["accuracy", "recall", "precision", "f1", "roc_auc"]
    }


def fit_final_model():
    """Train on ALL data and return (pipeline, base_cols). Used by main.py as a fallback."""
    df = pd.read_csv(CSV_PATH)
    base_cols = [c for c in df.columns if c != "Outcome"]
    X = engineer_features(df[base_cols].copy())
    y = df["Outcome"].astype(int)
    return build_ensemble().fit(X, y), base_cols


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    base_cols = [c for c in df.columns if c != "Outcome"]
    X = engineer_features(df[base_cols].copy())
    y = df["Outcome"].astype(int)

    print("Running 5-fold cross-validation (this takes a few seconds)...")
    ensemble_cv = cross_validated_scores(build_ensemble(), X, y)
    baseline_cv = cross_validated_scores(build_baseline(), X, y)

    # One holdout split, only for a confusion-matrix style sanity check
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    holdout_model = build_ensemble().fit(X_train, y_train)
    holdout = {
        "accuracy": round(float(accuracy_score(y_test, holdout_model.predict(X_test))), 4),
        "recall": round(float(recall_score(y_test, holdout_model.predict(X_test))), 4),
        "roc_auc": round(float(roc_auc_score(y_test, holdout_model.predict_proba(X_test)[:, 1])), 4),
    }

    # Final model: train on ALL data, then save
    final_model = build_ensemble().fit(X, y)
    joblib.dump({"pipeline": final_model, "base_cols": base_cols}, MODEL_PATH)

    metrics = {
        "ensemble_cv": ensemble_cv,
        "baseline_logreg_cv": baseline_cv,
        "holdout": holdout,
        "n_rows": int(len(df)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== 5-fold cross-validation (mean) ===")
    for name in ["accuracy", "recall", "precision", "f1", "roc_auc"]:
        print(
            f"{name:10s} ensemble: {ensemble_cv[name]['mean']:.3f}   "
            f"logistic regression: {baseline_cv[name]['mean']:.3f}"
        )
    print(f"\nSaved: {MODEL_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()