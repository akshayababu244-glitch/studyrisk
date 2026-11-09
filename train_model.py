"""
train_model.py
---------------
Trains a RandomForest classifier to predict student risk category
(High / Medium / Low) from early-warning features (attendance, study time,
past failures, G1/G2 checkpoint grades, demographics).

Saves:
  models/risk_model.joblib       - trained classifier
  models/encoders.joblib         - categorical encoders (needed to score new data)
  models/feature_importance.csv  - ranked feature importances (used by the LLM layer)

Usage:
    python src/train_model.py --data data/student_data.csv
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import build_features, risk_int_to_label  # noqa: E402


def train(data_path: str, model_dir: str, seed: int = 42):
    df = pd.read_csv(data_path)
    X, y, encoders, feature_names = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("=" * 60)
    print(f"Accuracy:        {acc:.3f}")
    print(f"Macro F1-score:  {f1:.3f}")
    print("-" * 60)
    target_names = [risk_int_to_label(i) for i in sorted(y.unique())]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix (rows=actual, cols=predicted), order:", target_names)
    print(confusion_matrix(y_test, y_pred))
    print("=" * 60)

    # Feature importance -> used later by the LLM to explain WHY a student is flagged
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    print("\nTop 10 features driving risk prediction:")
    print(importance_df.head(10).to_string(index=False))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "risk_model.joblib"))
    joblib.dump(encoders, os.path.join(model_dir, "encoders.joblib"))
    importance_df.to_csv(os.path.join(model_dir, "feature_importance.csv"), index=False)
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))

    print(f"\nSaved model + encoders + feature importances to '{model_dir}/'")
    return clf, encoders, importance_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/student_data.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.model_dir, args.seed)
