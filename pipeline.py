"""
pipeline.py
-----------
End-to-end "EduSentry-lite" pipeline:
  1. Load trained model + encoders
  2. Score every student -> risk label + probability
  3. For High/Medium risk students, get per-student top contributing factors
     (via the model's feature importances, intersected with that student's
     own weak-signal values) and generate an LLM alert
  4. Return a results DataFrame + list of alert dicts

This is the module both the CLI demo (demo.py) and the Streamlit app (app.py)
call into, so there's one source of truth for the pipeline logic.
"""

import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import build_features, risk_int_to_label  # noqa: E402
from src.llm_alerts import generate_alert  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def load_artifacts(model_dir: str = MODEL_DIR):
    clf = joblib.load(os.path.join(model_dir, "risk_model.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    importance_df = pd.read_csv(os.path.join(model_dir, "feature_importance.csv"))
    return clf, feature_names, importance_df


def _per_student_top_factors(student_row, importance_df, k=5):
    """
    Heuristic: rank globally-important features by importance, but only surface
    ones where this specific student's value looks "weak" (low study time, high
    absences/failures, low G1/G2, high Dalc/Walc, low famrel). This gives a
    personalized explanation rather than repeating the same global top-5 for
    every student.
    """
    weak_if_low = {"studytime", "G1", "G2", "famrel", "freetime", "Medu", "Fedu", "health"}
    weak_if_high = {"failures", "absences", "Dalc", "Walc", "goout"}

    scored = []
    for _, row in importance_df.iterrows():
        feat, imp = row["feature"], row["importance"]
        if feat not in student_row:
            continue
        val = student_row[feat]
        is_weak = False
        if feat in weak_if_low and val <= 2:
            is_weak = True
        elif feat in weak_if_high and val >= 3:
            is_weak = True
        elif feat in ("G1", "G2") and val < 10:
            is_weak = True
        if is_weak:
            scored.append((feat, float(imp)))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:  # fallback: just use global top-k
        scored = list(importance_df.head(k)[["feature", "importance"]].itertuples(index=False, name=None))
    return scored[:k]


def run_pipeline(data_path: str, model_dir: str = MODEL_DIR, force_offline_llm: bool = False,
                  only_flagged: bool = True):
    df = pd.read_csv(data_path)
    clf, feature_names, importance_df = load_artifacts(model_dir)

    X, y_true, _, _ = build_features(df)
    proba = clf.predict_proba(X)
    pred = clf.predict(X)

    results = df.copy()
    results["predicted_risk"] = [risk_int_to_label(p) for p in pred]
    results["risk_confidence"] = proba.max(axis=1).round(3)

    alerts = []
    to_process = results if not only_flagged else results[results["predicted_risk"] != "Low"]

    for _, row in to_process.iterrows():
        top_factors = _per_student_top_factors(row, importance_df)
        alert = generate_alert(
            student_id=row["student_id"],
            risk_label=row["predicted_risk"],
            top_factors=top_factors,
            student_row=row,
            force_offline=force_offline_llm,
        )
        alerts.append(alert)

    return results, alerts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/student_data.csv")
    parser.add_argument("--offline", action="store_true", help="force template-based alerts, skip LLM API calls")
    parser.add_argument("--limit", type=int, default=5, help="how many sample alerts to print")
    args = parser.parse_args()

    results, alerts = run_pipeline(args.data, force_offline_llm=args.offline)

    print(f"\nScored {len(results)} students.")
    print(results["predicted_risk"].value_counts())
    print(f"\nGenerated {len(alerts)} alerts. Showing first {args.limit}:\n")

    for a in alerts[: args.limit]:
        print("-" * 60)
        print(f"Student: {a['student_id']}  |  Risk: {a['risk_label']}  |  mode: {a['mode']}")
        print(f"Explanation: {a['explanation']}")
        for s in a.get("suggestions", []):
            print(f"  - {s}")
        if a.get("parent_note"):
            print(f"Parent note: {a['parent_note']}")
