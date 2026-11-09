"""
app.py
------
StudyRisk teacher dashboard (Streamlit).

Run with:
    streamlit run app.py

Set ANTHROPIC_API_KEY as an environment variable to get live LLM-generated
alerts; without it, the app automatically falls back to template-based
alerts so it still works offline/without a key.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pipeline import run_pipeline, MODEL_DIR  # noqa: E402

st.set_page_config(page_title="StudyRisk — Early Warning Dashboard", page_icon="🎓", layout="wide")

RISK_COLORS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

st.title("🎓 StudyRisk — AI Early Warning Dashboard")
st.caption("SDG 4: Quality Education — ML risk prediction + LLM-generated teacher alerts")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Dataset path", value="data/student_data.csv")
    force_offline = st.checkbox(
        "Force offline template alerts (skip LLM API)",
        value=not bool(os.environ.get("ANTHROPIC_API_KEY")),
        help="Unchecked = calls the Anthropic API using ANTHROPIC_API_KEY env var.",
    )
    risk_filter = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
    run_btn = st.button("🔄 Run / Refresh predictions", type="primary")

if "results" not in st.session_state or run_btn:
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at '{data_path}'. Run data/generate_data.py first, or point to your CSV.")
        st.stop()
    if not os.path.exists(os.path.join(MODEL_DIR, "risk_model.joblib")):
        st.error("No trained model found in models/. Run `python src/train_model.py` first.")
        st.stop()

    with st.spinner("Scoring students and generating alerts..."):
        results, alerts = run_pipeline(data_path, force_offline_llm=force_offline, only_flagged=True)
        st.session_state["results"] = results
        st.session_state["alerts"] = {a["student_id"]: a for a in alerts}

results = st.session_state["results"]
alerts = st.session_state["alerts"]

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
counts = results["predicted_risk"].value_counts()
col1.metric("Total Students", len(results))
col2.metric("🔴 High Risk", int(counts.get("High", 0)))
col3.metric("🟡 Medium Risk", int(counts.get("Medium", 0)))
col4.metric("🟢 Low Risk", int(counts.get("Low", 0)))

st.divider()

# --- Filtered student table ---
filtered = results[results["predicted_risk"].isin(risk_filter)].sort_values(
    by="predicted_risk", key=lambda s: s.map({"High": 0, "Medium": 1, "Low": 2})
)

st.subheader(f"Flagged Students ({len(filtered)})")
display_cols = ["student_id", "predicted_risk", "risk_confidence", "G1", "G2", "studytime", "failures", "absences"]
st.dataframe(
    filtered[display_cols].rename(columns={
        "predicted_risk": "Risk", "risk_confidence": "Confidence",
        "studytime": "Study Time (1-4)", "failures": "Past Failures", "absences": "Absences",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --- Student detail / AI alert viewer ---
st.subheader("Student Detail & AI-Generated Alert")
options = filtered["student_id"].tolist()
if options:
    selected = st.selectbox("Select a student", options)
    row = results[results["student_id"] == selected].iloc[0]
    alert = alerts.get(selected)

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### {RISK_COLORS.get(row['predicted_risk'], '')} {selected}")
        st.markdown(f"**Risk level:** {row['predicted_risk']}  \n**Confidence:** {row['risk_confidence']:.0%}")
        st.markdown("**Key stats**")
        st.write({
            "G1 (checkpoint 1)": int(row["G1"]),
            "G2 (checkpoint 2)": int(row["G2"]),
            "Study time (1-4)": int(row["studytime"]),
            "Past failures": int(row["failures"]),
            "Absences": int(row["absences"]),
        })

    with right:
        if alert:
            st.markdown("#### 🤖 AI Explanation")
            st.info(alert["explanation"])
            if alert.get("suggestions"):
                st.markdown("#### ✅ Suggested Interventions")
                for s in alert["suggestions"]:
                    st.markdown(f"- {s}")
            if alert.get("parent_note"):
                st.markdown("#### ✉️ Draft Note to Parent/Guardian")
                st.text_area("", value=alert["parent_note"], height=100, label_visibility="collapsed")
            st.caption(f"Generation mode: `{alert['mode']}`")
        else:
            st.success("This student is low risk — no alert generated.")
else:
    st.info("No students match the selected risk filters.")

st.divider()
st.caption(
    "Built for an FYP project demonstrating ML prediction + LLM-based automation "
    "in support of SDG 4 (Quality Education). Dataset is synthetic by default — "
    "swap in data/student_data.csv with real (anonymized) records for production use."
)
