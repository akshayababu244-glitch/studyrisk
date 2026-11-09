"""
llm_alerts.py
-------------
Turns a raw risk prediction (label + top contributing features) into a
human-readable alert: a short explanation for the teacher and 2-3 concrete
intervention suggestions, plus an optional plain-language note for parents.

Two modes:
  - LIVE mode:  calls the Anthropic API (requires ANTHROPIC_API_KEY env var)
  - OFFLINE mode: falls back to a template-based generator so the project
    still runs end-to-end with no API key / no internet (useful for demos,
    grading, or CI).

Usage:
    from src.llm_alerts import generate_alert
    alert = generate_alert(student_id, risk_label, top_factors, student_row)
"""

import os
import json

TEMPLATE_PLAYBOOK = {
    "G1": "has scored low on the first checkpoint assessment",
    "G2": "has scored low on the second checkpoint assessment",
    "failures": "has one or more past course failures",
    "studytime": "reports low weekly study time",
    "absences": "has a high number of absences",
    "Dalc": "reports elevated weekday alcohol consumption",
    "Walc": "reports elevated weekend alcohol consumption",
    "goout": "reports frequent socializing that may be cutting into study time",
    "famrel": "reports strained family relationships",
    "freetime": "reports a large amount of unstructured free time",
    "Medu": "comes from a household with lower parental education level",
    "Fedu": "comes from a household with lower parental education level",
    "Mjob": "parental occupation context",
    "health": "reports lower self-rated health",
}

SUGGESTION_BANK = {
    "G1": "Schedule a 1:1 review of recent assessment mistakes within the next week.",
    "G2": "Pair with a peer tutor or provide targeted practice on the weakest topic area.",
    "failures": "Enroll in a structured remedial/catch-up track for the subject previously failed.",
    "studytime": "Set up a weekly study plan with checkpoints; consider a study-skills workshop.",
    "absences": "Reach out to the student/guardian to understand the cause of absences and agree on an attendance plan.",
    "Dalc": "Flag discreetly to the school counselor for a wellbeing check-in.",
    "Walc": "Flag discreetly to the school counselor for a wellbeing check-in.",
    "goout": "Suggest a structured after-school study block to balance social time.",
    "famrel": "Loop in the school counselor for a supportive family conversation.",
    "freetime": "Encourage joining a structured extracurricular or study group.",
    "Medu": "Offer additional take-home learning resources suitable for self-study.",
    "Fedu": "Offer additional take-home learning resources suitable for self-study.",
    "health": "Recommend a check-in with the school nurse/counselor.",
}


def _offline_alert(student_id, risk_label, top_factors, student_row):
    """Deterministic, no-API-key fallback generator."""
    factor_names = [f for f, _ in top_factors[:3]]
    reasons = [TEMPLATE_PLAYBOOK.get(f, f"shows a notable signal in '{f}'") for f in factor_names]
    suggestions = []
    for f in factor_names:
        s = SUGGESTION_BANK.get(f)
        if s and s not in suggestions:
            suggestions.append(s)
    if not suggestions:
        suggestions = ["Schedule a general check-in to assess overall academic wellbeing."]

    explanation = (
        f"Student {student_id} is flagged as {risk_label} risk. "
        f"Main contributing signals: " + "; ".join(reasons) + "."
    )
    parent_note = (
        f"We wanted to check in about {student_id}'s recent progress. "
        f"A few areas could use extra support right now, and we'd love to work together "
        f"on a simple plan to help them get back on track."
    )

    return {
        "student_id": student_id,
        "risk_label": risk_label,
        "explanation": explanation,
        "suggestions": suggestions[:3],
        "parent_note": parent_note,
        "mode": "offline-template",
    }


def _build_prompt(student_id, risk_label, top_factors, student_row):
    factors_str = ", ".join([f"{name} (importance {score:.2f})" for name, score in top_factors[:5]])
    context = {
        "student_id": student_id,
        "risk_label": risk_label,
        "top_contributing_factors": factors_str,
        "study_time_level_1to4": int(student_row.get("studytime", -1)),
        "past_failures": int(student_row.get("failures", -1)),
        "absences": int(student_row.get("absences", -1)),
        "checkpoint_grade_1_of_20": int(student_row.get("G1", -1)),
        "checkpoint_grade_2_of_20": int(student_row.get("G2", -1)),
    }
    return f"""You are an assistant helping teachers interpret an early-warning system for student academic risk.

A machine learning model has flagged this student as **{risk_label} risk** of falling behind academically.
Here is the student's context:
{json.dumps(context, indent=2)}

Respond ONLY with valid JSON (no markdown fences, no preamble) in this exact shape:
{{
  "explanation": "2-3 sentence plain-language explanation of why this student was flagged, referencing the actual contributing factors",
  "suggestions": ["intervention 1", "intervention 2", "intervention 3"],
  "parent_note": "a short, warm, non-alarming message a teacher could send to a parent/guardian"
}}

Keep it concise, specific to the data given, and actionable. Do not invent information not present in the context."""


def _live_alert(student_id, risk_label, top_factors, student_row):
    """Calls the Anthropic API. Requires: pip install anthropic, and ANTHROPIC_API_KEY set."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    prompt = _build_prompt(student_id, risk_label, top_factors, student_row)

    response = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    # Guard against the model wrapping output in markdown fences despite instructions
    text = text.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(text)
    parsed["student_id"] = student_id
    parsed["risk_label"] = risk_label
    parsed["mode"] = "live-llm"
    return parsed


def generate_alert(student_id, risk_label, top_factors, student_row, force_offline=False):
    """
    student_id:   str
    risk_label:   "High" | "Medium" | "Low"
    top_factors:  list of (feature_name, importance_score) tuples, sorted desc
    student_row:  dict-like (pandas Series ok) of the student's raw feature values
    force_offline: if True, always use the template fallback (useful for testing/demo)
    """
    if risk_label == "Low":
        return {
            "student_id": student_id,
            "risk_label": risk_label,
            "explanation": f"Student {student_id} is on track. No intervention needed at this time.",
            "suggestions": [],
            "parent_note": None,
            "mode": "no-alert-needed",
        }

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if force_offline or not has_key:
        return _offline_alert(student_id, risk_label, top_factors, student_row)

    try:
        return _live_alert(student_id, risk_label, top_factors, student_row)
    except Exception as e:  # network issues, bad JSON, missing package, etc.
        fallback = _offline_alert(student_id, risk_label, top_factors, student_row)
        fallback["mode"] = f"offline-fallback (live call failed: {e})"
        return fallback
