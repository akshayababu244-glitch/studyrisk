# StudyRisk — AI Early Warning System for Student Learning Risk

**SDG Alignment:** SDG 4 — Quality Education (Target 4.1: effective learning outcomes; Target 4.5: reducing disparities)

## Problem Statement

Teachers often discover that a student is falling behind only after it's already
severe — by the time a report card or final exam reveals the problem, there's
little time left to intervene. Existing school systems store the data needed to
catch this earlier (attendance, checkpoint grades, study habits) but don't
*predict* risk or turn that prediction into *action*. StudyRisk closes that gap:
it predicts which students are at risk of falling behind using early
checkpoint data, then uses an LLM to turn that raw prediction into a
plain-language explanation, concrete intervention suggestions for the teacher,
and a draft note for parents — automating the "prediction → action" step that
most academic-analytics tools leave to busy humans.

## Architecture

```
student_data.csv
      │
      ▼
┌─────────────────────┐
│  Preprocessing       │  (src/preprocess.py)
│  - builds risk label │
│  - encodes features  │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  ML Risk Model        │  (src/train_model.py)
│  RandomForest          │  -> predicts Low / Medium / High risk
│  classifier             │  -> feature importances
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  LLM Alert Layer      │  (src/llm_alerts.py)
│  - explains WHY        │  -> Anthropic API (live) or
│  - suggests actions     │     template fallback (offline)
│  - drafts parent note   │
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Dashboard (app.py)   │  Streamlit UI for teachers
└─────────────────────┘
```

Why split ML (prediction) from LLM (communication)? Numeric risk scoring is
better handled by a traditional classifier — it's fast, cheap, and
explainable via feature importances. The LLM's job is turning that signal
into something a human can actually act on. This also means the project
demonstrates **both** classic ML and LLM-based automation, not just one.

## Project Structure

```
studyrisk/
├── data/
│   ├── generate_data.py      # synthetic dataset generator (no internet needed)
│   └── student_data.csv      # generated dataset (650 students)
├── src/
│   ├── preprocess.py         # risk labeling + feature encoding
│   ├── train_model.py        # trains & evaluates the RandomForest classifier
│   ├── llm_alerts.py         # LLM (+ offline fallback) alert generation
│   └── pipeline.py           # ties everything together; also runnable as CLI demo
├── models/                   # saved model + encoders (created by train_model.py)
├── app.py                    # Streamlit teacher dashboard
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) generate a fresh synthetic dataset — a default one is already included
python data/generate_data.py --n 650 --seed 42 --out data/student_data.csv

# 4. Train the risk model
python src/train_model.py --data data/student_data.csv

# 5. (Optional) enable live LLM alerts — without this, offline template alerts are used
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
export $(cat .env | xargs)      # or use python-dotenv / your shell's env loading
```

## Running It

**Quick CLI demo** (no Streamlit needed):
```bash
python src/pipeline.py --data data/student_data.csv --limit 5
# add --offline to force template alerts even if you have an API key set
```

**Full dashboard:**
```bash
streamlit run app.py
```
This opens a browser dashboard where you can filter students by risk level,
click into any flagged student, and see the AI-generated explanation,
suggested interventions, and a draft parent note.

## Using Real Data Instead of Synthetic

If you have access to the real UCI "Student Performance" dataset
(`student-mat.csv` / `student-por.csv`, Cortez & Silva 2008) or your own
institution's anonymized data, just place a CSV with the same column names
at `data/student_data.csv` (see `data/generate_data.py` for the exact column
list) and re-run steps 4 onward. No code changes needed.

## Evaluation (for your report)

- **Model metrics:** accuracy, macro F1, confusion matrix — printed by
  `train_model.py` (current synthetic run: ~80% accuracy, 0.78 macro F1).
- **Feature importance:** saved to `models/feature_importance.csv` — use
  this to discuss *what drives risk* in your report (in this dataset: early
  checkpoint grades, past failures, study time, and absences dominate, which
  matches published education-analytics literature).
- **LLM output quality:** since there's no ground truth for "good alert
  text," evaluate qualitatively — e.g. a small rubric scoring clarity,
  specificity, and actionability across a sample of generated alerts, or a
  short user study with 2-3 teachers reviewing sample outputs.

## Possible Extensions (if you want to expand scope)

- Swap the offline/live fallback for a small local SLM (e.g. Phi-3-mini,
  Llama-3.2-1B via Ollama) to demonstrate an on-device alternative to the
  cloud LLM call — useful if your institution restricts sending student
  data to external APIs.
- Add a scheduled job (cron/Celery) to simulate real-time weekly re-scoring
  as new checkpoint data arrives.
- Add an email/SMS delivery step for parent notes (e.g. via SendGrid/Twilio).
- Add a simple chat interface where a teacher can ask natural-language
  questions ("which students need help with attendance this month?") —
  a lightweight RAG-style query over the results table.

## Notes on the Dataset

The included `student_data.csv` is **synthetically generated** (see
`data/generate_data.py`) to mirror the structure and statistical patterns of
the real UCI Student Performance dataset, so the project runs fully offline
with no external downloads. Grades, absences, and risk correlations are
constructed with realistic relationships (e.g. low study time + past
failures + high absences → higher risk) but are not real student records.
For a real deployment or publication, replace with real (properly
anonymized/consented) institutional data.
