"""
generate_data.py
-----------------
Generates a synthetic student performance dataset that mirrors the structure
of the well-known UCI "Student Performance" dataset (Cortez & Silva, 2008).

Why synthetic data?
If you have access to the real dataset (student-mat.csv / student-por.csv from
UCI/Kaggle), just drop it into data/student_data.csv with the same column names
and skip running this script. This generator exists so the project runs
end-to-end with zero external downloads, and so you can regenerate a fresh
dataset anytime (e.g. with a different size or random seed) for testing.

Usage:
    python generate_data.py --n 650 --seed 42 --out student_data.csv
"""

import argparse
import numpy as np
import pandas as pd


def generate(n: int = 650, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    school = rng.choice(["GP", "MS"], n, p=[0.7, 0.3])
    sex = rng.choice(["F", "M"], n)
    age = rng.integers(15, 20, n)
    address = rng.choice(["U", "R"], n, p=[0.75, 0.25])
    famsize = rng.choice(["LE3", "GT3"], n, p=[0.4, 0.6])
    pstatus = rng.choice(["T", "A"], n, p=[0.9, 0.1])
    medu = rng.integers(0, 5, n)
    fedu = rng.integers(0, 5, n)
    mjob = rng.choice(["teacher", "health", "services", "at_home", "other"], n)
    fjob = rng.choice(["teacher", "health", "services", "at_home", "other"], n)
    reason = rng.choice(["home", "reputation", "course", "other"], n)
    guardian = rng.choice(["mother", "father", "other"], n, p=[0.65, 0.25, 0.10])
    traveltime = rng.integers(1, 5, n)
    studytime = rng.integers(1, 5, n)
    failures = rng.choice([0, 0, 0, 1, 2, 3], n)  # skewed toward 0
    schoolsup = rng.choice(["yes", "no"], n, p=[0.15, 0.85])
    famsup = rng.choice(["yes", "no"], n, p=[0.6, 0.4])
    paid = rng.choice(["yes", "no"], n, p=[0.4, 0.6])
    activities = rng.choice(["yes", "no"], n)
    nursery = rng.choice(["yes", "no"], n, p=[0.8, 0.2])
    higher = rng.choice(["yes", "no"], n, p=[0.9, 0.1])
    internet = rng.choice(["yes", "no"], n, p=[0.8, 0.2])
    romantic = rng.choice(["yes", "no"], n, p=[0.35, 0.65])
    famrel = rng.integers(1, 6, n)
    freetime = rng.integers(1, 6, n)
    goout = rng.integers(1, 6, n)
    dalc = rng.integers(1, 6, n)   # workday alcohol consumption
    walc = rng.integers(1, 6, n)   # weekend alcohol consumption
    health = rng.integers(1, 6, n)
    absences = rng.poisson(6, n)
    absences = np.clip(absences, 0, 40)

    # --- Build grades with realistic correlations ---
    # Base ability signal (latent), partly driven by parental education & study habits
    ability = (
        0.6 * (medu + fedu) / 2
        + 0.8 * studytime
        - 1.3 * failures
        - 0.07 * absences
        - 0.3 * dalc
        - 0.2 * walc
        + rng.normal(0, 2.2, n)
    )
    # Scale to 0-20 grading range (Portuguese system, as in the original dataset)
    g1 = np.clip(10 + ability + rng.normal(0, 1.5, n), 0, 20).round().astype(int)
    g2 = np.clip(g1 + rng.normal(0, 1.8, n) - 0.3 * failures, 0, 20).round().astype(int)
    g3 = np.clip(0.5 * g1 + 0.5 * g2 + rng.normal(0, 1.5, n), 0, 20).round().astype(int)

    df = pd.DataFrame({
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": pstatus, "Medu": medu, "Fedu": fedu,
        "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
        "health": health, "absences": absences, "G1": g1, "G2": g2, "G3": g3,
    })

    df.insert(0, "student_id", [f"S{idx:04d}" for idx in range(1, n + 1)])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="student_data.csv")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} student records -> {args.out}")
    print(df[["student_id", "studytime", "failures", "absences", "G1", "G2", "G3"]].head())
