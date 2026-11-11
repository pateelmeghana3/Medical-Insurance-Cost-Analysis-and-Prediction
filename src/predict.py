"""
Predict insurance charges for one person using the saved model.

Example:
    python src/predict.py --age 35 --sex male --bmi 28.5 --children 2 --smoker no --region southeast
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate medical insurance charges")
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--sex", choices=["male", "female"], required=True)
    ap.add_argument("--bmi", type=float, required=True)
    ap.add_argument("--children", type=int, required=True)
    ap.add_argument("--smoker", choices=["yes", "no"], required=True)
    ap.add_argument("--region", choices=["northeast", "northwest", "southeast", "southwest"], required=True)
    a = ap.parse_args()

    bundle = joblib.load(MODEL_PATH)
    row = pd.DataFrame([vars(a)])
    # same engineered features as training
    row["bmi_obese"] = int(a.bmi >= 30)
    row["smoker_x_bmi"] = int(a.smoker == "yes") * a.bmi
    row["age_sq"] = a.age ** 2

    pred = bundle["model"].predict(row[bundle["columns"]])[0]
    print(f"Estimated annual charges: {pred:,.2f}")


if __name__ == "__main__":
    main()
