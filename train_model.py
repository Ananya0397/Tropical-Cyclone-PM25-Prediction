"""
Train and save the deployment Linear Regression model.

Run this script ONCE whenever the historical training CSV is updated.
The Streamlit app loads the saved model and does NOT retrain on uploads.

Historical CSV must be an export of complete_df and contain:
Time, Cyclone, Cyclone_Phase, PM25, AOD, Wind_Speed, Pressure,
Rainfall, PM25_Next_Hour.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Cyclone_Dashboard_Training_Data.csv"
MODEL_FILE = ROOT / "cyclone_pm25_linear_regression.joblib"

FEATURES = [
    "PM25", "PM25_Lag1", "PM25_Lag2", "PM25_Lag3",
    "Hour", "DayOfYear", "Hour_Sin", "Hour_Cos",
    "AOD", "Wind_Speed", "Pressure", "Rainfall",
    "Phase_Phase 1", "Phase_Phase 2", "Phase_Phase 3"
]
TARGET = "PM25_Next_Hour"
RAW_REQUIRED = [
    "Time", "Cyclone", "Cyclone_Phase", "PM25",
    "AOD", "Wind_Speed", "Pressure", "Rainfall", TARGET
]
NUMERIC = ["PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall", TARGET]

def normalize_phase(value):
    s = str(value).strip().lower().replace("_", " ")
    s = " ".join(s.split())
    mapping = {
        "1": "Phase 1", "phase1": "Phase 1", "phase 1": "Phase 1",
        "2": "Phase 2", "phase2": "Phase 2", "phase 2": "Phase 2",
        "3": "Phase 3", "phase3": "Phase 3", "phase 3": "Phase 3",
    }
    return mapping.get(s, np.nan)

def build_features(raw, group_lags=True, infer_phase=False):
    df = raw.copy()
    missing = [c for c in ["Time", "Cyclone", "PM25", "AOD",
                           "Wind_Speed", "Pressure", "Rainfall"] if c not in df]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    for c in ["PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Time", "Cyclone"]).copy()
    df["Cyclone"] = df["Cyclone"].astype(str)
    df = df.sort_values(["Cyclone", "Time"]).reset_index(drop=True)

    # Lag values are calculated within each cyclone so one cyclone's
    # measurements can never become another cyclone's lag inputs.
    groups = df.groupby("Cyclone", sort=False)["PM25"]
    for lag in (1, 2, 3):
        df[f"PM25_Lag{lag}"] = groups.shift(lag)

    df["Hour"] = df["Time"].dt.hour
    df["DayOfYear"] = df["Time"].dt.dayofyear
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    if "Cyclone_Phase" in df.columns and df["Cyclone_Phase"].notna().any():
        phase = df["Cyclone_Phase"].map(normalize_phase)
    elif infer_phase:
        # Fallback only for new cyclone files that do not include phase.
        # This is an estimate based on event progress, not an observed phase.
        phase = pd.Series(index=df.index, dtype="object")
        for cyclone, idx in df.groupby("Cyclone", sort=False).groups.items():
            n = len(idx)
            ranks = np.arange(n) / max(n, 1)
            # Notebook project phases are event-stage labels. This fallback
            # partitions an unseen event into three approximate stages.
            labels = np.select(
                [ranks < 0.25, ranks < 0.75],
                ["Phase 1", "Phase 2"],
                default="Phase 3"
            )
            phase.loc[idx] = labels
    else:
        raise ValueError("Historical training CSV must contain Cyclone_Phase.")

    for p in ["Phase 1", "Phase 2", "Phase 3"]:
        df[f"Phase_{p}"] = (phase == p).astype(int)

    return df

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE.name}. Put it beside train_model.py."
        )

    raw = pd.read_csv(DATA_FILE)
    missing = [c for c in RAW_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(
            "Historical CSV is missing required columns: " + ", ".join(missing)
        )

    data = build_features(raw, group_lags=True, infer_phase=False)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    for c in FEATURES:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURES + [TARGET]).copy()

    if len(data) < 20:
        raise ValueError(f"Only {len(data)} complete rows available; need more data.")
    if data["Cyclone"].nunique() < 2:
        raise ValueError("Historical training data should contain multiple cyclones.")

    model = LinearRegression()
    model.fit(data[FEATURES], data[TARGET])

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "training_rows": int(len(data)),
        "training_cyclones": sorted(data["Cyclone"].unique().tolist()),
        "training_cyclone_count": int(data["Cyclone"].nunique()),
        "training_time_min": str(data["Time"].min()),
        "training_time_max": str(data["Time"].max()),
        "phase_note": "Uses Cyclone_Phase from historical data; uploaded files without phase use approximate event-progress phases.",
        "lag_note": "PM25 lag features are calculated within each cyclone."
    }
    joblib.dump(bundle, MODEL_FILE)

    print("MODEL TRAINED AND SAVED")
    print(f"Model file: {MODEL_FILE.name}")
    print(f"Training rows: {len(data):,}")
    print(f"Training cyclones: {data['Cyclone'].nunique()}")
    print("Model: Linear Regression")
    print(f"Features: {len(FEATURES)}")
    print("The Streamlit app will load this saved model; it will not retrain on uploaded cyclone files.")

if __name__ == "__main__":
    main()
