import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "Cyclone_Dashboard_Training_Data.csv"
MODEL_FILE = ROOT / "cyclone_pm25_linear_regression.joblib"

st.set_page_config(
    page_title="Cyclone PM2.5 Prediction Dashboard",
    page_icon="🌪️",
    layout="wide"
)

st.title("🌪️ Tropical Cyclone PM2.5 Prediction Dashboard")
st.caption("Historical analysis and next-hour PM2.5 forecasting for unseen cyclones")

@st.cache_resource
def load_model(path, modified):
    return joblib.load(path)

@st.cache_data
def load_historical(path, modified):
    return pd.read_csv(path)

def normalize_phase(value):
    s = str(value).strip().lower().replace("_", " ")
    s = " ".join(s.split())
    mapping = {
        "1": "Phase 1", "phase1": "Phase 1", "phase 1": "Phase 1",
        "2": "Phase 2", "phase2": "Phase 2", "phase 2": "Phase 2",
        "3": "Phase 3", "phase3": "Phase 3", "phase 3": "Phase 3",
    }
    return mapping.get(s, np.nan)

def prepare_observations(raw, infer_phase=True):
    df = raw.copy()
    required = ["Time", "PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing: " + ", ".join(missing))

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    for c in ["PM25", "AOD", "Wind_Speed", "Pressure", "Rainfall"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Cyclone" not in df.columns:
        df["Cyclone"] = "Uploaded cyclone"

    df["Cyclone"] = df["Cyclone"].fillna("Uploaded cyclone").astype(str)
    df = df.dropna(subset=["Time"]).copy()
    df = df.sort_values(["Cyclone", "Time"]).drop_duplicates(
        subset=["Cyclone", "Time"], keep="last"
    ).reset_index(drop=True)

    # Enforce regular one-hour observations. Do not silently create missing PM2.5.
    if df.empty:
        raise ValueError("No valid timestamps were found.")

    df["Hour"] = df["Time"].dt.hour
    df["DayOfYear"] = df["Time"].dt.dayofyear
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    for lag in (1, 2, 3):
        df[f"PM25_Lag{lag}"] = df.groupby("Cyclone")["PM25"].shift(lag)

    if "Cyclone_Phase" in df.columns and df["Cyclone_Phase"].notna().any():
        phase = df["Cyclone_Phase"].map(normalize_phase)
        phase_source = "Cyclone_Phase supplied in uploaded file"
    else:
        if not infer_phase:
            raise ValueError("Cyclone_Phase is required.")
        phase = pd.Series(index=df.index, dtype="object")
        for cyclone, idx in df.groupby("Cyclone", sort=False).groups.items():
            n = len(idx)
            progress = np.arange(n) / max(n, 1)
            labels = np.select(
                [progress < 0.25, progress < 0.75],
                ["Phase 1", "Phase 2"],
                default="Phase 3"
            )
            phase.loc[idx] = labels
        phase_source = "Estimated from event progress (approximation)"

    for p in ["Phase 1", "Phase 2", "Phase 3"]:
        df[f"Phase_{p}"] = (phase == p).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df, phase_source

def cpcb_category(value):
    if value <= 30:
        return "Good"
    if value <= 60:
        return "Satisfactory"
    if value <= 90:
        return "Moderate"
    if value <= 120:
        return "Poor"
    if value <= 250:
        return "Very Poor"
    return "Severe"

if not MODEL_FILE.exists():
    st.error("Saved model not found. First run: python train_model.py")
    st.code("python train_model.py", language="bash")
    st.stop()

try:
    bundle = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
    model = bundle["model"]
    FEATURES = bundle["features"]
except Exception as e:
    st.error(f"Could not load saved model: {e}")
    st.stop()

if not DATA_FILE.exists():
    st.warning("Historical CSV is not present. Prediction can still work if the saved model exists, but historical analysis will be unavailable.")
    historical = None
else:
    try:
        historical = load_historical(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    except Exception as e:
        st.error(f"Could not read historical CSV: {e}")
        historical = None

st.sidebar.title("Dashboard Navigation")
pages = ["Dashboard Overview", "Historical Cyclone Analysis", "Unseen Cyclone Prediction", "Model Information"]
page = st.sidebar.radio("Select a section", pages)
st.sidebar.divider()
st.sidebar.write("**Deployment model:** Linear Regression")
st.sidebar.write(f"Training cyclones: {bundle['training_cyclone_count']}")
st.sidebar.write(f"Training rows: {bundle['training_rows']:,}")
st.sidebar.caption("Uploaded cyclone data is prediction-only; it is never added to model training.")

if page == "Dashboard Overview":
    st.header("Dashboard Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training cyclones", bundle["training_cyclone_count"])
    c2.metric("Training records", f"{bundle['training_rows']:,}")
    c3.metric("Model", "Linear Regression")
    st.info("The saved model is trained once from the historical cyclone dataset. Uploading a new cyclone does not retrain or modify it.")
    if historical is not None and "PM25" in historical.columns:
        hist_pm = pd.to_numeric(historical["PM25"], errors="coerce").dropna()
        a, b = st.columns(2)
        a.metric("Historical mean PM2.5", f"{hist_pm.mean():.2f}")
        a.metric("Historical maximum PM2.5", f"{hist_pm.max():.2f}")
        fig = px.histogram(historical, x="PM25", nbins=35, title="Historical PM2.5 Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Historical dataset preview")
        st.dataframe(historical.head(100), use_container_width=True)

elif page == "Historical Cyclone Analysis":
    st.header("Historical Cyclone Analysis")

    if historical is None or not {"Cyclone", "Time", "PM25"}.issubset(historical.columns):
        st.warning("Historical CSV must contain Cyclone, Time and PM25 columns.")

    else:
        hist = historical.copy()
        hist["Time"] = pd.to_datetime(hist["Time"], errors="coerce")
        hist = hist.dropna(subset=["Time"])

        names = sorted(hist["Cyclone"].dropna().astype(str).unique())
        selected = st.selectbox("Select a historical cyclone", names)

        part = hist[hist["Cyclone"].astype(str) == selected].sort_values("Time")

        if part.empty:
            st.warning("No historical data available for this cyclone.")

        else:
            st.subheader("Select Time Period")

            min_time = part["Time"].min().to_pydatetime()
            max_time = part["Time"].max().to_pydatetime()

            col1, col2 = st.columns(2)

            with col1:
                start_time = st.datetime_input(
                    "Start Date and Time",
                    value=min_time,
                    min_value=min_time,
                    max_value=max_time
                )

            with col2:
                end_time = st.datetime_input(
                    "End Date and Time",
                    value=max_time,
                    min_value=min_time,
                    max_value=max_time
                )

            if start_time > end_time:
                st.error("Start time must be earlier than end time.")

            else:
                filtered = part[
                    (part["Time"] >= pd.Timestamp(start_time)) &
                    (part["Time"] <= pd.Timestamp(end_time))
                ].copy()

                if filtered.empty:
                    st.warning("No data available for the selected time period.")

                else:
                    st.success(f"Showing {len(filtered)} records for the selected period.")

                    # PM2.5 Analysis
                    st.subheader("PM2.5 Concentration")

                    fig = px.line(
                        filtered,
                        x="Time",
                        y="PM25",
                        markers=True,
                        title=f"PM2.5 Trend — {selected}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    cols = st.columns(3)
                    vals = pd.to_numeric(filtered["PM25"], errors="coerce")

                    cols[0].metric("Mean PM2.5", f"{vals.mean():.2f}")
                    cols[1].metric("Maximum PM2.5", f"{vals.max():.2f}")
                    cols[2].metric("Minimum PM2.5", f"{vals.min():.2f}")

                    # Meteorological Measurements
                    st.subheader("Meteorological Measurements")

                    if "Wind_Speed" in filtered.columns:
                        fig_wind = px.line(
                            filtered,
                            x="Time",
                            y="Wind_Speed",
                            title="Wind Speed Variation",
                            markers=True
                        )
                        fig_wind.update_layout(
                            yaxis_title="Wind Speed",
                            xaxis_title="Time"
                        )
                        st.plotly_chart(fig_wind, use_container_width=True)

                    if "Pressure" in filtered.columns:
                        fig_pressure = px.line(
                            filtered,
                            x="Time",
                            y="Pressure",
                            title="Atmospheric Pressure Variation",
                            markers=True
                        )
                        fig_pressure.update_layout(
                            yaxis_title="Pressure",
                            xaxis_title="Time"
                        )
                        st.plotly_chart(fig_pressure, use_container_width=True)

                    if "Rainfall" in filtered.columns:
                        fig_rain = px.line(
                            filtered,
                            x="Time",
                            y="Rainfall",
                            title="Rainfall Variation",
                            markers=True
                        )
                        fig_rain.update_layout(
                            yaxis_title="Rainfall",
                            xaxis_title="Time"
                        )
                        st.plotly_chart(fig_rain, use_container_width=True)

                    # Historical Data Table
                    st.subheader("Historical Cyclone Data Table")
                    st.dataframe(filtered, use_container_width=True)
                    
elif page == "Unseen Cyclone Prediction":
    st.header("Unseen Cyclone — Next-Hour PM2.5 Prediction")
    st.write("Upload observations for a cyclone that was not part of the historical training dataset.")
    st.warning("The file must contain observed PM2.5 up to the latest available hour. Do not include or provide future PM2.5 as an input.")
    uploaded = st.file_uploader("Upload new cyclone CSV", type=["csv"], key="new_cyclone_upload")
    if uploaded is not None:
        try:
            raw_new = pd.read_csv(uploaded)
            prepared, phase_source = prepare_observations(raw_new)
            cyclone_names = prepared["Cyclone"].dropna().unique()
            if len(cyclone_names) != 1:
                st.error("Upload exactly one cyclone per CSV file.")
            else:
                cyclone_name = str(cyclone_names[0])
                if cyclone_name.casefold() in {x.casefold() for x in bundle["training_cyclones"]}:
                    st.error("This cyclone name already exists in the training dataset. Upload a genuinely unseen cyclone.")
                else:
                    # Check chronological gaps; model forecasts one hour ahead.
                    times = prepared["Time"].sort_values().drop_duplicates()
                    gaps = times.diff().dropna()
                    if not gaps.empty and not (gaps == pd.Timedelta(hours=1)).all():
                        st.error("The timestamps are not consistently one hour apart. Correct/resample the time series before forecasting; this model is for one-hour intervals.")
                    else:
                        valid = prepared.dropna(subset=FEATURES).copy()
                        if valid.empty:
                            st.error("At least four valid hourly observations are required to calculate PM25 and its three lag values.")
                        else:
                            X_new = valid[FEATURES].apply(pd.to_numeric, errors="coerce")
                            valid = valid.loc[X_new.notna().all(axis=1)].copy()
                            X_new = X_new.loc[valid.index]
                            if valid.empty:
                                st.error("Required model inputs contain missing or non-numeric values.")
                            else:
                                pred = np.maximum(model.predict(X_new), 0)
                                results = pd.DataFrame({
                                    "Cyclone": cyclone_name,
                                    "Current_Time": valid["Time"].values,
                                    "Forecast_Time": (valid["Time"] + pd.Timedelta(hours=1)).values,
                                    "Current_PM25": valid["PM25"].values,
                                    "Predicted_Next_Hour_PM25": pred,
                                    "Predicted_Category": [cpcb_category(v) for v in pred]
                                })


# =====================================================
# ACTUAL VS PREDICTED ACCURACY TEST FOR UNSEEN CYCLONE
# =====================================================

# Get actual PM2.5 measurements from the uploaded CSV
                                actual_data = prepared[
                                 ["Time", "PM25"]
                                ].copy()

                                actual_data = actual_data.rename(
                                    columns={
                                        "Time": "Forecast_Time",
                                        "PM25": "Actual_Next_Hour_PM25"
                                    }
        )

# Match the prediction with the actual measurement
                                evaluation = results.merge(
                                    actual_data,
                                    on="Forecast_Time",
                                    how="inner"
                    )

# Calculate prediction errors
                                if not evaluation.empty:

                                    evaluation["Error"] = (
                                        evaluation["Actual_Next_Hour_PM25"]
        -                               evaluation["Predicted_Next_Hour_PM25"]
                                )

                                    evaluation["Absolute_Error"] = (
                                        evaluation["Error"].abs()
            )

                                    evaluation["Squared_Error"] = (
                                        evaluation["Error"] ** 2
    )

    # Accuracy metrics
                                    mae = evaluation["Absolute_Error"].mean()

                                    rmse = np.sqrt(
                                        evaluation["Squared_Error"].mean()
    )

                                    st.divider()

                                    st.header("Unseen Cyclone Accuracy Evaluation")

                                    st.caption(
                                        "Retrospective evaluation: Ditwah was not used "
                                        "to train the historical model."
        )

                                    c1, c2, c3 = st.columns(3)

                                    c1.metric(
                                        "Test MAE",
                                        f"{mae:.4f}"
    )

                                    c2.metric(
                                        "Test RMSE",
                                        f"{rmse:.4f}"
    )

                                    c3.metric(
                                        "Evaluated Hours",
                                        len(evaluation)
    )

    # Actual vs predicted graph
                                    st.subheader("Actual vs Predicted PM2.5")

                                    fig_accuracy = px.line(
                                        evaluation,
                                        x="Forecast_Time",
                                        y=[
                                            "Actual_Next_Hour_PM25",
                                            "Predicted_Next_Hour_PM25"
        ],
                                        markers=True,
                                        title="Ditwah: Actual vs Predicted PM2.5"
    )

                                    st.plotly_chart(
                                        fig_accuracy,
                                        use_container_width=True
    )

    # Detailed comparison
                                    st.subheader("Prediction Error Details")

                                    st.dataframe(
                                        evaluation,
                                        use_container_width=True
    )

    # Download evaluation
                                    st.download_button(
                                        "Download Accuracy Evaluation",
                                        evaluation.to_csv(index=False).encode("utf-8"),
                                        file_name="Ditwah_Accuracy_Evaluation.csv",
                                        mime="text/csv"
    )

                                else:

                                    st.warning(
                                        "No matching actual next-hour observations found."
    )


                                st.success(f"Forecasts generated for unseen cyclone: {cyclone_name}")
                                st.caption(f"Model trained on {bundle['training_cyclone_count']} historical cyclones. Uploaded cyclone was not used for training.")
                                st.caption(f"Phase feature source: {phase_source}")
                                latest = results.iloc[-1]
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Latest observed PM2.5", f"{latest['Current_PM25']:.2f}")
                                c2.metric("Next-hour forecast", f"{latest['Predicted_Next_Hour_PM25']:.2f}")
                                c3.metric("Predicted category", latest["Predicted_Category"])
                                st.write(f"Current observation: **{latest['Current_Time']}**")
                                st.write(f"Forecast time: **{latest['Forecast_Time']}**")
                                st.plotly_chart(
                                    px.line(results, x="Forecast_Time", y="Predicted_Next_Hour_PM25", markers=True, title="Next-Hour PM2.5 Forecasts"),
                                    use_container_width=True
                                )
                                st.subheader("Forecast results")
                                st.dataframe(results, use_container_width=True)
                                st.subheader("Actual vs Predicted Values")

                                st.dataframe(
                                    evaluation[
                                        [
                                            "Forecast_Time",
                                            "Actual_Next_Hour_PM25",
                                            "Predicted_Next_Hour_PM25",
                                            "Error",
                                            "Absolute_Error"
                                        ]
                                    ],

    use_container_width=True
)
                                st.download_button(
                                    "Download forecast CSV",
                                    results.to_csv(index=False).encode("utf-8"),
                                    file_name=f"{cyclone_name}_PM25_next_hour_forecasts.csv",
                                    mime="text/csv"
                                )
                                st.info("Each row is a one-step-ahead forecast using the PM2.5 observation at that row and the three preceding observed PM2.5 values. For later rows, the CSV supplies those subsequent observations. This is not a multi-hour forecast made from only the first observation.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Information":
    st.header("Saved Model Information")
    st.write("The dashboard loads a saved Linear Regression model. It does not train a new model when an unseen cyclone CSV is uploaded.")
    st.json({
        "model": "Linear Regression",
        "training_rows": bundle["training_rows"],
        "training_cyclone_count": bundle["training_cyclone_count"],
        "training_cyclones": bundle["training_cyclones"],
        "features": FEATURES,
        "lag_handling": bundle["lag_note"],
        "phase_handling": bundle["phase_note"]
    })
    st.caption("The model is retrained only when you manually run train_model.py again after updating the historical training dataset.")
