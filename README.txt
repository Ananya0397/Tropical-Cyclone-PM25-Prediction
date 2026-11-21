# Cyclone PM2.5 deployment dashboard

## Folder contents
- `app.py`: Streamlit dashboard; loads a saved model and predicts on unseen cyclone CSVs.
- `train_model.py`: trains the Linear Regression model once from the historical CSV and saves it.
- `Cyclone_Dashboard_Training_Data.csv`: your historical `complete_df` export (you provide this file).
- `requirements.txt`: required packages.

## Setup
1. Put your historical CSV in this folder. It must be named `Cyclone_Dashboard_Training_Data.csv`.
2. The historical CSV must include `Time`, `Cyclone`, `Cyclone_Phase`, `PM25`, `AOD`, `Wind_Speed`, `Pressure`, `Rainfall`, and `PM25_Next_Hour`.
3. Install dependencies:
   `python -m pip install -r requirements.txt`
4. Train and save the model ONCE:
   `python train_model.py`
5. Start the dashboard:
   `streamlit run app.py`

The saved model is `cyclone_pm25_linear_regression.joblib`. Do not upload a new cyclone into the training CSV. Upload it only through the dashboard's Unseen Cyclone Prediction page.

## New cyclone CSV
Required columns: `Time`, `PM25`, `AOD`, `Wind_Speed`, `Pressure`, `Rainfall`. `Cyclone` is recommended; if absent it is labelled "Uploaded cyclone". `Cyclone_Phase` is optional. If omitted, the app estimates phases from event progress; this is an approximation.

At least four valid hourly observations are needed to calculate current PM2.5 plus three lag values. Timestamps must be exactly one hour apart. Each output row is a one-step-ahead forecast. For historical evaluation, the next-hour actual PM2.5 can be joined separately after forecasting, but it is never an input.

## Important modeling note
The notebook's Phase 32 reconstructs lag columns with global shifts after sorting by time. This package deliberately calculates lags within each cyclone for the historical training CSV and within the uploaded cyclone for inference, avoiding cross-cyclone lag contamination. Train and inference use the same corrected lag logic. Re-run `train_model.py` once after replacing the historical CSV if you change the training data or feature engineering.
