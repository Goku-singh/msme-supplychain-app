import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

st.set_page_config(page_title="📈 MSME Forecast App", layout="wide")
st.title("📦 MSME Supply Chain Forecasting Dashboard")

uploaded_file = st.file_uploader("Upload your retail sales dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
    
    st.subheader("Step 1: Column Mapping")
    st.write(df.head())
    
    date_col = st.selectbox("Select Date Column", df.columns)
    sales_col = st.selectbox("Select Sales/Revenue Column", df.columns)
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df = df[[date_col, sales_col]].rename(columns={date_col: "ds", sales_col: "y"})

    # Split into train and test (last 3 months for test)
    forecast_horizon = 3
    train_df = df[:-forecast_horizon]
    test_df = df[-forecast_horizon:]

    st.subheader("Step 2: Choose Forecasting Model")
    model_choice = st.selectbox("Select a model", ["Holt-Winters", "ARIMA", "Prophet", "Random Forest", "XGBoost"])

    st.subheader("Step 3: Forecast & Evaluation")
    forecast_period = st.slider("Forecast how many months?", 3, 12, 6)

    future_dates = pd.date_range(df["ds"].max(), periods=forecast_period + 1, freq='MS')[1:]

    if model_choice == "Holt-Winters":
        model = ExponentialSmoothing(train_df['y'], trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(forecast_period)
        predictions = model.forecast(len(test_df))
        
    elif model_choice == "ARIMA":
        model = ARIMA(train_df['y'], order=(1,1,1)).fit()
        forecast = model.forecast(steps=forecast_period)
        predictions = model.forecast(steps=len(test_df))

    elif model_choice == "Prophet":
        prophet_model = Prophet()
        prophet_model.fit(train_df)
        future = prophet_model.make_future_dataframe(periods=forecast_period, freq='MS')
        forecast_df = prophet_model.predict(future)
        forecast = forecast_df[['ds', 'yhat']].tail(forecast_period).set_index('ds')['yhat']
        predictions = prophet_model.predict(df[['ds']].tail(forecast_horizon))[["yhat"]].values.flatten()

    elif model_choice == "Random Forest":
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        features = ['month', 'year']
        
        X_train = train_df.copy()
        X_train['month'] = X_train['ds'].dt.month
        X_train['year'] = X_train['ds'].dt.year
        y_train = X_train['y']
        X_train = X_train[features]

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)

        future_df = pd.DataFrame({'ds': future_dates})
        future_df['month'] = future_df['ds'].dt.month
        future_df['year'] = future_df['ds'].dt.year
        forecast = rf.predict(future_df[features])

        test_feat = test_df.copy()
        test_feat['month'] = test_feat['ds'].dt.month
        test_feat['year'] = test_feat['ds'].dt.year
        predictions = rf.predict(test_feat[features])

    elif model_choice == "XGBoost":
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        features = ['month', 'year']

        X_train = train_df.copy()
        X_train['month'] = X_train['ds'].dt.month
        X_train['year'] = X_train['ds'].dt.year
        y_train = X_train['y']
        X_train = X_train[features]

        model = XGBRegressor()
        model.fit(X_train, y_train)

        future_df = pd.DataFrame({'ds': future_dates})
        future_df['month'] = future_df['ds'].dt.month
        future_df['year'] = future_df['ds'].dt.year
        forecast = model.predict(future_df[features])

        test_feat = test_df.copy()
        test_feat['month'] = test_feat['ds'].dt.month
        test_feat['year'] = test_feat['ds'].dt.year
        predictions = model.predict(test_feat[features])

    st.write("### 📉 Forecast Results")
    result_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    st.line_chart(result_df.set_index('Date'))

    st.write("### 📊 Forecast Accuracy (on last 3 months)")
    rmse = np.sqrt(mean_squared_error(test_df['y'], predictions))
    mape = mean_absolute_percentage_error(test_df['y'], predictions)

    st.metric("MAPE (%)", f"{mape*100:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")
