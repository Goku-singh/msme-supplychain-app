import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from prophet import Prophet

# Streamlit configuration
st.set_page_config(page_title="MSME Forecasting Tool", layout="wide")
st.title("📈 MSME Supply Chain Forecasting Tool")

# File upload functionality
uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "csv"])

if uploaded_file:
    # Load the dataset based on file type
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show the first few rows to preview the data
    st.subheader("Step 1: Column Mapping and Data Preview")
    st.write(df.head())

    # Let the user select the Date and Sales/Order columns
    date_col = st.selectbox("Select the Date Column", df.columns)
    sales_col = st.selectbox("Select the Sales/Order Column", df.columns)

    # Convert the selected date column to datetime format and sort the data by date
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    # Resample data to monthly sales
    monthly_data = df.resample('M')[sales_col].sum().fillna(0)

    # Let the user choose a model for forecasting
    st.subheader("Step 2: Select Forecasting Model")
    model_choice = st.selectbox("Choose Forecasting Model", 
                                ["ARIMA", "Holt-Winters", "Prophet", "Random Forest", "XGBoost"])

    # Let the user select the forecast horizon (next 3 to 12 months)
    forecast_period = st.slider("Select Forecast Horizon (Months)", 3, 12, 6)

    # Split the data into train and test (last 3 months for evaluation)
    train = monthly_data[:-3]
    test = monthly_data[-3:]

    forecast = None

    # ARIMA model
    if model_choice == "ARIMA":
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_period)

    # Holt-Winters model
    elif model_choice == "Holt-Winters":
        try:
            model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
            fit = model.fit()
            forecast = fit.forecast(forecast_period)
        except:
            st.error("Holt-Winters needs at least 2 seasonal cycles. Use ARIMA or ML models instead.")

    # Prophet model
    elif model_choice == "Prophet":
        prophet_df = monthly_data.reset_index()
        prophet_df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast_prophet = model.predict(future)
        forecast = forecast_prophet.set_index('ds')['yhat'].tail(forecast_period)

    # Machine Learning models (Random Forest, XGBoost)
    elif model_choice in ["Random Forest", "XGBoost"]:
        df_ml = monthly_data.reset_index()
        df_ml['month'] = df_ml[date_col].dt.month
        df_ml['year'] = df_ml[date_col].dt.year
        df_ml['lag1'] = df_ml[sales_col].shift(1)
        df_ml.dropna(inplace=True)

        X = df_ml[['month', 'year', 'lag1']]
        y = df_ml[sales_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if model_choice == "Random Forest":
            model = RandomForestRegressor()
        else:
            model = XGBRegressor(objective='reg:squarederror')

        model.fit(X_train, y_train)
        future_months = pd.date_range(start=monthly_data.index[-1], periods=forecast_period+1, freq='M')[1:]
        future_df = pd.DataFrame()
        future_df['month'] = future_months.month
        future_df['year'] = future_months.year
        future_df['lag1'] = [monthly_data[-1]] * forecast_period

        forecast = pd.Series(model.predict(future_df), index=future_months)

    # Display the forecast
    if forecast is not None:
        st.subheader("Step 3: Forecast Result")
        st.line_chart(pd.concat([monthly_data, forecast]))

        # Compute and display forecast accuracy (MAPE and RMSE) on the last 3 months
        if len(test) >= 3 and forecast_period >= 3:
            aligned = forecast[:3]
            aligned.index = test.index
            rmse = mean_squared_error(test, aligned, squared=False)
            mape = mean_absolute_percentage_error(test, aligned) * 100

            st.subheader("📌 Forecast Accuracy (Test Data: Last 3 Months)")
            st.markdown(f"**MAPE:** {mape:.2f}%")
            st.markdown(f"**RMSE:** {rmse:.2f}")
