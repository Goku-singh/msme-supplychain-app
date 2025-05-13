import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

st.set_page_config(page_title="MSME Forecasting App", layout="wide")
st.title("📊 MSME Supply Chain Dashboard")

# Upload Section
uploaded_file = st.file_uploader("Upload your sales dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show preview
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Column selectors
    date_col = st.selectbox("Select Date column", df.columns)
    sales_col = st.selectbox("Select Sales/Revenue column", df.columns)
    product_col = st.selectbox("Select Product Category column (Optional)", [None] + list(df.columns))

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if product_col:
        product_selected = st.selectbox("Select a Product Category to Filter", df[product_col].unique())
        df = df[df[product_col] == product_selected]

    # Group by Date if necessary
    daily_data = df.groupby(date_col)[sales_col].sum().reset_index()
    daily_data.set_index(date_col, inplace=True)

    # Visualizations
    st.subheader("📊 Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Line Chart", "Bar Chart", "Boxplot"])

    with tab1:
        st.line_chart(daily_data)
    with tab2:
        st.bar_chart(daily_data)
    with tab3:
        fig, ax = plt.subplots()
        sns.boxplot(data=daily_data, y=sales_col, ax=ax)
        st.pyplot(fig)

    # Forecasting Model Selection
    st.subheader("🔮 Select Forecasting Model")
    model_option = st.selectbox("Choose Model", ["Holt-Winters", "ARIMA", "Prophet", "Random Forest", "XGBoost"])

    forecast_period = st.slider("Select forecast period (months)", 3, 12, 6)
    test_data = daily_data[-90:]  # Last 3 months as test

    def evaluate(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return mape, rmse

    st.subheader("📈 Forecast Results")

    if model_option == "Holt-Winters":
        try:
            model = ExponentialSmoothing(daily_data[sales_col], trend='add', seasonal='add', seasonal_periods=30)
            fit = model.fit()
            forecast = fit.forecast(forecast_period * 30)
            st.line_chart(forecast)
            mape, rmse = evaluate(test_data, fit.fittedvalues[-90:])
        except Exception as e:
            st.error(f"Holt-Winters Error: {e}")

    elif model_option == "ARIMA":
        try:
            model = ARIMA(daily_data[sales_col], order=(5,1,0))
            fit = model.fit()
            forecast = fit.forecast(steps=forecast_period * 30)
            st.line_chart(forecast)
            mape, rmse = evaluate(test_data, fit.fittedvalues[-90:])
        except Exception as e:
            st.error(f"ARIMA Error: {e}")

    elif model_option == "Prophet":
        try:
            prophet_df = daily_data.reset_index().rename(columns={date_col: 'ds', sales_col: 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_period * 30)
            forecast = model.predict(future)
            fig1 = model.plot(forecast)
            st.pyplot(fig1)
            pred = forecast.set_index('ds').loc[test_data.index]['yhat']
            mape, rmse = evaluate(test_data, pred)
        except Exception as e:
            st.error(f"Prophet Error: {e}")

    elif model_option == "Random Forest":
        try:
            df_rf = daily_data.copy()
            df_rf['day'] = np.arange(len(df_rf))
            model = RandomForestRegressor()
            model.fit(df_rf[['day']], df_rf[sales_col])
            future_days = np.arange(len(df_rf), len(df_rf) + forecast_period * 30)
            forecast = pd.Series(model.predict(future_days.reshape(-1, 1)),
                                 index=pd.date_range(df_rf.index[-1] + timedelta(days=1), periods=forecast_period * 30))
            st.line_chart(forecast)
            pred = model.predict(df_rf[['day']].tail(90))
            mape, rmse = evaluate(test_data, pred)
        except Exception as e:
            st.error(f"Random Forest Error: {e}")

    elif model_option == "XGBoost":
        try:
            df_xgb = daily_data.copy()
            df_xgb['day'] = np.arange(len(df_xgb))
            model = XGBRegressor()
            model.fit(df_xgb[['day']], df_xgb[sales_col])
            future_days = np.arange(len(df_xgb), len(df_xgb) + forecast_period * 30)
            forecast = pd.Series(model.predict(future_days.reshape(-1, 1)),
                                 index=pd.date_range(df_xgb.index[-1] + timedelta(days=1), periods=forecast_period * 30))
            st.line_chart(forecast)
            pred = model.predict(df_xgb[['day']].tail(90))
            mape, rmse = evaluate(test_data, pred)
        except Exception as e:
            st.error(f"XGBoost Error: {e}")

    st.subheader("📊 Forecast Accuracy (Last 3 Months)")
    st.write(f"**MAPE**: {mape:.2f}%")
    st.write(f"**RMSE**: {rmse:.2f}")
