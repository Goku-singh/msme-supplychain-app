import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MSME Forecasting Dashboard", layout="wide")
st.title("📈 MSME Supply Chain Forecasting & Analysis Tool")

# Upload the data
uploaded_file = st.file_uploader("Upload your sales Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Step 1: Data Mapping and Filtering")
    st.write("Preview of dataset:", df.head())

    date_col = st.selectbox("Select Date Column", df.columns)
    sales_col = st.selectbox("Select Sales Column", df.columns)
    product_col = st.selectbox("Select Product Category Column (optional)", ["None"] + list(df.columns))

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if product_col != "None":
        product_selected = st.selectbox("Choose Product Category to Analyze", df[product_col].unique())
        df = df[df[product_col] == product_selected]

    df_grouped = df.groupby(date_col)[sales_col].sum().reset_index()
    df_grouped.columns = ["ds", "y"]

    st.write("Aggregated Sales Data:", df_grouped.tail())

    forecast_horizon = st.slider("Select Forecast Horizon (months)", 3, 12, 6)
    model_type = st.selectbox("Select Forecasting Model", ["Holt-Winters", "ARIMA", "Prophet", "Random Forest", "XGBoost"])

    forecast = None
    mape = None
    rmse = None

    # Splitting into train and test
    df_grouped = df_grouped.set_index("ds").resample("M").sum().reset_index()
    train = df_grouped[:-3]
    test = df_grouped[-3:]

    st.subheader("Step 2: Forecast Results")

    try:
        if model_type == "Holt-Winters":
            model = ExponentialSmoothing(train["y"], trend="add", seasonal="add", seasonal_periods=12)
            fit = model.fit()
            forecast = fit.forecast(forecast_horizon)

        elif model_type == "Prophet":
            prophet_df = train.rename(columns={"ds": "ds", "y": "y"})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_horizon, freq='M')
            forecast_df = model.predict(future)
            forecast = forecast_df.set_index("ds")["yhat"].tail(forecast_horizon)

        elif model_type == "Random Forest":
            train_rf = train.copy()
            train_rf['month'] = train_rf['ds'].dt.month
            X_train = train_rf[['month']]
            y_train = train_rf['y']

            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)

            future_months = [(train_rf['ds'].max() + pd.DateOffset(months=i)).month for i in range(1, forecast_horizon+1)]
            forecast = pd.Series(rf.predict(np.array(future_months).reshape(-1, 1)))

        elif model_type == "XGBoost":
            train_xgb = train.copy()
            train_xgb['month'] = train_xgb['ds'].dt.month
            X_train = train_xgb[['month']]
            y_train = train_xgb['y']

            xgb = XGBRegressor(objective="reg:squarederror")
            xgb.fit(X_train, y_train)

            future_months = [(train_xgb['ds'].max() + pd.DateOffset(months=i)).month for i in range(1, forecast_horizon+1)]
            forecast = pd.Series(xgb.predict(np.array(future_months).reshape(-1, 1)))

        if forecast is not None:
            forecast.index = pd.date_range(start=train["ds"].max() + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
            forecast_df = pd.DataFrame({"Date": forecast.index, "Forecast": forecast.values})

            fig1 = px.line(df_grouped, x="ds", y="y", title="Historical Sales")
            fig1.add_scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode='lines', name='Forecast')
            st.plotly_chart(fig1, use_container_width=True)

            # Accuracy metrics on last 3 months
            if len(test) >= 3:
                actual = test["y"].values
                predicted = forecast.head(3).values
                mape = mean_absolute_percentage_error(actual, predicted) * 100
                rmse = mean_squared_error(actual, predicted, squared=False)

                st.subheader("Forecast Accuracy (Test Data: Last 3 Months)")
                if mape is not None and not np.isnan(mape):
                    st.write(f"**MAPE**: {mape:.2f}%")
                else:
                    st.warning("MAPE could not be calculated.")

                if rmse is not None and not np.isnan(rmse):
                    st.write(f"**RMSE**: {rmse:.2f}")
                else:
                    st.warning("RMSE could not be calculated.")

            st.subheader("Forecast Table")
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Model error: {e}")

    # Additional Analysis
    st.subheader("Step 3: Additional Visualizations")
    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.box(df, x=product_col if product_col != "None" else sales_col, y=sales_col, title="Boxplot Analysis")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(df, x=sales_col, nbins=30, title="Sales Distribution")
        st.plotly_chart(fig3, use_container_width=True)
