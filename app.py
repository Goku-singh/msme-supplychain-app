import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

st.set_page_config(page_title="📊 MSME Forecast Dashboard", layout="wide")
st.title("MSME Forecasting and Analysis Dashboard")

# Upload section
uploaded_file = st.file_uploader("Upload your retail sales CSV", type="csv")
if not uploaded_file:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# User selection: Product category
product_category = st.selectbox("Select Product Category to Analyze:", df['Product Category'].unique())
filtered_df = df[df['Product Category'] == product_category].copy()

# Aggregate sales monthly
df_monthly = filtered_df.resample('M', on='Date').sum(numeric_only=True)
df_monthly = df_monthly[['Total Amount']]
df_monthly.columns = ['Sales']

# Show raw chart
st.subheader(f"📈 Monthly Sales Trend: {product_category}")
st.line_chart(df_monthly)

# Train-test split (last 3 months for testing)
train = df_monthly[:-3]
test = df_monthly[-3:]

# Forecast model selection
model_option = st.selectbox("Select Forecasting Model:", ["Holt-Winters", "ARIMA", "Prophet", "Random Forest", "XGBoost"])
forecast_months = st.slider("Select months to forecast:", 3, 12, 6)

# Forecasting logic
def forecast_with_model(model_name):
    if model_name == "Holt-Winters":
        model = ExponentialSmoothing(train['Sales'], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(forecast_months)
        return forecast

    elif model_name == "ARIMA":
        model = ARIMA(train['Sales'], order=(1,1,1))
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_months)
        return forecast

    elif model_name == "Prophet":
        prophet_df = train.reset_index().rename(columns={"Date": "ds", "Sales": "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_months, freq='M')
        forecast = model.predict(future)
        return forecast.set_index('ds')['yhat'].tail(forecast_months)

    elif model_name == "Random Forest":
        df_rf = train.copy()
        df_rf['month'] = np.arange(len(df_rf))
        X = df_rf[['month']]
        y = df_rf['Sales']
        model = RandomForestRegressor().fit(X, y)
        future = pd.DataFrame({'month': np.arange(len(df_rf), len(df_rf)+forecast_months)})
        forecast = model.predict(future)
        return pd.Series(forecast, index=pd.date_range(df_rf.index[-1]+timedelta(days=30), periods=forecast_months, freq='M'))

    elif model_name == "XGBoost":
        df_xgb = train.copy()
        df_xgb['month'] = np.arange(len(df_xgb))
        X = df_xgb[['month']]
        y = df_xgb['Sales']
        model = XGBRegressor(objective='reg:squarederror').fit(X, y)
        future = pd.DataFrame({'month': np.arange(len(df_xgb), len(df_xgb)+forecast_months)})
        forecast = model.predict(future)
        return pd.Series(forecast, index=pd.date_range(df_xgb.index[-1]+timedelta(days=30), periods=forecast_months, freq='M'))

forecast = forecast_with_model(model_option)

# Accuracy calculation (only for 3-month test data if applicable)
if len(test) >= 3:
    try:
        eval_forecast = forecast[:3]
        mape = mean_absolute_percentage_error(test['Sales'], eval_forecast) * 100
        rmse = np.sqrt(mean_squared_error(test['Sales'], eval_forecast))
        st.subheader("📌 Forecast Accuracy (on last 3 months of test data)")
        st.markdown(f"**MAPE:** {mape:.2f}%  ")
        st.markdown(f"**RMSE:** {rmse:.2f}")
    except:
        st.warning("Forecast accuracy could not be calculated for the selected model.")

# Forecast Chart
st.subheader(f"🔮 Sales Forecast for Next {forecast_months} Months")
fig, ax = plt.subplots(figsize=(10, 5))
train['Sales'].plot(ax=ax, label='Train')
test['Sales'].plot(ax=ax, label='Test', color='orange')
forecast.plot(ax=ax, label='Forecast', color='green')
plt.legend()
plt.title(f"{model_option} Forecast for {product_category}")
st.pyplot(fig)

# Extra charts
st.subheader("📊 Additional Insights")
col1, col2 = st.columns(2)

with col1:
    gender_plot = filtered_df.groupby('Gender')['Total Amount'].sum().plot.pie(autopct='%1.0f%%', title='Sales by Gender')
    st.pyplot(gender_plot.get_figure())

with col2:
    age_plot = sns.boxplot(data=filtered_df, x='Gender', y='Age')
    plt.title("Age Distribution by Gender")
    st.pyplot(age_plot.figure)

st.success("Dashboard generated successfully!")
