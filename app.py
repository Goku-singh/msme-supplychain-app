import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import datetime
import warnings
warnings.filterwarnings("ignore")

# --- Sidebar ---
st.sidebar.title("Sales Forecasting Dashboard")
st.sidebar.markdown("Upload your 2-year sales dataset")

# File upload
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    st.title("📊 Sales Dashboard with Forecasting & ABC Analysis")
    st.markdown("---")

    # Product Category Filter
    selected_product = st.sidebar.selectbox("Select Product Category", options=df['Product Category'].unique())
    product_df = df[df['Product Category'] == selected_product]

    # Monthly Sales Aggregation
    sales_data = product_df.groupby(pd.Grouper(key='Date', freq='M')).agg({'Total Amount':'sum'}).reset_index()
    sales_data.columns = ['ds', 'y']

    # ABC Analysis
    st.subheader("🔍 ABC Analysis")
    abc_df = df.groupby('Product Category').agg({'Total Amount':'sum'}).sort_values(by='Total Amount', ascending=False)
    abc_df['% Contribution'] = 100 * abc_df['Total Amount'] / abc_df['Total Amount'].sum()
    abc_df['Cumulative %'] = abc_df['% Contribution'].cumsum()
    abc_df['ABC Category'] = pd.cut(abc_df['Cumulative %'], bins=[0,70,90,100], labels=['A','B','C'])
    st.dataframe(abc_df)

    # Sales Trend Line Chart
    st.subheader(f"📈 Sales Trend for {selected_product}")
    fig1 = px.line(sales_data, x='ds', y='y', title='Monthly Sales')
    st.plotly_chart(fig1)

    # Sidebar - Model selection
    model_type = st.sidebar.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "Holt-Winters", "Random Forest", "XGBoost"])
    forecast_months = st.sidebar.slider("Forecast Months", 3, 12, 6)

    # Forecasting Logic
    if model_type == "Prophet":
        model = Prophet()
        model.fit(sales_data)
        future = model.make_future_dataframe(periods=forecast_months, freq='M')
        forecast = model.predict(future)
        fig2 = model.plot(forecast)
        st.pyplot(fig2)
        forecast_result = forecast[['ds','yhat']].tail(forecast_months)

    elif model_type == "ARIMA":
        model = ARIMA(sales_data['y'], order=(1,1,1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_months)
        forecast_result = pd.DataFrame({'ds': pd.date_range(sales_data['ds'].max()+pd.DateOffset(months=1), periods=forecast_months, freq='M'), 'yhat': forecast})
        st.line_chart(forecast_result.set_index('ds'))

    elif model_type == "Holt-Winters":
        model = ExponentialSmoothing(sales_data['y'], trend='add', seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(forecast_months)
        forecast_result = pd.DataFrame({'ds': pd.date_range(sales_data['ds'].max()+pd.DateOffset(months=1), periods=forecast_months, freq='M'), 'yhat': forecast})
        st.line_chart(forecast_result.set_index('ds'))

    elif model_type in ["Random Forest", "XGBoost"]:
        sales_data['month'] = sales_data['ds'].dt.month
        sales_data['year'] = sales_data['ds'].dt.year
        X = sales_data[['month', 'year']]
        y = sales_data['y']

        if model_type == "Random Forest":
            model = RandomForestRegressor()
        else:
            model = XGBRegressor(objective='reg:squarederror')

        model.fit(X, y)

        future_dates = pd.date_range(sales_data['ds'].max()+pd.DateOffset(months=1), periods=forecast_months, freq='M')
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['month'] = future_df['ds'].dt.month
        future_df['year'] = future_df['ds'].dt.year

        future_df['yhat'] = model.predict(future_df[['month','year']])
        st.line_chart(future_df.set_index('ds')['yhat'])
        forecast_result = future_df[['ds','yhat']]

    # Display Forecast
    st.subheader("🔮 Forecast Results")
    st.dataframe(forecast_result)

    # Accuracy Evaluation
    st.subheader("📏 Forecast Accuracy (Test Data: Last 3 Months)")
    test_data = sales_data.tail(3)
    if model_type == "Prophet":
        test_forecast = forecast[forecast['ds'].isin(test_data['ds'])]
        y_true = test_data['y'].values
        y_pred = test_forecast['yhat'].values
    else:
        y_true = test_data['y'].values
        y_pred = forecast_result.head(3)['yhat'].values

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    st.write(f"**MAPE**: {mape:.2f}%")
    st.write(f"**RMSE**: {rmse:.2f}")

    # Other Visuals
    st.subheader("📊 Additional Visuals")
    st.markdown("**Sales by Category**")
    fig3 = px.bar(df, x='Product Category', y='Total Amount', color='Product Category', title="Sales by Category")
    st.plotly_chart(fig3)

    st.markdown("**Monthly Sales Heatmap**")
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    pivot = df.pivot_table(index='Year', columns='Month', values='Total Amount', aggfunc='sum')
    fig4, ax = plt.subplots()
    sns.heatmap(pivot, cmap='Blues', annot=True, fmt='.0f', ax=ax)
    st.pyplot(fig4)

    st.markdown("**Customer Age Distribution**")
    fig5 = px.histogram(df, x='Age', nbins=20, title="Age Distribution")
    st.plotly_chart(fig5)

    st.markdown("**Gender Breakdown**")
    fig6 = px.pie(df, names='Gender', title='Customer Gender Ratio')
    st.plotly_chart(fig6)

else:
    st.info("Please upload a CSV file to get started.")
