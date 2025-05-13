import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from datetime import datetime

st.set_page_config(page_title="MSME Supply Chain App", layout="wide")
st.title("📊 MSME Supply Chain Dashboard")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')
    df["Month"] = df["Date"].dt.to_period("M")

    # Sidebar filters
    st.sidebar.header("Filter Data")
    date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])
    product_filter = st.sidebar.multiselect("Select Product Category", options=df["Product Category"].unique(), default=df["Product Category"].unique())

    df = df[(df["Date"] >= pd.to_datetime(date_range[0])) & (df["Date"] <= pd.to_datetime(date_range[1])) & (df["Product Category"].isin(product_filter))]

    # KPI Cards
    total_revenue = df["Total Amount"].sum()
    total_orders = df["Transaction ID"].nunique()
    total_customers = df["Customer ID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("💰 Total Revenue", f"₹{total_revenue:,.2f}")
    kpi2.metric("🧾 Total Orders", total_orders)
    kpi3.metric("👥 Unique Customers", total_customers)
    kpi4.metric("📦 Avg Order Value", f"₹{avg_order_value:,.2f}")

    st.markdown("---")

    # Sales Trend Chart
    monthly_sales = df.groupby("Month")["Total Amount"].sum().reset_index()
    st.subheader("📈 Monthly Sales Trend")
    fig = px.line(monthly_sales, x="Month", y="Total Amount", title="Monthly Revenue", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("🌡️ Monthly Revenue Heatmap")
    heatmap_df = df.copy()
    heatmap_df["Year"] = heatmap_df["Date"].dt.year
    heatmap_df["Month_Num"] = heatmap_df["Date"].dt.month
    pivot = heatmap_df.pivot_table(index="Year", columns="Month_Num", values="Total Amount", aggfunc="sum")
    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
    st.pyplot(fig)

    # ABC Analysis
    st.subheader("🔹 ABC Analysis")
    abc_df = df.groupby("Product Category").agg({"Total Amount": "sum", "Quantity": "sum", "Transaction ID": "count"}).reset_index()
    abc_df.columns = ["Product Category", "Revenue", "Quantity", "Frequency"]
    abc_df = abc_df.sort_values("Revenue", ascending=False)
    abc_df["Revenue%"] = 100 * abc_df["Revenue"] / abc_df["Revenue"].sum()
    abc_df["CumRevenue%"] = abc_df["Revenue%"].cumsum()
    abc_df["Class"] = pd.cut(abc_df["CumRevenue%"], bins=[0, 70, 90, 100], labels=["A", "B", "C"])
    st.dataframe(abc_df)

    # Forecasting
    st.subheader("🔮 Sales Forecasting")
    forecast_model = st.selectbox("Select Forecasting Model", ["Holt-Winters", "Prophet", "Random Forest", "XGBoost"])

    sales_by_date = df.groupby("Date")["Total Amount"].sum().reset_index()
    sales_by_date = sales_by_date.sort_values("Date")

    if forecast_model == "Holt-Winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(sales_by_date["Total Amount"], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(12)
        forecast_df = pd.DataFrame({"Date": pd.date_range(start=sales_by_date["Date"].max(), periods=12, freq='M'), "Forecast": forecast})

    elif forecast_model == "Prophet":
        prophet_df = sales_by_date.rename(columns={"Date": "ds", "Total Amount": "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=90)
        forecast = model.predict(future)
        forecast_df = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecast"})
        forecast_df = forecast_df[forecast_df["Date"] > sales_by_date["Date"].max()]

    elif forecast_model in ["Random Forest", "XGBoost"]:
        df_ml = sales_by_date.copy()
        df_ml["Day"] = df_ml["Date"].dt.day
        df_ml["Month"] = df_ml["Date"].dt.month
        df_ml["Year"] = df_ml["Date"].dt.year

        X = df_ml[["Day", "Month", "Year"]]
        y = df_ml["Total Amount"]

        if forecast_model == "Random Forest":
            model = RandomForestRegressor()
        else:
            model = XGBRegressor()

        model.fit(X, y)

        future_dates = pd.date_range(start=sales_by_date["Date"].max(), periods=90)
        future_df = pd.DataFrame({"Date": future_dates})
        future_df["Day"] = future_df["Date"].dt.day
        future_df["Month"] = future_df["Date"].dt.month
        future_df["Year"] = future_df["Date"].dt.year

        forecast_df = future_df.copy()
        forecast_df["Forecast"] = model.predict(future_df[["Day", "Month", "Year"]])

    # Plot forecast
    st.line_chart(forecast_df.set_index("Date"))

    # Evaluate accuracy on last 3 months
    test_data = sales_by_date.tail(90)
    test_true = test_data["Total Amount"]
    if forecast_model == "Prophet":
        test_pred = forecast_df["Forecast"].iloc[:90].values
    else:
        test_pred = forecast_df["Forecast"].head(90).values

    rmse = mean_squared_error(test_true, test_pred, squared=False)
    mape = np.mean(np.abs((test_true - test_pred) / test_true)) * 100

    st.markdown("#### Forecast Accuracy (Last 3 Months)")
    st.write(f"**MAPE**: {mape:.2f}%")
    st.write(f"**RMSE**: {rmse:.2f}")
