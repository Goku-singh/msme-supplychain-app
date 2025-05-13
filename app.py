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
from sklearn.cluster import KMeans
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

    # Monthly Sales Trend
    monthly_sales = df.groupby("Month")["Total Amount"].sum().reset_index()
    st.subheader("📈 Monthly Sales Trend")
    fig = px.line(monthly_sales, x="Month", y="Total Amount", title="Monthly Revenue", markers=True, color_discrete_sequence=['royalblue'])
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

    # ABC Analysis by Revenue, Quantity, Frequency
    st.subheader("🔹 ABC Analysis")
    abc_df = df.groupby("Product Category").agg({"Total Amount": "sum", "Quantity": "sum", "Transaction ID": "count"}).reset_index()
    abc_df.columns = ["Product Category", "Revenue", "Quantity", "Frequency"]
    for col in ["Revenue", "Quantity", "Frequency"]:
        abc_df[f"{col}%"] = 100 * abc_df[col] / abc_df[col].sum()
        abc_df[f"Cum{col}%"] = abc_df[f"{col}%"].cumsum()
        abc_df[f"{col}_Class"] = pd.cut(abc_df[f"Cum{col}%"], bins=[0, 70, 90, 100], labels=["A", "B", "C"])
    st.dataframe(abc_df)

    # Inventory Metrics
    st.subheader("📦 Inventory Metrics")
    inventory_df = df.groupby("Product Category").agg({"Quantity": "sum", "Total Amount": "sum"}).reset_index()
    inventory_df["Average Daily Sales"] = inventory_df["Quantity"] / ((df["Date"].max() - df["Date"].min()).days + 1)
    inventory_df["Days of Supply"] = np.where(inventory_df["Average Daily Sales"] > 0, inventory_df["Quantity"] / inventory_df["Average Daily Sales"], np.nan)
    inventory_df["Stockouts"] = df[df["Quantity"] == 0].groupby("Product Category")["Transaction ID"].count()
    inventory_df["Stockouts"] = inventory_df["Stockouts"].fillna(0).astype(int)
    st.dataframe(inventory_df)

    # Customer Clustering
    st.subheader("👥 Customer Segmentation (K-Means Clustering)")
    customer_data = df.groupby("Customer ID").agg({"Total Amount": "sum", "Quantity": "sum", "Transaction ID": "count"}).reset_index()
    customer_data.columns = ["Customer ID", "Revenue", "Quantity", "Frequency"]
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data["Cluster"] = kmeans.fit_predict(customer_data[["Revenue", "Quantity", "Frequency"]])
    fig = px.scatter_3d(customer_data, x="Revenue", y="Quantity", z="Frequency", color="Cluster", symbol="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # Forecasting
    st.subheader("🔮 Sales Forecasting")
    forecast_model = st.selectbox("Select Forecasting Model", ["Holt-Winters", "Prophet", "Random Forest", "XGBoost"])

    sales_by_date = df.groupby("Date")["Total Amount"].sum().reset_index().sort_values("Date")
    forecast_df = pd.DataFrame()

    if forecast_model == "Holt-Winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(sales_by_date["Total Amount"], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(90)
        forecast_df = pd.DataFrame({"Date": pd.date_range(start=sales_by_date["Date"].max() + pd.Timedelta(days=1), periods=90), "Forecast": forecast})

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

        model = RandomForestRegressor() if forecast_model == "Random Forest" else XGBRegressor()
        model.fit(X, y)

        future_dates = pd.date_range(start=sales_by_date["Date"].max() + pd.Timedelta(days=1), periods=90)
        future_df = pd.DataFrame({"Date": future_dates})
        future_df["Day"] = future_df["Date"].dt.day
        future_df["Month"] = future_df["Date"].dt.month
        future_df["Year"] = future_df["Date"].dt.year
        future_df["Forecast"] = model.predict(future_df[["Day", "Month", "Year"]])
        forecast_df = future_df

    # Plot forecast vs actual
    st.subheader("📊 Actual vs Forecasted Sales")
    combined = pd.concat([sales_by_date.rename(columns={"Total Amount": "Actual"}), forecast_df], ignore_index=True)
    fig2 = px.line(combined, x="Date", y=["Actual", "Forecast"], markers=True, title="Forecast vs Actual", color_discrete_map={"Actual": "green", "Forecast": "orange"})
    st.plotly_chart(fig2, use_container_width=True)

    # Accuracy
    test_actual = sales_by_date["Total Amount"].tail(len(forecast_df)).values
    test_forecast = forecast_df["Forecast"].values[:len(test_actual)]

    if len(test_actual) == len(test_forecast):
        rmse = mean_squared_error(test_actual, test_forecast, squared=False)
        mape = np.mean(np.abs((test_actual - test_forecast) / test_actual)) * 100
        st.markdown("#### Forecast Accuracy (Last Available Days)")
        st.write(f"**MAPE**: {mape:.2f}%")
        st.write(f"**RMSE**: {rmse:.2f}")
    else:
        st.warning("Insufficient data overlap for forecast accuracy calculation.")
