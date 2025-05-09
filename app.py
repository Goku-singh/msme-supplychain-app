import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="MSME Forecasting App", layout="wide")
st.title("📦 MSME Sales Forecasting Tool")

# Step 1: Upload
uploaded_file = st.file_uploader("📁 Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Step 2: Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Step 3: Basic cleaning
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Total Amount"] = df["Quantity"] * df["Price per Unit"]
    
    st.subheader("📋 Data Preview")
    st.write(df.head())

    # Step 4: Group by Category + Month
    df["Month"] = df["Date"].dt.to_period("M")
    monthly_sales = df.groupby(["Month", "Product Category"])["Total Amount"].sum().reset_index()
    monthly_sales["Month"] = monthly_sales["Month"].dt.to_timestamp()

    # Step 5: User selects category
    categories = monthly_sales["Product Category"].unique()
    selected_cat = st.selectbox("🛍️ Select a Product Category to Forecast", categories)

    cat_df = monthly_sales[monthly_sales["Product Category"] == selected_cat].set_index("Month")

    st.subheader(f"📈 Historical Sales for '{selected_cat}'")
    st.line_chart(cat_df["Total Amount"])

    # Step 6: Holt-Winters Forecast
    st.subheader("🔮 Sales Forecast (Next 6 Months)")
    if len(cat_df) >= 12:
        model = ExponentialSmoothing(cat_df["Total Amount"], trend="add", seasonal="add", seasonal_periods=12)
        fit = model.fit()
        forecast = fit.forecast(6)
        st.line_chart(forecast)
    else:
        st.warning("Not enough data to forecast (min 12 months required).")
