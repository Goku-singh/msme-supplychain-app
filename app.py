import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="MSME Forecasting App", layout="wide")
st.title("📦 MSME Sales Forecasting Tool")

# Step 1: Upload
uploaded_file = st.file_uploader("📁 Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Step 2: Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Step 3: Data cleaning
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["Date"])  # Drop rows with invalid dates
    df["Total Amount"] = df["Quantity"] * df["Price per Unit"]
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Step 4: Monthly aggregation
    df["Month"] = df["Date"].dt.to_period("M")
    monthly_sales = df.groupby(["Month", "Product Category"])["Total Amount"].sum().reset_index()
    monthly_sales["Month"] = monthly_sales["Month"].dt.to_timestamp()

    # Step 5: Category selection
    categories = monthly_sales["Product Category"].unique()
    selected_cat = st.selectbox("🛍️ Select a Product Category to Forecast", categories)

    cat_df = monthly_sales[monthly_sales["Product Category"] == selected_cat].set_index("Month")

    st.subheader(f"📈 Historical Sales Trend for '{selected_cat}'")
    st.line_chart(cat_df["Total Amount"])

    # Step 6: Holt-Winters Forecasting (12-month seasonal)
    st.subheader("🔮 Sales Forecast (Next 6 Months)")
    if len(cat_df) >= 12:
        try:
            model = ExponentialSmoothing(cat_df["Total Amount"], trend="add", seasonal="add", seasonal_periods=6)
            fit = model.fit()
            forecast = fit.forecast(6)

            forecast_df = forecast.reset_index()
            forecast_df.columns = ["Month", "Forecasted Sales"]
            st.line_chart(forecast)

            st.write("📌 Forecasted Values:")
            st.dataframe(forecast_df)
        except Exception as e:
            st.error(f"Model error: {e}")
    else:
        st.warning("Not enough monthly data to forecast (minimum 12 months required).")
