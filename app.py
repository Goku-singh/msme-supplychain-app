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
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Ensure there are enough data points
if len(cat_df) > 12:
    # Split into train/test (last 3 months as test)
    train = cat_df.iloc[:-3]
    test = cat_df.iloc[-3:]

    # Forecast using trend only
    model = ExponentialSmoothing(train["Total Amount"], trend="add", seasonal=None)
    fit = model.fit()
    prediction = fit.forecast(3)

    # Accuracy metrics
    mape = mean_absolute_percentage_error(test["Total Amount"], prediction) * 100
    rmse = np.sqrt(mean_squared_error(test["Total Amount"], prediction))

    st.subheader("📈 Step 2: Sales Forecast (Next 6 Months)")
    forecast_6mo = fit.forecast(6)
    st.line_chart(forecast_6mo)

    st.markdown("### 📊 Forecast Accuracy on Test Data (Last 3 Months)")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**RMSE:** {rmse:.2f}")
else:
    st.warning("Not enough data points to evaluate forecast accuracy. Please upload at least 15+ months of data.")
