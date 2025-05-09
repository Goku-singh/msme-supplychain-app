import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="MSME Supply Chain App", layout="wide")
st.title("📊 MSME Supply Chain Helper")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Step 1: Map your data columns")
    st.write("Preview:", df.head())

    date_col = st.selectbox("Select the column for Date", df.columns)
    sales_col = st.selectbox("Select the column for Sales/Orders", df.columns)
    inv_col = st.selectbox("Select the column for Inventory", df.columns)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    st.subheader("Step 2: Sales Forecast (Next 6 months)")
    model = ExponentialSmoothing(df[sales_col], trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(6)

    st.line_chart(forecast)

    st.subheader("Step 3: Inventory Trend")
    st.line_chart(df[inv_col])
