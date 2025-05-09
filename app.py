import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set up page title and layout
st.set_page_config(page_title="MSME Supply Chain App", layout="wide")
st.title("📊 MSME Supply Chain Helper")

# Allow user to upload multiple Excel/CSV files
uploaded_files = st.file_uploader("Upload your Excel or CSV files", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    # Initialize an empty DataFrame to store merged data
    df_all = pd.DataFrame()

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Optionally, merge or concatenate the data from multiple files here if needed
        df_all = pd.concat([df_all, df], ignore_index=True)
    
    # Display the first few rows of the combined data
    st.subheader("Step 1: Map your data columns")
    st.write("Preview:", df_all.head())

    # Allow users to select columns from the dataset
    date_col = st.selectbox("Select the column for Date", df_all.columns)
    sales_col = st.selectbox("Select the column for Sales/Orders", df_all.columns)
    inv_col = st.selectbox("Select the column for Inventory", df_all.columns)

    # Convert the date column to datetime
    df_all[date_col] = pd.to_datetime(df_all[date_col])
    df_all = df_all.sort_values(date_col)
    df_all.set_index(date_col, inplace=True)

    # Check for missing data and handle it
    if df_all[sales_col].isnull().any():
        st.warning(f"Missing values found in the '{sales_col}' column. Consider filling or dropping them.")

    if df_all[inv_col].isnull().any():
        st.warning(f"Missing values found in the '{inv_col}' column. Consider filling or dropping them.")
    
    # Step 2: Sales Forecast (Next N months)
    st.subheader("Step 2: Sales Forecast (Predict Next Months)")
    forecast_periods = st.number_input("How many periods do you want to forecast?", min_value=1, max_value=12, value=6)

    model = ExponentialSmoothing(df_all[sales_col], trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(forecast_periods)

    # Display forecast plot
    st.line_chart(forecast)

    # Step 3: Inventory Trend
    st.subheader("Step 3: Inventory Trend")
    st.line_chart(df_all[inv_col])

    # Optionally, display the raw data used for the forecast and trend
    if st.checkbox("Show Raw Data"):
        st.write(df_all)
else:
    st.write("Please upload one or more Excel or CSV files to proceed.")

