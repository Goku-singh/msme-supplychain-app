import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set page title
st.set_page_config(page_title="MSME Supply Chain App", layout="wide")
st.title("📊 MSME Supply Chain Helper")

# Upload file
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

# Custom MAPE and RMSE functions (no sklearn required)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Step 1: Map Your Data Columns")
    st.write("Preview of your data:")
    st.dataframe(df.head())

    date_col = st.selectbox("Select the Date column", df.columns)
    category_col = st.selectbox("Select the Product Category column", df.columns)
    sales_col = st.selectbox("Select the Sales Amount column (e.g., Total Amount)", df.columns)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # User selects product category to forecast
    unique_categories = df[category_col].unique()
    selected_category = st.selectbox("Choose a Product Category for Forecasting", unique_categories)

    # Filter data for selected category
    cat_df = df[df[category_col] == selected_category]
    cat_df = cat_df.groupby(date_col)[sales_col].sum().reset_index()
    cat_df.set_index(date_col, inplace=True)

    st.subheader("Step 2: Visualize Historical Sales")
    st.line_chart(cat_df)

    # Forecasting block
    if len(cat_df) > 15:
        try:
            train = cat_df.iloc[:-3]
            test = cat_df.iloc[-3:]

            model = ExponentialSmoothing(train[sales_col], trend="add", seasonal=None)
            fit = model.fit()
            prediction = fit.forecast(3)

            mape = mean_absolute_percentage_error(test[sales_col], prediction)
            rmse = root_mean_squared_error(test[sales_col], prediction)

            st.subheader("📈 Step 3: Sales Forecast (Next 6 Months)")
            forecast_6mo = fit.forecast(6)
            st.line_chart(forecast_6mo)

            st.markdown("### 📊 Forecast Accuracy (Test Data: Last 3 Months)")
            st.write(f"**MAPE:** {mape:.2f}%")
            st.write(f"**RMSE:** {rmse:.2f}")

        except Exception as e:
            st.error(f"Model error: {e}")
    else:
        st.warning("Not enough data points to perform reliable forecasting. Please upload at least 15+ months of data.")
