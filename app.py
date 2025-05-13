import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import datetime

# Page config
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

# Title
st.title("📊 Retail Sales & Forecasting Dashboard")

# Upload data
uploaded_file = st.sidebar.file_uploader("Upload your retail sales CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])

    # Sidebar filters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['Date'].max().date())

    # Filter by product category
    categories = st.sidebar.multiselect("Product Categories", df['Product Category'].unique(), default=df['Product Category'].unique())

    # Apply filters
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) &
                     (df['Date'] <= pd.to_datetime(end_date)) &
                     (df['Product Category'].isin(categories))]

    # Preprocessing
    df_filtered['Total Amount'] = df_filtered['Quantity'] * df_filtered['Price per Unit']

    # KPI Cards
    total_revenue = df_filtered['Total Amount'].sum()
    total_orders = df_filtered.shape[0]
    total_customers = df_filtered['Customer ID'].nunique()
    aov = total_revenue / total_orders if total_orders != 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"₹{total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders}")
    col3.metric("Unique Customers", f"{total_customers}")
    col4.metric("Average Order Value", f"₹{aov:,.2f}")

    st.markdown("---")

    # ABC Analysis: Revenue-Based
    st.subheader("🔍 ABC Analysis by Revenue")
    revenue_by_product = df_filtered.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False).reset_index()
    revenue_by_product['Cumulative %'] = 100 * revenue_by_product['Total Amount'].cumsum() / revenue_by_product['Total Amount'].sum()
    revenue_by_product['Class'] = pd.cut(revenue_by_product['Cumulative %'], bins=[0, 70, 90, 100], labels=['A', 'B', 'C'])
    st.dataframe(revenue_by_product)

    fig_revenue = px.bar(revenue_by_product, x='Product Category', y='Total Amount', color='Class', title="Revenue Contribution by Product Category")
    st.plotly_chart(fig_revenue, use_container_width=True)

    # ABC Analysis: Quantity Sold
    st.subheader("🔍 ABC Analysis by Quantity Sold")
    quantity_by_product = df_filtered.groupby('Product Category')['Quantity'].sum().sort_values(ascending=False).reset_index()
    quantity_by_product['Cumulative %'] = 100 * quantity_by_product['Quantity'].cumsum() / quantity_by_product['Quantity'].sum()
    quantity_by_product['Class'] = pd.cut(quantity_by_product['Cumulative %'], bins=[0, 70, 90, 100], labels=['A', 'B', 'C'])
    st.dataframe(quantity_by_product)

    fig_qty = px.bar(quantity_by_product, x='Product Category', y='Quantity', color='Class', title="Quantity Sold by Product Category")
    st.plotly_chart(fig_qty, use_container_width=True)

    # ABC Analysis: Frequency of Sale
    st.subheader("🔍 ABC Analysis by Frequency of Sale")
    freq_by_product = df_filtered.groupby('Product Category').size().reset_index(name='Frequency')
    freq_by_product = freq_by_product.sort_values(by='Frequency', ascending=False)
    freq_by_product['Cumulative %'] = 100 * freq_by_product['Frequency'].cumsum() / freq_by_product['Frequency'].sum()
    freq_by_product['Class'] = pd.cut(freq_by_product['Cumulative %'], bins=[0, 70, 90, 100], labels=['A', 'B', 'C'])
    st.dataframe(freq_by_product)

    fig_freq = px.bar(freq_by_product, x='Product Category', y='Frequency', color='Class', title="Frequency of Sales by Product Category")
    st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("---")

    # Time Series Revenue Chart
    st.subheader("📈 Sales Trend Over Time")
    df_ts = df_filtered.groupby('Date')['Total Amount'].sum().reset_index()
    fig_time = px.line(df_ts, x='Date', y='Total Amount', title="Revenue Over Time")
    st.plotly_chart(fig_time, use_container_width=True)

    # Optional: Monthly Heatmap
    st.subheader("📅 Monthly Sales Heatmap")
    df_filtered['Month'] = df_filtered['Date'].dt.strftime('%Y-%m')
    heat_data = df_filtered.groupby(['Month', 'Product Category'])['Total Amount'].sum().reset_index()
    heat_pivot = heat_data.pivot(index='Product Category', columns='Month', values='Total Amount').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heat_pivot, cmap='YlGnBu', annot=True, fmt='.0f')
    st.pyplot(fig)

else:
    st.warning("📂 Please upload a CSV file to begin analysis.")
