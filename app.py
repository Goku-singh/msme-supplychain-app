import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_product_clusters(df, col_mapping, n_clusters=5):
    """
    Generate product clusters for recommendations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    col_mapping : dict
        Dictionary mapping column types to column names
    n_clusters : int
        Number of clusters to generate
    
    Returns:
    --------
    dict
        Dictionary containing cluster results
    """
    product_col = col_mapping['product']
    quantity_col = col_mapping['quantity']
    revenue_col = col_mapping['revenue']
    date_col = col_mapping['date']
    
    # Group by product and calculate metrics
    product_metrics = df.groupby(product_col).agg({
        quantity_col: ['sum', 'mean', 'std'],
        revenue_col: ['sum', 'mean'],
        date_col: ['count']
    }).reset_index()
    
    # Flatten column names
    product_metrics.columns = [
        '_'.join(col).strip('_') for col in product_metrics.columns.values
    ]
    
    # Calculate additional metrics
    product_metrics['revenue_per_unit'] = product_metrics[f'{revenue_col}_sum'] / product_metrics[f'{quantity_col}_sum']
    product_metrics['cv'] = product_metrics[f'{quantity_col}_std'] / product_metrics[f'{quantity_col}_mean']
    product_metrics['frequency'] = product_metrics[f'{date_col}_count']
    
    # Handle missing values
    product_metrics = product_metrics.fillna(0)
    
    # Select features for clustering
    features = [
        f'{quantity_col}_sum',
        f'{revenue_col}_sum',
        'revenue_per_unit',
        'cv',
        'frequency'
    ]
    
    X = product_metrics[features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    product_metrics['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add PCA components
    product_metrics['pca1'] = X_pca[:, 0]
    product_metrics['pca2'] = X_pca[:, 1]
    
    # Calculate cluster centers in original feature space
    # Transform cluster centers back to original scale
    centers_transformed = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers = pd.DataFrame(centers_transformed)
    # Assign column names
    for i, feature in enumerate(features):
        cluster_centers = cluster_centers.rename(columns={i: feature})
    cluster_centers['cluster'] = np.arange(n_clusters)
    
    # Calculate cluster summaries
    cluster_summary = product_metrics.groupby('cluster').agg({
        product_col: 'count',
        f'{quantity_col}_sum': 'sum',
        f'{revenue_col}_sum': 'sum',
        'revenue_per_unit': 'mean',
        'cv': 'mean',
        'frequency': 'mean'
    }).reset_index()
    
    cluster_summary['percent_products'] = cluster_summary[product_col] / cluster_summary[product_col].sum() * 100
    cluster_summary['percent_revenue'] = cluster_summary[f'{revenue_col}_sum'] / cluster_summary[f'{revenue_col}_sum'].sum() * 100
    
    # Generate cluster names
    cluster_names = {}
    for idx, row in cluster_summary.iterrows():
        cluster_id = row['cluster']
        
        # Determine key characteristics
        high_revenue = row[f'{revenue_col}_sum'] > cluster_summary[f'{revenue_col}_sum'].median()
        high_quantity = row[f'{quantity_col}_sum'] > cluster_summary[f'{quantity_col}_sum'].median()
        high_price = row['revenue_per_unit'] > cluster_summary['revenue_per_unit'].median()
        high_variability = row['cv'] > cluster_summary['cv'].median()
        high_frequency = row['frequency'] > cluster_summary['frequency'].median()
        
        # Generate cluster name
        if high_revenue and high_quantity:
            prefix = "High Volume"
        elif high_revenue and not high_quantity:
            prefix = "High Value"
        elif not high_revenue and high_quantity:
            prefix = "High Quantity"
        else:
            prefix = "Low Performance"
        
        suffix = []
        if high_price:
            suffix.append("Premium")
        if high_variability:
            suffix.append("Variable")
        if high_frequency:
            suffix.append("Frequent")
        
        if suffix:
            name = f"{prefix}, {' & '.join(suffix)}"
        else:
            name = prefix
        
        cluster_names[cluster_id] = name
    
    # Add names to summary
    cluster_summary['name'] = cluster_summary['cluster'].map(cluster_names)
    
    # Generate recommendations for each cluster
    cluster_recommendations = {}
    
    for cluster_id, name in cluster_names.items():
        if "High Volume" in name:
            recommendations = [
                "Implement lean inventory management",
                "Optimize supply chain for efficiency",
                "Negotiate volume discounts",
                "Monitor closely for demand changes"
            ]
        elif "High Value" in name:
            recommendations = [
                "Focus on premium segment marketing",
                "Ensure consistent availability",
                "Implement special handling procedures",
                "Consider bundling with complementary products"
            ]
        elif "High Quantity" in name:
            recommendations = [
                "Optimize warehouse space utilization",
                "Consider bulk ordering to reduce costs",
                "Implement automated reordering",
                "Review pricing strategy"
            ]
        else:  # Low Performance
            recommendations = [
                "Evaluate product viability",
                "Consider phase-out strategy",
                "Test promotional pricing",
                "Reduce inventory investment"
            ]
        
        if "Variable" in name:
            recommendations.extend([
                "Implement safety stock buffers",
                "Develop flexible supply arrangements",
                "Use forecasting algorithms that handle volatility"
            ])
        
        if "Premium" in name:
            recommendations.extend([
                "Ensure high service levels",
                "Monitor quality metrics closely",
                "Develop premium customer segment"
            ])
        
        if "Frequent" in name:
            recommendations.extend([
                "Optimize replenishment cycles",
                "Consider dedicated storage locations",
                "Monitor stockout metrics closely"
            ])
        
        cluster_recommendations[cluster_id] = recommendations
    
    return {
        'product_data': product_metrics,
        'cluster_centers': cluster_centers,
        'cluster_summary': cluster_summary,
        'cluster_names': cluster_names,
        'features': features,
        'recommendations': cluster_recommendations
    }

def plot_product_clusters(cluster_results):
    """
    Plot product clusters for visualization.
    
    Parameters:
    -----------
    cluster_results : dict
        Dictionary containing cluster results
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with cluster plot
    """
    product_data = cluster_results['product_data']
    cluster_names = cluster_results['cluster_names']
    
    # Create scatterplot using PCA components
    fig = px.scatter(
        product_data,
        x='pca1',
        y='pca2',
        color='cluster',
        hover_name=product_data.columns[0],  # product column
        hover_data=[c for c in product_data.columns if c not in ['pca1', 'pca2', 'cluster']],
        title='Product Clusters (PCA Visualization)',
        color_continuous_scale=px.colors.qualitative.G10,
        height=600
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        legend_title='Cluster',
        coloraxis_colorbar=dict(
            title='Cluster',
            tickvals=list(cluster_names.keys()),
            ticktext=list(cluster_names.values())
        )
    )
    
    return fig

def generate_product_recommendations(df, col_mapping, cluster_results=None, n_recommendations=5):
    """
    Generate product recommendations based on clustering and association rules.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    col_mapping : dict
        Dictionary mapping column types to column names
    cluster_results : dict
        Dictionary containing cluster results
    n_recommendations : int
        Number of recommendations to generate per product
    
    Returns:
    --------
    dict
        Dictionary containing product recommendations
    """
    product_col = col_mapping['product']
    date_col = col_mapping['date']
    customer_id_col = col_mapping['customer_id']
    
    if customer_id_col is None:
        return {"error": "Customer ID column is required for product recommendations"}
    
    # If cluster results not provided, generate them
    if cluster_results is None:
        cluster_results = generate_product_clusters(df, col_mapping)
    
    # Create a simple association matrix based on co-occurrence
    # Group transactions by customer and date
    transactions = df.groupby([customer_id_col, date_col])[product_col].apply(list).reset_index()
    
    # Get unique products
    unique_products = df[product_col].unique()
    
    # Initialize co-occurrence matrix
    co_occurrence = pd.DataFrame(0, index=unique_products, columns=unique_products)
    
    # Fill co-occurrence matrix
    for _, row in transactions.iterrows():
        products = row[product_col]
        for p1 in products:
            for p2 in products:
                if p1 != p2:
                    co_occurrence.loc[p1, p2] += 1
    
    # Generate recommendations
    recommendations = {}
    
    for product in unique_products:
        # Get similar products based on co-occurrence
        product_co_occurrences = co_occurrence[product]
        similar_products = product_co_occurrences.sort_values(ascending=False).head(n_recommendations)
        
        # Get product cluster
        product_cluster = None
        if cluster_results and 'product_data' in cluster_results:
            product_data = cluster_results['product_data']
            product_row = product_data[product_data.iloc[:, 0] == product]
            if not product_row.empty:
                product_cluster = product_row['cluster'].values[0]
        
        # Get recommended products info
        recs = []
        for rec_product, co_occurrences in similar_products.items():
            if co_occurrences > 0:
                rec_info = {
                    'product': rec_product,
                    'co_occurrences': int(co_occurrences),
                    'strength': 'High' if co_occurrences > similar_products.median() else 'Medium'
                }
                
                # Add cluster info if available
                if product_cluster is not None:
                    rec_row = product_data[product_data.iloc[:, 0] == rec_product]
                    if not rec_row.empty:
                        rec_cluster = rec_row['cluster'].values[0]
                        rec_info['same_cluster'] = rec_cluster == product_cluster
                        rec_info['cluster'] = int(rec_cluster)
                        if 'cluster_names' in cluster_results:
                            rec_info['cluster_name'] = cluster_results['cluster_names'].get(rec_cluster, f"Cluster {rec_cluster}")
                
                recs.append(rec_info)
        
        recommendations[product] = recs
    
    return recommendations

def plot_bundle_recommendations(product_recommendations, top_n=5):
    """
    Plot product bundle recommendations.
    
    Parameters:
    -----------
    product_recommendations : dict
        Dictionary containing product recommendations
    top_n : int
        Number of top products to display
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure with bundle recommendations
    """
    # Flatten recommendations data
    rec_data = []
    for product, recs in product_recommendations.items():
        for rec in recs:
            rec_data.append({
                'Product': product,
                'Recommendation': rec['product'],
                'Co-occurrences': rec['co_occurrences'],
                'Strength': rec['strength'],
                'Same Cluster': rec.get('same_cluster', None),
                'Cluster': rec.get('cluster_name', None)
            })
    
    rec_df = pd.DataFrame(rec_data)
    
    # Get top product pairs by co-occurrence
    top_pairs = rec_df.sort_values('Co-occurrences', ascending=False).head(top_n)
    
    # Create network graph
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    products = set(top_pairs['Product'].tolist() + top_pairs['Recommendation'].tolist())
    
    # Create positions for nodes (circular layout)
    pos = {}
    n = len(products)
    radius = 1
    for i, product in enumerate(products):
        angle = 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[product] = (x, y)
        node_x.append(x)
        node_y.append(y)
        node_text.append(product)
        
        # Calculate node size based on frequency
        size = 20  # default size
        if product in rec_df['Product'].values:
            size += rec_df[rec_df['Product'] == product]['Co-occurrences'].sum() / 5
        if product in rec_df['Recommendation'].values:
            size += rec_df[rec_df['Recommendation'] == product]['Co-occurrences'].sum() / 5
        
        node_size.append(size)
    
    # Create edges
    for _, row in top_pairs.iterrows():
        p1, p2 = row['Product'], row['Recommendation']
        x0, y0 = pos[p1]
        x1, y1 = pos[p2]
        
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Connections'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Connection Strength',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        hovertemplate='%{text}<extra></extra>',
        name='Products'
    ))
    
    # Update layout
    fig.update_layout(
        title='Top Product Bundle Recommendations',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def generate_executive_summary(df, col_mapping, forecast_results, abc_results=None):
    """
    Generate executive summary dashboard data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    col_mapping : dict
        Dictionary mapping column types to column names
    forecast_results : dict
        Dictionary of forecast results
    abc_results : dict
        Dictionary containing ABC analysis results
    
    Returns:
    --------
    dict
        Dictionary containing executive summary data
    """
    date_col = col_mapping['date']
    quantity_col = col_mapping['quantity']
    revenue_col = col_mapping['revenue']
    product_col = col_mapping['product']
    customer_id_col = col_mapping['customer_id']
    
    # Calculate historical metrics
    historical = {
        'total_revenue': df[revenue_col].sum(),
        'total_quantity': df[quantity_col].sum(),
        'unique_products': df[product_col].nunique(),
        'unique_customers': df[customer_id_col].nunique() if customer_id_col else None,
        'first_date': df[date_col].min(),
        'last_date': df[date_col].max(),
    }
    
    # Calculate average metrics
    date_range = (historical['last_date'] - historical['first_date']).days + 1
    months = date_range / 30.44  # average month length
    
    historical['avg_monthly_revenue'] = historical['total_revenue'] / months
    historical['avg_monthly_quantity'] = historical['total_quantity'] / months
    historical['avg_order_value'] = historical['total_revenue'] / df.shape[0]
    
    # Calculate forecast metrics for 3M, 6M, 12M
    forecast_horizons = {
        '3M': 90,
        '6M': 180,
        '12M': 365
    }
    
    forecast_metrics = {}
    
    if forecast_results:
        # Get average daily revenue
        avg_daily_revenue = historical['total_revenue'] / date_range
        
        for horizon_name, days in forecast_horizons.items():
            total_forecast_quantity = 0
            
            for product, product_forecasts in forecast_results.items():
                # Use first model's forecast
                first_model = list(product_forecasts.keys())[0]
                forecast_df = product_forecasts[first_model]
                
                # Calculate forecast for this horizon
                horizon_forecast = forecast_df.head(days)['yhat'].sum()
                total_forecast_quantity += horizon_forecast
            
            # Estimate revenue based on historical revenue per unit
            revenue_per_unit = historical['total_revenue'] / historical['total_quantity']
            forecast_revenue = total_forecast_quantity * revenue_per_unit
            
            forecast_metrics[horizon_name] = {
                'forecast_quantity': total_forecast_quantity,
                'forecast_revenue': forecast_revenue
            }
    
    # Get ABC classification data
    abc_metrics = {}
    
    if abc_results and 'data' in abc_results:
        abc_data = abc_results['data']
        
        a_products = abc_data[abc_data['Class'] == 'A']['Product'].tolist()
        b_products = abc_data[abc_data['Class'] == 'B']['Product'].tolist()
        c_products = abc_data[abc_data['Class'] == 'C']['Product'].tolist()
        
        a_revenue = abc_data[abc_data['Class'] == 'A']['Revenue'].sum()
        b_revenue = abc_data[abc_data['Class'] == 'B']['Revenue'].sum()
        c_revenue = abc_data[abc_data['Class'] == 'C']['Revenue'].sum()
        
        abc_metrics = {
            'a_count': len(a_products),
            'b_count': len(b_products),
            'c_count': len(c_products),
            'a_revenue': a_revenue,
            'b_revenue': b_revenue,
            'c_revenue': c_revenue,
            'a_revenue_pct': a_revenue / (a_revenue + b_revenue + c_revenue) * 100,
            'b_revenue_pct': b_revenue / (a_revenue + b_revenue + c_revenue) * 100,
            'c_revenue_pct': c_revenue / (a_revenue + b_revenue + c_revenue) * 100,
        }
    
    # Calculate top products
    top_products = df.groupby(product_col).agg({
        quantity_col: 'sum',
        revenue_col: 'sum'
    }).reset_index()
    
    top_products = top_products.sort_values(revenue_col, ascending=False).head(5)
    
    return {
        'historical': historical,
        'forecast': forecast_metrics,
        'abc': abc_metrics,
        'top_products': top_products
    }

def plot_executive_summary(executive_summary):
    """
    Plot executive summary dashboard.
    
    Parameters:
    -----------
    executive_summary : dict
        Dictionary containing executive summary data
    
    Returns:
    --------
    tuple
        Tuple containing KPI data and forecast comparison figure
    """
    historical = executive_summary['historical']
    forecast = executive_summary['forecast']
    abc = executive_summary['abc']
    
    # Create KPI cards data
    kpi_data = {
        "Total Revenue": {
            "value": f"${historical['total_revenue']:,.2f}",
            "delta": "Historical Total",
            "color": "blue"
        },
        "Total Orders": {
            "value": f"{historical['total_quantity']:,}",
            "delta": "Historical Units",
            "color": "blue"
        }
    }
    
    if historical['unique_customers']:
        kpi_data["Total Customers"] = {
            "value": f"{historical['unique_customers']:,}",
            "delta": "Unique Customers",
            "color": "blue"
        }
    
    # Add forecast KPIs
    if forecast and '6M' in forecast:
        kpi_data["6-Month Forecast"] = {
            "value": f"${forecast['6M']['forecast_revenue']:,.2f}",
            "delta": f"{forecast['6M']['forecast_quantity']:,} Units",
            "color": "green"
        }
    
    # Create forecast comparison plot
    fig = go.Figure()
    
    if forecast:
        # Prepare data
        horizons = []
        values = []
        
        for horizon, data in forecast.items():
            horizons.append(horizon)
            values.append(data['forecast_revenue'])
        
        # Add bars
        fig.add_trace(go.Bar(
            x=horizons,
            y=values,
            text=[f"${x:,.2f}" for x in values],
            textposition='auto',
            marker_color='rgb(26, 118, 255)'
        ))
        
        # Update layout
        fig.update_layout(
            title='Forecasted Revenue by Horizon',
            xaxis_title='Forecast Horizon',
            yaxis_title='Forecasted Revenue',
            yaxis_tickprefix='$',
            height=400
        )
    else:
        # Empty plot with message
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No forecast data available",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title='Forecasted Revenue by Horizon',
            height=400
        )
    
    return kpi_data, fig
