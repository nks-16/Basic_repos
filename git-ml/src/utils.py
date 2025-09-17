"""
Utility functions for GitHub Security Anomaly Detection
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

def format_large_numbers(num: float) -> str:
    """Format large numbers for display"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"

def calculate_anomaly_statistics(results_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive statistics for anomaly detection results"""
    
    stats = {}
    
    # Basic counts
    stats['total_samples'] = len(results_df)
    stats['total_anomalies'] = len(results_df[results_df['is_anomaly']])
    stats['anomaly_rate'] = stats['total_anomalies'] / stats['total_samples']
    
    # Anomaly types
    if 'anomaly_type' in results_df.columns:
        anomaly_types = results_df[results_df['is_anomaly']]['anomaly_type'].value_counts()
        stats['anomaly_types'] = anomaly_types.to_dict()
    
    # Score statistics
    if 'anomaly_score' in results_df.columns:
        stats['score_stats'] = {
            'min': results_df['anomaly_score'].min(),
            'max': results_df['anomaly_score'].max(),
            'mean': results_df['anomaly_score'].mean(),
            'std': results_df['anomaly_score'].std(),
            'median': results_df['anomaly_score'].median()
        }
    
    return stats

def create_anomaly_score_plot(results_df: pd.DataFrame) -> go.Figure:
    """Create anomaly score distribution plot"""
    
    fig = go.Figure()
    
    # Normal users
    normal_scores = results_df[~results_df['is_anomaly']]['anomaly_score']
    anomaly_scores = results_df[results_df['is_anomaly']]['anomaly_score']
    
    fig.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal Users',
        nbinsx=50,
        opacity=0.7,
        marker_color='blue'
    ))
    
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Anomalous Users',
        nbinsx=20,
        opacity=0.7,
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Distribution of Anomaly Scores',
        xaxis_title='Anomaly Score',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_white'
    )
    
    return fig

def create_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create feature importance bar plot"""
    
    top_features = importance_df.head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Features for Anomaly Detection',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=600
    )
    
    return fig

def create_anomaly_type_plot(results_df: pd.DataFrame) -> go.Figure:
    """Create pie chart of anomaly types"""
    
    if 'anomaly_type' not in results_df.columns:
        return go.Figure()
    
    anomaly_data = results_df[results_df['is_anomaly']]['anomaly_type'].value_counts()
    
    colors = {
        'suspicious': '#ff4444',
        'risky_behavior': '#ffaa00',
        'misuse': '#ff6600',
        'unknown': '#888888',
        'normal': '#44ff44'
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=anomaly_data.index,
            values=anomaly_data.values,
            marker_colors=[colors.get(label, '#888888') for label in anomaly_data.index]
        )
    ])
    
    fig.update_layout(
        title='Distribution of Anomaly Types',
        template='plotly_white'
    )
    
    return fig

def create_monthly_trend_plot(results_df: pd.DataFrame) -> go.Figure:
    """Create monthly trend of anomalies"""
    
    if 'year_month' not in results_df.columns:
        return go.Figure()
    
    monthly_data = results_df.groupby('year_month').agg({
        'is_anomaly': ['count', 'sum'],
        'anomaly_score': 'mean'
    }).round(3)
    
    monthly_data.columns = ['total_users', 'anomalies', 'avg_score']
    monthly_data['anomaly_rate'] = monthly_data['anomalies'] / monthly_data['total_users']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Number of Anomalies by Month', 'Average Anomaly Score by Month'),
        vertical_spacing=0.15
    )
    
    # Anomaly count
    fig.add_trace(
        go.Bar(
            x=monthly_data.index,
            y=monthly_data['anomalies'],
            name='Anomalies',
            marker_color='red'
        ),
        row=1, col=1
    )
    
    # Average score
    fig.add_trace(
        go.Scatter(
            x=monthly_data.index,
            y=monthly_data['avg_score'],
            mode='lines+markers',
            name='Avg Score',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Monthly Anomaly Trends',
        template='plotly_white',
        height=600
    )
    
    return fig

def create_user_activity_heatmap(audit_logs: pd.DataFrame) -> go.Figure:
    """Create heatmap of user activity patterns"""
    
    if 'timestamp' not in audit_logs.columns:
        return go.Figure()
    
    # Ensure timestamp is datetime
    audit_logs['timestamp'] = pd.to_datetime(audit_logs['timestamp'])
    audit_logs['hour'] = audit_logs['timestamp'].dt.hour
    audit_logs['day_of_week'] = audit_logs['timestamp'].dt.day_name()
    
    # Create pivot table
    heatmap_data = audit_logs.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        colorbar=dict(title="Activity Count")
    ))
    
    fig.update_layout(
        title='User Activity Heatmap (Hour vs Day of Week)',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white'
    )
    
    return fig

def format_anomaly_details(row: pd.Series, feature_names: List[str]) -> Dict:
    """Format anomaly details for display"""
    
    details = {
        'user_id': row.get('user_id', 'Unknown'),
        'username': row.get('username', 'Unknown'),
        'anomaly_score': round(row.get('anomaly_score', 0), 4),
        'anomaly_type': row.get('anomaly_type', 'unknown'),
        'year_month': row.get('year_month', 'Unknown'),
        'in_org': 'Yes' if row.get('in_org', 0) == 1 else 'No'
    }
    
    # Get top contributing features
    contributing_features = []
    for feature in feature_names:
        if feature in row.index and row[feature] > 0:
            contributing_features.append({
                'feature': feature,
                'value': row[feature]
            })
    
    # Sort by value and take top 5
    contributing_features.sort(key=lambda x: x['value'], reverse=True)
    details['top_features'] = contributing_features[:5]
    
    return details

def generate_anomaly_report(results_df: pd.DataFrame, feature_names: List[str]) -> str:
    """Generate a text report of anomaly detection results"""
    
    stats = calculate_anomaly_statistics(results_df)
    anomalies = results_df[results_df['is_anomaly']].sort_values('anomaly_score')
    
    report = f"""
# GitHub Security Anomaly Detection Report

## Summary Statistics
- Total Users Analyzed: {stats['total_samples']:,}
- Anomalies Detected: {stats['total_anomalies']:,}
- Anomaly Rate: {stats['anomaly_rate']:.2%}

## Anomaly Types Distribution
"""
    
    if 'anomaly_types' in stats:
        for anomaly_type, count in stats['anomaly_types'].items():
            report += f"- {anomaly_type.replace('_', ' ').title()}: {count} users\n"
    
    report += f"""
## Score Statistics
- Minimum Score: {stats['score_stats']['min']:.4f}
- Maximum Score: {stats['score_stats']['max']:.4f}
- Mean Score: {stats['score_stats']['mean']:.4f}
- Standard Deviation: {stats['score_stats']['std']:.4f}

## Top 10 Most Anomalous Users
"""
    
    for idx, (_, row) in enumerate(anomalies.head(10).iterrows(), 1):
        details = format_anomaly_details(row, feature_names)
        report += f"""
### {idx}. User: {details['username']} ({details['user_id']})
- Anomaly Score: {details['anomaly_score']}
- Type: {details['anomaly_type'].replace('_', ' ').title()}
- Still in Organization: {details['in_org']}
- Period: {details['year_month']}
- Top Contributing Features:
"""
        for feature_info in details['top_features']:
            report += f"  - {feature_info['feature']}: {feature_info['value']}\n"
    
    return report

def export_results_to_csv(results_df: pd.DataFrame, filepath: str):
    """Export results to CSV file"""
    
    # Select relevant columns for export
    export_columns = ['user_id', 'username', 'year_month', 'anomaly_score', 
                     'is_anomaly', 'anomaly_type', 'in_org']
    
    # Add feature columns
    feature_columns = [col for col in results_df.columns 
                      if col.startswith(('sum.', 'count.', 'days.'))]
    export_columns.extend(feature_columns)
    
    # Filter to existing columns
    export_columns = [col for col in export_columns if col in results_df.columns]
    
    # Export
    results_df[export_columns].to_csv(filepath, index=False)
    
def validate_data_quality(data: pd.DataFrame, required_columns: List[str]) -> Dict:
    """Validate data quality and return issues"""
    
    issues = {
        'missing_columns': [],
        'missing_values': {},
        'duplicate_rows': 0,
        'data_types': {},
        'summary': ''
    }
    
    # Check for missing columns
    for col in required_columns:
        if col not in data.columns:
            issues['missing_columns'].append(col)
    
    # Check for missing values
    for col in data.columns:
        null_count = data[col].isnull().sum()
        if null_count > 0:
            issues['missing_values'][col] = null_count
    
    # Check for duplicates
    issues['duplicate_rows'] = data.duplicated().sum()
    
    # Check data types
    for col in data.columns:
        issues['data_types'][col] = str(data[col].dtype)
    
    # Generate summary
    total_issues = (len(issues['missing_columns']) + 
                   len(issues['missing_values']) + 
                   issues['duplicate_rows'])
    
    if total_issues == 0:
        issues['summary'] = "✅ Data quality check passed - no issues found"
    else:
        issues['summary'] = f"⚠️ Data quality issues found: {total_issues} total issues"
    
    return issues
