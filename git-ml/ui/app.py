"""
Streamlit Web Interface for GitHub Security Anomaly Detection
Main application file that provides an interactive dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Some visualizations may not be available.")

from git_analyzer import GitRepositoryAnalyzer
from git_feature_extractor import GitFeatureExtractor
from ml_anomaly_criteria import MLAnomalyCriteriaDetector
from anomaly_detector import GitHubAnomalyDetector
from utils import (
    calculate_anomaly_statistics,
    format_anomaly_details,
    generate_anomaly_report,
    format_large_numbers,
    validate_data_quality
)

# Page configuration
st.set_page_config(
    page_title="GitHub Security Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .anomaly-card {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4444;
    }
    .feature-list {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'audit_logs' not in st.session_state:
        st.session_state.audit_logs = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'detector' not in st.session_state:
        st.session_state.detector = None

def main():
    """Main application function"""
    
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 GitHub Security Anomaly Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application demonstrates security anomaly detection in Enterprise GitHub based on the research paper 
    *"Security Anomaly Detection in Enterprise GitHub"* by Adam Jordan and Yong Chen.
    
    **Key Features:**
    - 🤖 **Machine Learning**: Isolation Forest algorithm with 0.9% contamination rate
    - 📊 **31 Security Features**: Comprehensive feature extraction from audit logs
    - 🚨 **Smart Classification**: Categorizes anomalies as Misuse, Risky Behavior, or Suspicious
    - 📈 **Interactive Visualization**: Real-time charts and analytics
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data generation parameters
        st.subheader("Data Generation")
        num_users = st.slider("Number of Users", 100, 2000, 1000)
        months = st.multiselect(
            "Months to Analyze",
            list(range(1, 13)),
            default=[10, 11, 12],
            format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1]
        )
        
        # ML parameters
        st.subheader("ML Configuration")
        contamination = st.slider("Contamination Rate", 0.001, 0.05, 0.009, 0.001)
        train_month = st.selectbox("Training Month", months[:-1] if len(months) > 1 else months)
        test_month = st.selectbox("Test Month", months[1:] if len(months) > 1 else months)
        
        # Action buttons
        if st.button("Generate Data", type="primary"):
            generate_data(num_users, months)
        
        if st.button("Train Model", disabled=not st.session_state.data_generated):
            train_model(contamination, train_month)
        
        if st.button("Detect Anomalies", disabled=not st.session_state.model_trained):
            detect_anomalies(test_month)
    
    # Main content area
    if not st.session_state.data_generated:
        show_welcome_screen()
    elif not st.session_state.model_trained:
        show_data_overview()
    elif st.session_state.predictions is None:
        show_training_results()
    else:
        show_anomaly_results()

def show_welcome_screen():
    """Show the welcome screen with instructions"""
    
    st.info("👈 Configure your analysis parameters in the sidebar and click 'Generate Data' to begin!")
    
    # Research paper information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Research Paper Implementation")
        st.markdown("""
        This application implements the methodology from:
        
        **"Security Anomaly Detection in Enterprise GitHub"**  
        *Adam Jordan & Yong Chen (PEARC '24)*
        
        **Key Components:**
        - ✅ 31 security-related features
        - ✅ Isolation Forest algorithm  
        - ✅ 0.9% contamination rate
        - ✅ User-month aggregation
        - ✅ Three anomaly classifications
        """)
    
    with col2:
        st.subheader("🎯 Anomaly Classifications")
        
        st.markdown("""
        **🔴 Suspicious**
        - Large git clones/downloads
        - User left the organization
        - Potential data theft
        
        **🟡 Risky Behavior**  
        - Excessive branch protection changes
        - Policy bypasses
        - Process violations
        
        **🟠 Misuse**
        - Large clones/pushes from multiple IPs
        - Improper automation usage
        - Token abuse
        """)

def generate_data(num_users, months):
    """Generate synthetic audit log data"""
    
    with st.spinner(f"Generating audit logs for {num_users} users across {len(months)} months..."):
        try:
            # Generate data
            generator = GitHubDataGenerator(num_users=num_users)
            audit_logs = generator.generate_monthly_data(year=2023, months=months)
            
            # Extract features
            extractor = GitHubFeatureExtractor()
            features = extractor.extract_features(audit_logs)
            
            # Store in session state
            st.session_state.audit_logs = audit_logs
            st.session_state.features = features
            st.session_state.data_generated = True
            
            st.success(f"✅ Generated {len(audit_logs):,} audit log entries with {len(features):,} user-month feature sets!")
            
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")

def show_data_overview():
    """Show overview of generated data"""
    
    st.header("📊 Data Overview")
    
    audit_logs = st.session_state.audit_logs
    features = st.session_state.features
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Audit Logs", format_large_numbers(len(audit_logs)))
    with col2:
        st.metric("Unique Users", format_large_numbers(audit_logs['user_id'].nunique()))
    with col3:
        st.metric("Unique Actions", audit_logs['action'].nunique())
    with col4:
        st.metric("User-Month Features", format_large_numbers(len(features)))
    
    # Data quality check
    st.subheader("Data Quality Assessment")
    extractor = GitHubFeatureExtractor()
    quality_issues = validate_data_quality(features, extractor.all_features)
    
    if quality_issues['summary'].startswith("✅"):
        st.success(quality_issues['summary'])
    else:
        st.warning(quality_issues['summary'])
        with st.expander("View Details"):
            st.json(quality_issues)
    
    # Feature statistics
    st.subheader("Feature Statistics")
    extractor = GitHubFeatureExtractor()
    stats = extractor.get_feature_statistics(features)
    
    if len(stats) > 0:
        st.dataframe(stats, use_container_width=True)
    
    # Sample data
    with st.expander("View Sample Data"):
        st.subheader("Sample Audit Logs")
        st.dataframe(audit_logs.head(100))
        
        st.subheader("Sample Features")
        st.dataframe(features.head(20))

def train_model(contamination, train_month):
    """Train the anomaly detection model"""
    
    with st.spinner("Training Isolation Forest model..."):
        try:
            features = st.session_state.features
            extractor = GitHubFeatureExtractor()
            ml_data, ml_features = extractor.prepare_for_ml(features)
            
            # Filter training data
            train_data = ml_data[ml_data['year_month'] == f'2023-{train_month:02d}']
            
            if len(train_data) == 0:
                st.error(f"No training data found for month {train_month}")
                return
            
            # Train model
            detector = GitHubAnomalyDetector(contamination=contamination)
            training_results = detector.train(train_data, ml_features)
            
            # Store in session state
            st.session_state.detector = detector
            st.session_state.training_results = training_results
            st.session_state.ml_features = ml_features
            st.session_state.model_trained = True
            
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

def show_training_results():
    """Show model training results"""
    
    st.header("🤖 Model Training Results")
    
    results = st.session_state.training_results
    
    # Training metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", format_large_numbers(results['num_samples']))
    with col2:
        st.metric("Features Used", results['num_features'])
    with col3:
        st.metric("Anomalies Detected", results['num_anomalies'])
    with col4:
        st.metric("Anomaly Rate", f"{results['anomaly_rate']:.2%}")
    
    # Model parameters
    st.subheader("Model Configuration")
    detector = st.session_state.detector
    
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        st.info(f"**Algorithm:** Isolation Forest")
        st.info(f"**Contamination Rate:** {detector.contamination:.1%}")
    with config_col2:
        st.info(f"**Random State:** {detector.random_state}")
        st.info(f"**Score Range:** {results['score_range'][0]:.3f} to {results['score_range'][1]:.3f}")
    
    # Feature information
    with st.expander("View Feature Details"):
        st.subheader("Feature Categories")
        
        extractor = GitHubFeatureExtractor()
        
        cat_col1, cat_col2 = st.columns(2)
        with cat_col1:
            st.markdown("**Binary Features (1)**")
            for feature in extractor.binary_features:
                st.markdown(f"- {feature}")
            
            st.markdown("**Count Features (2)**")
            for feature in extractor.count_features:
                st.markdown(f"- {feature}")
        
        with cat_col2:
            st.markdown("**Days Features (2)**")
            for feature in extractor.days_features:
                st.markdown(f"- {feature}")
        
        st.markdown("**Sum Features (26)**")
        for i, feature in enumerate(extractor.sum_features):
            if i % 3 == 0:
                cols = st.columns(3)
            with cols[i % 3]:
                st.markdown(f"- {feature}")

def detect_anomalies(test_month):
    """Detect anomalies in test data"""
    
    with st.spinner("Detecting anomalies..."):
        try:
            features = st.session_state.features
            detector = st.session_state.detector
            extractor = GitHubFeatureExtractor()
            ml_data, ml_features = extractor.prepare_for_ml(features)
            
            # Filter test data
            test_data = ml_data[ml_data['year_month'] == f'2023-{test_month:02d}']
            
            if len(test_data) == 0:
                st.error(f"No test data found for month {test_month}")
                return
            
            # Make predictions
            predictions = detector.predict(test_data)
            
            # Store results
            st.session_state.predictions = predictions
            
            st.success("✅ Anomaly detection completed!")
            
        except Exception as e:
            st.error(f"Error detecting anomalies: {str(e)}")

def show_anomaly_results():
    """Show anomaly detection results"""
    
    st.header("🚨 Anomaly Detection Results")
    
    predictions = st.session_state.predictions
    detector = st.session_state.detector
    
    # Summary statistics
    summary = detector.get_anomaly_summary(predictions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", format_large_numbers(summary['total_samples']))
    with col2:
        st.metric("Anomalies Found", summary['total_anomalies'])
    with col3:
        st.metric("Anomaly Rate", f"{summary['anomaly_rate']:.2%}")
    with col4:
        score_range = summary['score_range']
        st.metric("Score Range", f"{score_range[0]:.3f} to {score_range[1]:.3f}")
    
    # Anomaly type distribution
    if summary['anomaly_types']:
        st.subheader("📊 Anomaly Type Distribution")
        
        type_data = pd.DataFrame(list(summary['anomaly_types'].items()), 
                                columns=['Type', 'Count'])
        
        # Create bar chart if plotly is available
        if PLOTLY_AVAILABLE:
            colors = {
                'suspicious': '#ff4444',
                'risky_behavior': '#ffaa00', 
                'misuse': '#ff6600',
                'unknown': '#888888'
            }
            
            fig = px.bar(
                type_data, 
                x='Type', 
                y='Count',
                color='Type',
                color_discrete_map=colors,
                title="Distribution of Anomaly Types"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(type_data.set_index('Type'))
    
    # Anomaly score distribution
    if PLOTLY_AVAILABLE:
        st.subheader("📈 Anomaly Score Distribution")
        
        normal_scores = predictions[~predictions['is_anomaly']]['anomaly_score']
        anomaly_scores = predictions[predictions['is_anomaly']]['anomaly_score']
        
        fig = go.Figure()
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
            barmode='overlay',
            title='Distribution of Anomaly Scores',
            xaxis_title='Anomaly Score',
            yaxis_title='Count'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    importance = detector.get_feature_importance(predictions)
    
    if len(importance) > 0:
        top_importance = importance.head(10)
        
        if PLOTLY_AVAILABLE:
            fig = px.bar(
                top_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 10 Most Important Features'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(top_importance.set_index('feature')['importance'])
    
    # Detailed anomaly list
    st.subheader("🔍 Anomalous Users")
    
    anomalies = predictions[predictions['is_anomaly']].sort_values('anomaly_score')
    
    if len(anomalies) > 0:
        # Filter and search
        col1, col2 = st.columns(2)
        with col1:
            type_filter = st.selectbox(
                "Filter by Type",
                ['All'] + list(anomalies['anomaly_type'].unique())
            )
        with col2:
            search_user = st.text_input("Search User ID or Username")
        
        # Apply filters
        filtered_anomalies = anomalies.copy()
        if type_filter != 'All':
            filtered_anomalies = filtered_anomalies[filtered_anomalies['anomaly_type'] == type_filter]
        if search_user:
            filtered_anomalies = filtered_anomalies[
                (filtered_anomalies['user_id'].str.contains(search_user, case=False)) |
                (filtered_anomalies['username'].str.contains(search_user, case=False))
            ]
        
        # Display results
        for idx, (_, row) in enumerate(filtered_anomalies.head(20).iterrows(), 1):
            with st.expander(f"{idx}. {row['username']} ({row['user_id']}) - Score: {row['anomaly_score']:.4f}"):
                details = format_anomaly_details(row, st.session_state.ml_features)
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(f"**User ID:** {details['user_id']}")
                    st.markdown(f"**Username:** {details['username']}")
                    st.markdown(f"**Anomaly Type:** {details['anomaly_type'].replace('_', ' ').title()}")
                    st.markdown(f"**Still in Organization:** {details['in_org']}")
                
                with detail_col2:
                    st.markdown(f"**Period:** {details['year_month']}")
                    st.markdown(f"**Anomaly Score:** {details['anomaly_score']}")
                
                if details['top_features']:
                    st.markdown("**Top Contributing Features:**")
                    for feature_info in details['top_features']:
                        st.markdown(f"- {feature_info['feature']}: {feature_info['value']}")
    
    # Export options
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Report"):
            report = generate_anomaly_report(predictions, st.session_state.ml_features)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="anomaly_report.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("Export CSV"):
            csv = predictions.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="anomaly_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
