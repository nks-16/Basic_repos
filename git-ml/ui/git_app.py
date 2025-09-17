"""
Git Repository Security Analyzer - Streamlit Web Interface
Analyzes real Git repositories for security anomalies
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
from utils import (
    calculate_anomaly_statistics,
    format_large_numbers
)

# Page configuration
st.set_page_config(
    page_title="Git Security Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

def cleanup_temp_repos_on_startup():
    """Clean up temporary repository clones on startup"""
    import shutil
    temp_repos_path = os.path.join(os.path.dirname(__file__), '..', 'temp_repos')
    if os.path.exists(temp_repos_path):
        try:
            shutil.rmtree(temp_repos_path)
            if 'cleanup_message' not in st.session_state:
                st.session_state.cleanup_message = "🧹 Cleaned up previous repository clones"
        except Exception as e:
            if 'cleanup_error' not in st.session_state:
                st.session_state.cleanup_error = f"Warning: Could not clean temp repos: {e}"

# Clean up temp repos on startup
cleanup_temp_repos_on_startup()

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
    .success-card {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00aa00;
    }
</style>
""", unsafe_allow_html=True)

def show_cleanup_status():
    """Show cleanup status messages"""
    if 'cleanup_message' in st.session_state:
        st.success(st.session_state.cleanup_message)
        del st.session_state.cleanup_message
    
    if 'cleanup_error' in st.session_state:
        st.warning(st.session_state.cleanup_error)
        del st.session_state.cleanup_error

def init_session_state():
    """Initialize session state variables"""
    if 'repo_analyzed' not in st.session_state:
        st.session_state.repo_analyzed = False
    if 'features_extracted' not in st.session_state:
        st.session_state.features_extracted = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'audit_logs' not in st.session_state:
        st.session_state.audit_logs = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'anomaly_results' not in st.session_state:
        st.session_state.anomaly_results = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'learned_criteria' not in st.session_state:
        st.session_state.learned_criteria = None

def main():
    """Main application function"""
    
    init_session_state()
    
    # Show cleanup status
    show_cleanup_status()
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Git Repository Security Analyzer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes Git repositories for security anomalies using machine learning.
    
    **Key Features:**
    - 🔍 **Real Git Analysis**: Analyzes actual Git commit logs and patterns
    - 🤖 **Smart ML Learning**: Automatically learns optimal anomaly detection criteria
    - 📊 **Advanced Features**: 20+ security-related features extracted from Git data
    - 🚨 **Intelligent Classification**: ML-driven anomaly categorization
    - 📈 **Interactive Visualization**: Real-time charts and analytics
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Repository Analysis")
        
        # Git repository input
        st.subheader("🔗 Git Repository")
        git_url = st.text_input(
            "Git Repository URL",
            placeholder="https://github.com/owner/repo.git",
            help="Enter a public Git repository URL to analyze"
        )
        
        # Analysis parameters
        st.subheader("⚙️ Analysis Settings")
        
        # Date range for analysis
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp.now().date() - pd.Timedelta(days=90)
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=pd.Timestamp.now().date()
            )
        
        # ML parameters
        contamination = st.slider(
            "Anomaly Detection Sensitivity",
            0.001, 0.1, 0.01, 0.001,
            help="Higher values detect more anomalies (less strict)"
        )
        
        # Action buttons
        if st.button("🔍 Analyze Repository", type="primary"):
            if git_url:
                analyze_repository(git_url, start_date, end_date)
            else:
                st.error("Please enter a Git repository URL")
        
        if st.button("🤖 Learn Anomaly Patterns", disabled=not st.session_state.features_extracted):
            learn_anomaly_patterns(contamination)
        
        if st.button("🚨 Detect Anomalies", disabled=not st.session_state.model_trained):
            detect_anomalies()
    
    # Main content area
    if not st.session_state.repo_analyzed:
        show_welcome_screen()
    elif not st.session_state.features_extracted:
        show_analysis_progress()
    elif not st.session_state.model_trained:
        show_feature_overview()
    elif not st.session_state.anomaly_results:
        show_learned_criteria()
    else:
        show_anomaly_results()

def show_welcome_screen():
    """Show the welcome screen with instructions"""
    
    st.info("👈 Enter a Git repository URL in the sidebar to begin analysis!")
    
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What We Analyze")
        st.markdown("""
        **Git Commit Patterns:**
        - ✅ Commit frequency and timing
        - ✅ Code change patterns (lines added/deleted)
        - ✅ File modification behaviors
        - ✅ Off-hours and weekend activity
        - ✅ Large commits and sensitive file changes
        
        **Security Indicators:**
        - ✅ Unusual time patterns
        - ✅ Rapid commit sequences (automation detection)
        - ✅ Repository access diversity
        - ✅ External domain activity
        - ✅ Sensitive content modifications
        """)
    
    with col2:
        st.subheader("🤖 ML-Driven Detection")
        
        st.markdown("""
        **Intelligent Learning:**
        - 🧠 **Automatic Criteria Learning**: ML analyzes your repo patterns
        - 📊 **Threshold Optimization**: Finds optimal detection thresholds
        - 🔍 **Pattern Discovery**: Identifies distinct anomaly types
        - ⚡ **Adaptive Rules**: Criteria tailored to your repository
        
        **Anomaly Types Detected:**
        - 🔴 **Critical**: Data exfiltration, security breaches
        - 🟡 **High Risk**: Suspicious patterns, policy violations  
        - 🟠 **Medium Risk**: Unusual but explainable behavior
        - 🟢 **Low Risk**: Minor deviations from normal patterns
        """)
    
    # Example repositories
    st.subheader("🎲 Try These Example Repositories")
    
    example_repos = [
        "https://github.com/octocat/Hello-World.git",
        "https://github.com/microsoft/vscode.git",
        "https://github.com/facebook/react.git",
        "https://github.com/torvalds/linux.git"
    ]
    
    cols = st.columns(len(example_repos))
    for i, repo in enumerate(example_repos):
        with cols[i]:
            repo_name = repo.split('/')[-1].replace('.git', '')
            if st.button(f"📁 {repo_name}", key=f"example_{i}"):
                st.session_state.example_repo = repo
                st.rerun()

def analyze_repository(git_url: str, start_date, end_date):
    """Analyze a Git repository"""
    
    with st.spinner(f"Analyzing repository: {git_url}..."):
        try:
            # Initialize analyzer
            analyzer = GitRepositoryAnalyzer()
            st.session_state.analyzer = analyzer
            
            # Clone repository
            local_path = analyzer.clone_repository(git_url)
            st.success(f"✅ Repository cloned to: {local_path}")
            
            # Extract audit logs
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date)
            
            audit_logs = analyzer.extract_git_logs(start_datetime, end_datetime)
            
            if len(audit_logs) == 0:
                st.error("No commit data found in the specified date range. Try expanding the date range.")
                return
            
            # Store results
            st.session_state.audit_logs = audit_logs
            st.session_state.repo_analyzed = True
            
            # Extract features
            extractor = GitFeatureExtractor()
            features = extractor.extract_features(audit_logs)
            
            st.session_state.features = features
            st.session_state.features_extracted = True
            
            st.success(f"✅ Analysis complete! Found {len(audit_logs):,} activities from {len(features):,} contributors.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error analyzing repository: {str(e)}")
            if st.session_state.analyzer:
                st.session_state.analyzer.cleanup()

def show_analysis_progress():
    """Show analysis progress and results"""
    
    st.header("📊 Repository Analysis Complete")
    
    audit_logs = st.session_state.audit_logs
    features = st.session_state.features
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Activities", format_large_numbers(len(audit_logs)))
    with col2:
        st.metric("Contributors", format_large_numbers(audit_logs['user_id'].nunique()))
    with col3:
        st.metric("Repositories", audit_logs['repository'].nunique())
    with col4:
        st.metric("Analysis Period", f"{len(features):,} user-periods")
    
    # Activity breakdown
    st.subheader("Activity Breakdown")
    
    activity_counts = audit_logs['action'].value_counts()
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=activity_counts.values, 
            y=activity_counts.index,
            orientation='h',
            title="Types of Git Activities Detected"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(activity_counts)
    
    # Contributors analysis
    st.subheader("Top Contributors")
    
    contributor_stats = audit_logs.groupby(['user_id', 'username']).agg({
        'action': 'count',
        'timestamp': ['min', 'max']
    }).round(2)
    
    contributor_stats.columns = ['Total_Activities', 'First_Activity', 'Last_Activity']
    contributor_stats = contributor_stats.sort_values('Total_Activities', ascending=False)
    
    st.dataframe(contributor_stats.head(10), use_container_width=True)
    
    # Time-based analysis
    if PLOTLY_AVAILABLE:
        st.subheader("Activity Timeline")
        
        # Daily activity
        audit_logs['date'] = pd.to_datetime(audit_logs['timestamp']).dt.date
        daily_activity = audit_logs.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_activity, 
            x='date', 
            y='count',
            title="Daily Activity Levels"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data
    with st.expander("View Sample Data"):
        st.subheader("Sample Git Activities")
        display_columns = ['timestamp', 'username', 'action', 'repository', 'commit_hash']
        available_columns = [col for col in display_columns if col in audit_logs.columns]
        st.dataframe(audit_logs[available_columns].head(20), use_container_width=True)

def learn_anomaly_patterns(contamination: float):
    """Learn anomaly patterns from the repository data"""
    
    with st.spinner("Learning anomaly detection patterns from your repository..."):
        try:
            features = st.session_state.features
            extractor = GitFeatureExtractor()
            ml_data, feature_names = extractor.prepare_for_ml(features)
            
            # Initialize ML criteria detector
            criteria_detector = MLAnomalyCriteriaDetector(contamination=contamination)
            
            # Learn criteria
            learned_criteria = criteria_detector.learn_anomaly_criteria(ml_data, feature_names)
            
            # Store results
            st.session_state.criteria_detector = criteria_detector
            st.session_state.learned_criteria = learned_criteria
            st.session_state.model_trained = True
            
            st.success("✅ Anomaly detection patterns learned successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error learning patterns: {str(e)}")

def show_feature_overview():
    """Show extracted features overview"""
    
    st.header("🔧 Extracted Features")
    
    features = st.session_state.features
    extractor = GitFeatureExtractor()
    
    # Feature statistics
    stats = extractor.get_feature_statistics(features)
    
    if len(stats) > 0:
        st.subheader("Feature Statistics")
        st.dataframe(stats, use_container_width=True)
        
        # Feature distribution
        if PLOTLY_AVAILABLE:
            st.subheader("Feature Distributions")
            
            # Select top features to visualize
            top_features = stats.nlargest(6, 'Mean')['Feature'].tolist()
            
            for i in range(0, len(top_features), 2):
                cols = st.columns(2)
                
                for j, col in enumerate(cols):
                    if i + j < len(top_features):
                        feature_name = top_features[i + j]
                        if feature_name in features.columns:
                            with col:
                                values = features[feature_name].fillna(0)
                                fig = px.histogram(
                                    x=values,
                                    title=f"Distribution of {feature_name}",
                                    nbins=30
                                )
                                st.plotly_chart(fig, use_container_width=True)

def show_learned_criteria():
    """Show learned anomaly detection criteria"""
    
    st.header("🧠 Learned Anomaly Detection Criteria")
    
    learned_criteria = st.session_state.learned_criteria
    
    # Training statistics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = learned_criteria['training_stats']
    with col1:
        st.metric("Samples Analyzed", format_large_numbers(stats['total_samples']))
    with col2:
        st.metric("Anomalies Found", stats['anomalies_found'])
    with col3:
        st.metric("Anomaly Rate", f"{stats['anomaly_rate']:.2%}")
    with col4:
        st.metric("Pattern Types", stats['num_profiles'])
    
    # Learned thresholds
    st.subheader("🎯 Learned Thresholds")
    
    thresholds_data = []
    for feature, threshold_info in learned_criteria['feature_thresholds'].items():
        thresholds_data.append({
            'Feature': feature,
            'Threshold': threshold_info['threshold'],
            'Confidence': threshold_info['confidence'],
            'Separation_Score': threshold_info.get('separation_score', 0)
        })
    
    if thresholds_data:
        thresholds_df = pd.DataFrame(thresholds_data)
        st.dataframe(thresholds_df, use_container_width=True)
    
    # Discovered patterns
    st.subheader("🔍 Discovered Anomaly Patterns")
    
    for profile_name, profile in learned_criteria['anomaly_profiles'].items():
        with st.expander(f"Pattern: {profile['pattern_description']} (Risk: {profile['risk_level'].title()})"):
            st.markdown(f"**Profile ID:** {profile_name}")
            st.markdown(f"**Sample Size:** {profile['size']} users")
            st.markdown(f"**Risk Level:** {profile['risk_level'].title()}")
            st.markdown(f"**Description:** {profile['pattern_description']}")
            
            if profile['key_features']:
                st.markdown("**Key Characteristics:**")
                for feature, stats in profile['key_features'].items():
                    st.markdown(f"- **{feature}**: Typical value ~{stats['typical_value']:.2f}, Max observed: {stats['max_observed']:.2f}")

def detect_anomalies():
    """Detect anomalies using learned criteria"""
    
    with st.spinner("Detecting anomalies using learned patterns..."):
        try:
            features = st.session_state.features
            criteria_detector = st.session_state.criteria_detector
            extractor = GitFeatureExtractor()
            
            ml_data, feature_names = extractor.prepare_for_ml(features)
            
            # Apply learned anomaly detection
            results = []
            
            for idx, row in ml_data.iterrows():
                # Get anomaly classification
                classification = criteria_detector.classify_anomaly_with_learned_rules(row)
                
                result_row = row.to_dict()
                result_row.update({
                    'anomaly_type': classification['type'],
                    'risk_level': classification['risk_level'],
                    'confidence': classification['confidence'],
                    'reasons': '; '.join(classification['reasons']),
                    'is_anomaly': classification['confidence'] > 0.3
                })
                
                results.append(result_row)
            
            results_df = pd.DataFrame(results)
            st.session_state.anomaly_results = results_df
            
            st.success("✅ Anomaly detection complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error detecting anomalies: {str(e)}")

def show_anomaly_results():
    """Show anomaly detection results"""
    
    st.header("🚨 Anomaly Detection Results")
    
    results = st.session_state.anomaly_results
    
    # Summary metrics
    anomalies = results[results['is_anomaly']]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(results))
    with col2:
        st.metric("Anomalies Found", len(anomalies))
    with col3:
        st.metric("Anomaly Rate", f"{len(anomalies)/len(results):.1%}")
    with col4:
        avg_confidence = anomalies['confidence'].mean() if len(anomalies) > 0 else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Risk level distribution
    if len(anomalies) > 0:
        st.subheader("🎯 Risk Level Distribution")
        
        risk_counts = anomalies['risk_level'].value_counts()
        
        if PLOTLY_AVAILABLE:
            colors = {
                'critical': '#ff0000',
                'high': '#ff6600',
                'medium': '#ffaa00',
                'low': '#88cc88'
            }
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Anomalies by Risk Level",
                color=risk_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed anomaly list
        st.subheader("🔍 Detected Anomalies")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ['All'] + list(anomalies['risk_level'].unique())
            )
        with col2:
            confidence_threshold = st.slider(
                "Minimum Confidence",
                0.0, 1.0, 0.3, 0.1
            )
        
        # Apply filters
        filtered_anomalies = anomalies.copy()
        if risk_filter != 'All':
            filtered_anomalies = filtered_anomalies[filtered_anomalies['risk_level'] == risk_filter]
        filtered_anomalies = filtered_anomalies[filtered_anomalies['confidence'] >= confidence_threshold]
        
        # Sort by confidence
        filtered_anomalies = filtered_anomalies.sort_values('confidence', ascending=False)
        
        # Display results
        for idx, (_, row) in enumerate(filtered_anomalies.head(20).iterrows(), 1):
            risk_color = {
                'critical': '🔴',
                'high': '🟡', 
                'medium': '🟠',
                'low': '🟢'
            }.get(row['risk_level'], '⚫')
            
            with st.expander(f"{risk_color} {idx}. {row.get('username', row['user_id'])} - {row['risk_level'].title()} Risk (Confidence: {row['confidence']:.2f})"):
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown(f"**User ID:** {row['user_id']}")
                    st.markdown(f"**Risk Level:** {row['risk_level'].title()}")
                    st.markdown(f"**Confidence:** {row['confidence']:.2f}")
                    st.markdown(f"**Period:** {row.get('year_month', 'All Time')}")
                
                with detail_col2:
                    st.markdown(f"**Anomaly Type:** {row['anomaly_type']}")
                    if row.get('email'):
                        st.markdown(f"**Email:** {row['email']}")
                
                if row['reasons']:
                    st.markdown("**Reasons for Detection:**")
                    reasons = row['reasons'].split('; ')
                    for reason in reasons[:5]:  # Show top 5 reasons
                        st.markdown(f"- {reason}")
    else:
        st.info("🎉 No significant anomalies detected in this repository!")
        st.markdown("This could mean:")
        st.markdown("- The repository has healthy, normal development patterns")
        st.markdown("- The detection sensitivity might need adjustment")
        st.markdown("- The analysis period might be too short")
    
    # Export options
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Download Full Results"):
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="git_anomaly_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔍 Download Anomalies Only"):
            csv = anomalies.to_csv(index=False)
            st.download_button(
                label="Download Anomalies CSV",
                data=csv,
                file_name="git_anomalies_only.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
