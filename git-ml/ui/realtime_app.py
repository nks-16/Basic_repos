"""
Real-time Git Repository Security Monitoring Interface
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime_monitor import RealtimeGitMonitor

class RealtimeMonitoringApp:
    """Streamlit app for real-time Git monitoring"""
    
    def __init__(self):
        self.monitor = RealtimeGitMonitor()
        self.setup_page()
        self.cleanup_temp_repos_on_startup()
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="🔍 Real-time Git Security Monitor",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .session-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .alert-danger {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .alert-warning {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .alert-info {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def cleanup_temp_repos_on_startup(self):
        """Clean up temporary repository clones on startup"""
        import shutil
        import subprocess
        temp_repos_path = os.path.join(os.path.dirname(__file__), '..', 'temp_repos')
        if os.path.exists(temp_repos_path):
            try:
                # Windows-specific: Remove read-only attributes first
                if os.name == 'nt':
                    try:
                        subprocess.run(['attrib', '-R', f'{temp_repos_path}\\*.*', '/S'], 
                                     capture_output=True, shell=True)
                    except:
                        pass  # Ignore if attrib fails
                
                shutil.rmtree(temp_repos_path)
                # Don't show cleanup message to avoid clutter
            except Exception:
                # Silently ignore cleanup errors - they're not critical
                pass
    
    def run(self):
        """Main application logic"""
        
        # Header
        st.markdown('<div class="main-header">🔍 Real-time Git Security Monitor</div>', unsafe_allow_html=True)
        
        # Sidebar for session management
        self.render_sidebar()
        
        # Main content area
        if 'current_session' not in st.session_state:
            self.render_welcome_page()
        else:
            self.render_monitoring_dashboard()
    
    def render_sidebar(self):
        """Render session management sidebar"""
        with st.sidebar:
            st.header("🎛️ Session Management")
            
            # Start new session
            with st.expander("🚀 Start New Session", expanded='current_session' not in st.session_state):
                repo_url = st.text_input(
                    "Git Repository URL",
                    placeholder="https://github.com/owner/repo.git",
                    help="Enter a public Git repository URL to monitor"
                )
                
                # Add examples
                st.markdown("**📝 Example URLs:**")
                example_urls = [
                    "https://github.com/torvalds/linux.git",
                    "https://github.com/microsoft/vscode.git", 
                    "https://github.com/python/cpython.git"
                ]
                
                for example_url in example_urls:
                    if st.button(f"📋 {example_url}", key=f"example_{example_url}"):
                        st.session_state.example_url = example_url
                        st.rerun()
                
                # Use example URL if selected
                if 'example_url' in st.session_state:
                    repo_url = st.session_state.example_url
                    del st.session_state.example_url
                
                session_name = st.text_input(
                    "Session Name (Optional)",
                    placeholder="my-project-monitor",
                    help="Custom name for this monitoring session"
                )
                
                # URL validation info
                st.info("""
                **💡 URL Tips:**
                - Use `.git` URLs like: `https://github.com/owner/repo.git`
                - Convert GitHub web URLs: `https://github.com/owner/repo/tree/branch` → `https://github.com/owner/repo.git`
                - Only public repositories are supported
                """)
                
                if st.button("🎯 Start Monitoring", type="primary"):
                    if repo_url.strip():
                        # Show different messages based on if session exists
                        with st.spinner("🔍 Setting up monitoring session..."):
                            result = self.monitor.start_monitoring_session(
                                repo_url.strip(), 
                                session_name.strip() if session_name.strip() else None
                            )
                        
                        if 'error' in result:
                            st.error(f"{result['error']}")
                        elif 'info' in result:
                            # Existing session
                            st.info(f"ℹ️ {result['info']}")
                            st.session_state.current_session = result['session_name']
                            st.rerun()
                        else:
                            # New session
                            st.session_state.current_session = result['session_name']
                            st.success(f"✅ {result['message']}")
                            st.balloons()  # Celebration for new session
                            st.rerun()
                    else:
                        st.warning("⚠️ Please enter a repository URL")
            
            # Active sessions
            active_sessions = self.monitor.list_active_sessions()
            
            if active_sessions:
                st.header("📊 Active Sessions")
                
                for session in active_sessions:
                    with st.container():
                        st.markdown(f"""
                        <div class="session-card">
                            <strong>{session['session_name']}</strong><br>
                            📦 {session['repo_url']}<br>
                            ⏱️ {session['uptime']}<br>
                            🚨 {session['alerts_count']} alerts<br>
                            📝 {session['new_commits']} new commits
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button(f"👀 View", key=f"view_{session['session_name']}"):
                                st.session_state.current_session = session['session_name']
                                st.rerun()
                        
                        with col2:
                            if st.button(f"🗑️ Close", key=f"close_{session['session_name']}"):
                                result = self.monitor.close_session(session['session_name'])
                                st.success(result['message'])
                                if st.session_state.get('current_session') == session['session_name']:
                                    del st.session_state.current_session
                                st.rerun()
            
            # Refresh button
            if st.button("🔄 Refresh Sessions"):
                st.rerun()
    
    def render_welcome_page(self):
        """Render simple welcome page when no session is active"""
        
        st.markdown("## 🔍 Git Security Monitor")
        st.markdown("Real-time security monitoring for Git repositories")
        
        # Check for existing sessions
        active_sessions = self.monitor.list_active_sessions()
        
        if active_sessions:
            st.markdown("### 📊 Active Monitoring Sessions")
            for session in active_sessions:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{session['session_name']}**")
                        st.caption(f"📦 {session['repo_url']}")
                        st.caption(f"⏱️ Running for {session['uptime']} • 🚨 {session['alerts_count']} alerts")
                    
                    with col2:
                        if st.button(f"👀 View", key=f"view_{session['session_name']}"):
                            st.session_state.current_session = session['session_name']
                            st.rerun()
                    
                    with col3:
                        if st.button(f"🗑️ Stop", key=f"close_{session['session_name']}"):
                            result = self.monitor.close_session(session['session_name'])
                            st.success(result['message'])
                            st.rerun()
                
                st.divider()
        
        else:
            st.info("👈 **Start monitoring by entering a repository URL in the sidebar**")
            
            # Show simple instructions
            with st.expander("ℹ️ How to use", expanded=False):
                st.markdown("""
                1. Enter a Git repository URL in the sidebar
                2. Click "Start Monitoring" 
                3. View real-time security analysis
                
                **Supported URLs:**
                - `https://github.com/owner/repo.git`
                - `https://github.com/owner/repo`
                """)
    
    def render_monitoring_dashboard(self):
        """Render the main monitoring dashboard"""
        
        session_name = st.session_state.current_session
        session_info = self.monitor.get_session_status(session_name)
        
        if 'error' in session_info:
            st.error(f"❌ {session_info['error']}")
            del st.session_state.current_session
            st.rerun()
            return
        
        # Session header
        st.markdown(f"## 📊 Monitoring: `{session_name}`")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "⚡ Status", 
                session_info['status'].upper(),
                delta="Active" if session_info['status'] == 'active' else 'Inactive'
            )
        
        with col2:
            st.metric(
                "⏱️ Uptime", 
                session_info['uptime']
            )
        
        with col3:
            st.metric(
                "🚨 Total Alerts", 
                session_info['total_alerts']
            )
        
        with col4:
            st.metric(
                "📝 New Commits", 
                session_info['new_commits_processed']
            )
        
        # Repository info
        st.markdown(f"**📦 Repository:** {session_info['repo_url']}")
        st.markdown(f"**📅 Started:** {session_info['created_at']}")
        st.markdown(f"**👥 Contributors Analyzed:** {session_info['contributors_analyzed']}")
        st.markdown(f"**📊 Historical Commits:** {session_info['historical_commits']}")
        
        st.divider()
        
        # Live alerts section
        st.markdown("## 🚨 Real-Time Security Monitoring")
        
        # Monitoring status banner
        monitoring_status = session_info.get('monitoring_active', True)
        if monitoring_status:
            st.success("🟢 **MONITORING ACTIVE** - Scanning for new commits every 30 seconds")
            
            # Show monitoring activity
            last_check = session_info.get('last_check_time', 'Never')
            if last_check != 'Never':
                st.info(f"🔍 **Last Scan:** {last_check} • **Status:** Checking for new commits...")
            else:
                st.info(f"🔍 **Status:** Initializing monitoring system...")
        else:
            st.warning("🟡 **MONITORING PAUSED** - Click refresh to resume")
        
        # Auto-refresh toggle
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**📡 Live Feed** - New anomalies appear automatically")
        with col2:
            auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        with col3:
            if st.button("🔄 Check Now"):
                st.rerun()
        
        # Show last check time
        last_check = session_info.get('last_check_time', 'Never')
        if last_check != 'Never':
            st.caption(f"🕐 Last checked: {last_check}")
        
        # Get recent alerts
        recent_alerts = self.monitor.get_recent_alerts(session_name, limit=20)
        
        # Show current monitoring status
        if recent_alerts:
            st.markdown(f"### 🔍 Found {len(recent_alerts)} Security Events")
            for alert in reversed(recent_alerts):  # Show newest first
                self.render_alert_card(alert)
        else:
            # Show positive status when no alerts
            st.markdown("### 🛡️ Security Status: All Clear")
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.success("✅ **No Anomalies Detected**")
                st.markdown("Your repository is showing normal activity patterns.")
            with col_b:
                st.info("� **Active Monitoring**")
                st.markdown("System is continuously scanning for suspicious activity.")
            
            # Show monitoring statistics
            st.markdown("#### 📊 Monitoring Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📝 Commits Scanned", session_info.get('historical_commits', 0))
            with col2:
                st.metric("👥 Contributors", session_info.get('contributors_analyzed', 0))
            with col3:
                st.metric("🔍 Checks Performed", session_info.get('new_commits_processed', 0))
            
            # Show what we're monitoring for
            with st.expander("🔎 What we're monitoring for", expanded=False):
                st.markdown("""
                **Real-time detection of:**
                - 🕐 **Timing Anomalies**: Commits at unusual hours or rapid sequences
                - 📊 **Code Volume**: Abnormally large commits or file changes  
                - 🔄 **Frequency Patterns**: Unusual commit frequencies or bursts
                - 👤 **User Behavior**: Deviations from normal contributor patterns
                - 📁 **File Access**: Suspicious file modification patterns
                - 🚨 **Security Indicators**: High-risk actions and sensitive file changes
                """)
                
            # Show real-time status
            st.markdown("#### ⚡ Real-Time Status")
            if monitoring_status:
                st.markdown("🟢 **System is actively monitoring** - New commits will be analyzed immediately")
                st.markdown("🔄 **Next check:** Automatically scanning every 30 seconds")
            else:
                st.markdown("🟡 **Monitoring paused** - Use refresh button to resume")
        
        # Auto-refresh logic with visual feedback
        if auto_refresh and monitoring_status:
            with st.empty():
                for i in range(30, 0, -1):
                    st.caption(f"🔄 Next check in {i} seconds...")
                    time.sleep(1)
            st.rerun()
        
        # Statistics and visualizations
        if recent_alerts:
            st.divider()
            self.render_analytics_section(recent_alerts)
    
    def render_alert_card(self, alert):
        """Render an individual alert card"""
        
        # Determine alert styling based on risk level
        risk_colors = {
            'Critical': 'alert-danger',
            'High': 'alert-danger', 
            'Medium': 'alert-warning',
            'Low': 'alert-info'
        }
        
        risk_icons = {
            'Critical': '🔴',
            'High': '🟠', 
            'Medium': '🟡',
            'Low': '🟢'
        }
        
        alert_class = risk_colors.get(alert.get('risk_level', 'Medium'), 'alert-info')
        risk_icon = risk_icons.get(alert.get('risk_level', 'Medium'), '🔵')
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(alert['timestamp'])
            time_str = timestamp.strftime("%H:%M:%S")
            date_str = timestamp.strftime("%Y-%m-%d")
        except:
            time_str = "Unknown"
            date_str = "Unknown"
        
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{risk_icon} {alert.get('anomaly_type', 'Security Anomaly')} - {alert.get('risk_level', 'Medium')} Risk</strong><br>
            <strong>🕒 {time_str}</strong> | <strong>📅 {date_str}</strong><br>
            <strong>📝 Commit:</strong> {alert.get('commit_hash', 'Unknown')}<br>
            <strong>👤 Author:</strong> {alert.get('author', 'Unknown')}<br>
            <strong>📊 Score:</strong> {alert.get('anomaly_score', 0):.3f}<br>
            <strong>💬 Message:</strong> {alert.get('message', 'No message')}<br>
            <strong>🔍 Explanation:</strong> {alert.get('explanation', 'No explanation')}
        </div>
        """, unsafe_allow_html=True)
        
        # Contributing features
        features = alert.get('contributing_features', [])
        if features:
            with st.expander(f"📋 Contributing Factors ({len(features)})"):
                for feature in features:
                    st.write(f"• {feature}")
    
    def render_analytics_section(self, alerts):
        """Render analytics and visualizations"""
        
        st.markdown("## 📊 Alert Analytics")
        
        # Convert alerts to DataFrame
        df_alerts = pd.DataFrame(alerts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution
            if 'risk_level' in df_alerts.columns:
                risk_counts = df_alerts['risk_level'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="🎯 Risk Level Distribution",
                    color_discrete_map={
                        'Critical': '#d32f2f',
                        'High': '#f57c00', 
                        'Medium': '#fbc02d',
                        'Low': '#388e3c'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anomaly type distribution
            if 'anomaly_type' in df_alerts.columns:
                type_counts = df_alerts['anomaly_type'].value_counts()
                fig = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    title="🔍 Anomaly Type Distribution",
                    labels={'x': 'Anomaly Type', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Timeline of alerts
        if 'timestamp' in df_alerts.columns:
            try:
                df_alerts['datetime'] = pd.to_datetime(df_alerts['timestamp'])
                df_alerts['hour'] = df_alerts['datetime'].dt.hour
                
                hourly_counts = df_alerts.groupby('hour').size()
                
                fig = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="⏰ Alert Timeline (by Hour)",
                    labels={'x': 'Hour of Day', 'y': 'Alert Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not render timeline: {e}")

def main():
    """Run the Streamlit app"""
    app = RealtimeMonitoringApp()
    app.run()

if __name__ == "__main__":
    main()
