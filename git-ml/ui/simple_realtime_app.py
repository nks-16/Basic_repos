"""
Real-time Git Repository Security Monitoring Interface - SIMPLIFIED VERSION
"""

import streamlit as st
import time
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime_monitor import RealtimeGitMonitor

class SimpleRealtimeMonitoringApp:
    """Simplified Streamlit app for real-time Git monitoring"""
    
    def __init__(self):
        self.monitor = RealtimeGitMonitor()
        self.setup_page()
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="🔍 Git Security Monitor",
            page_icon="🔍",
            layout="wide"
        )
        
        # Minimal CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .status-active { color: #28a745; font-weight: bold; }
        .status-error { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application logic"""
        
        # Simple header
        st.markdown('<div class="main-header">🔍 Git Security Monitor</div>', unsafe_allow_html=True)
        
        # Check for active sessions
        active_sessions = self.monitor.list_active_sessions()
        
        if not active_sessions:
            self.render_start_page()
        else:
            # Auto-select the first session if none selected
            if 'current_session' not in st.session_state:
                st.session_state.current_session = active_sessions[0]['session_name']
            
            self.render_monitoring_view()
    
    def render_start_page(self):
        """Render page to start monitoring"""
        
        st.markdown("### Start Monitoring")
        
        with st.form("start_monitoring"):
            repo_url = st.text_input(
                "Repository URL",
                placeholder="https://github.com/owner/repo.git",
                help="Enter a Git repository URL to monitor"
            )
            
            submitted = st.form_submit_button("🚀 Start Monitoring", type="primary")
            
            if submitted and repo_url.strip():
                with st.spinner("Setting up monitoring..."):
                    result = self.monitor.start_monitoring_session(repo_url.strip())
                
                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.session_state.current_session = result['session_name']
                    st.success(f"✅ Started monitoring {repo_url}")
                    st.rerun()
    
    def render_monitoring_view(self):
        """Render enhanced but clean monitoring dashboard"""
        
        session_name = st.session_state.current_session
        session_info = self.monitor.get_session_status(session_name)
        
        if 'error' in session_info:
            st.error(f"❌ Session error: {session_info['error']}")
            if st.button("🔄 Restart"):
                del st.session_state.current_session
                st.rerun()
            return
        
        # Header with repository info
        st.markdown(f"### 📊 Monitoring: {session_name}")
        st.markdown(f"**📦 Repository:** {session_info['repo_url']}")
        
        # Key metrics in a more readable format
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if session_info['status'] == 'active' else "🔴"
            status_text = "ACTIVE" if session_info['status'] == 'active' else "STOPPED"
            st.metric("Monitoring Status", f"{status_color} {status_text}")
            
        with col2:
            st.metric("Session Uptime", session_info['uptime'])
            
        with col3:
            alerts_count = session_info['total_alerts']
            st.metric("Security Alerts", alerts_count, delta="🚨" if alerts_count > 0 else "✅")
        
        # Additional important info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Commits Analyzed", session_info.get('historical_commits', 0))
            
        with col2:
            st.metric("Contributors", session_info.get('contributors_analyzed', 0))
            
        with col3:
            new_commits = session_info['new_commits_processed']
            st.metric("New Commits", new_commits, delta=f"+{new_commits}" if new_commits > 0 else None)
        
        # Real-time status with timestamp
        last_check = session_info.get('last_check_time', 'Never')
        if session_info['status'] == 'active':
            if last_check != 'Never':
                st.success(f"🔍 **MONITORING ACTIVE** - Last checked at {last_check}")
            else:
                st.info("🔄 **INITIALIZING** - Setting up monitoring...")
        else:
            st.error("⏸️ **MONITORING STOPPED**")
        
        # Session details
        created_at = session_info.get('created_at', 'Unknown')
        if created_at != 'Unknown':
            try:
                from datetime import datetime
                created_dt = datetime.fromisoformat(created_at)
                created_str = created_dt.strftime("%Y-%m-%d %H:%M")
                st.caption(f"� Session started: {created_str}")
            except:
                st.caption(f"📅 Session started: {created_at}")
        
        st.divider()
        
        # Security alerts section with better display
        st.markdown("### 🚨 Security Analysis")
        
        recent_alerts = self.monitor.get_recent_alerts(session_name, limit=10)
        
        if recent_alerts:
            st.warning(f"⚠️ **{len(recent_alerts)} Security Events Detected**")
            
            # Show recent alerts in an organized way
            for i, alert in enumerate(reversed(recent_alerts[-5:])):  # Show last 5 alerts
                with st.expander(f"🚨 Alert #{len(recent_alerts)-i}: {alert.get('anomaly_type', 'Security Event')}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**🔗 Commit:** `{alert.get('commit_hash', 'Unknown')[:12]}`")
                        st.markdown(f"**👤 Author:** {alert.get('author', 'Unknown')}")
                        st.markdown(f"**📝 Message:** {alert.get('message', 'No message')[:100]}...")
                        
                        # Show explanation if available
                        explanation = alert.get('explanation', '')
                        if explanation:
                            st.markdown(f"**🔍 Analysis:** {explanation}")
                    
                    with col2:
                        # Risk level with color coding
                        risk = alert.get('risk_level', 'Medium')
                        risk_colors = {'Critical': '🔴', 'High': '�', 'Medium': '🟡', 'Low': '🟢'}
                        risk_color = risk_colors.get(risk, '🔵')
                        st.markdown(f"**Risk Level:**")
                        st.markdown(f"## {risk_color} {risk}")
                        
                        # Timestamp
                        timestamp = alert.get('timestamp', 'Unknown')
                        if timestamp != 'Unknown':
                            try:
                                alert_dt = datetime.fromisoformat(timestamp)
                                time_str = alert_dt.strftime("%H:%M:%S")
                                st.caption(f"🕐 {time_str}")
                            except:
                                st.caption(f"🕐 {timestamp}")
                        
                        # Anomaly score
                        score = alert.get('anomaly_score', 0)
                        if score > 0:
                            st.caption(f"📊 Score: {score:.3f}")
            
            if len(recent_alerts) > 5:
                st.info(f"📋 Showing 5 most recent alerts. Total: {len(recent_alerts)}")
                
        else:
            st.success("✅ **All Clear** - No security anomalies detected")
            st.info("🔍 System is actively monitoring for suspicious activity")
            
            # Show what we're watching for
            with st.expander("🔍 What we monitor", expanded=False):
                st.markdown("""
                **Real-time detection of:**
                - 🕐 Unusual commit timing patterns
                - 📊 Abnormal code change volumes  
                - 🔄 Suspicious commit frequencies
                - 👤 Anomalous user behavior
                - 📁 Unusual file modification patterns
                """)
        
        st.divider()
        
        # Control panel
        st.markdown("### 🎛️ Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        with col2:
            if st.button("� Full Report"):
                st.info("Full report feature coming soon!")
        
        with col3:
            if st.button("�🗑️ Stop Monitoring"):
                if st.session_state.get('confirm_stop', False):
                    self.monitor.close_session(session_name)
                    del st.session_state.current_session
                    if 'confirm_stop' in st.session_state:
                        del st.session_state.confirm_stop
                    st.rerun()
                else:
                    st.session_state.confirm_stop = True
                    st.warning("⚠️ Click again to confirm stopping")
        
        with col4:
            auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        # Auto refresh with better feedback
        if auto_refresh and session_info['status'] == 'active':
            with st.empty():
                for remaining in range(30, 0, -1):
                    st.caption(f"🔄 Next refresh in {remaining}s...")
                    time.sleep(1)
            st.rerun()

if __name__ == "__main__":
    app = SimpleRealtimeMonitoringApp()
    app.run()
