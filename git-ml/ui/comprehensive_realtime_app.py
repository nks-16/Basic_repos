#!/usr/bin/env python3
"""
Comprehensive Real-time Git Repository Security Monitoring Interface
Enhanced dashboard with detailed repository information and metrics
"""

import time
import streamlit as st
from datetime import datetime
import sys
import os

# Add the parent directory to sys.path to import from src
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.realtime_monitor import RealtimeGitMonitor

class ComprehensiveRealtimeMonitoringApp:
    def __init__(self):
        st.set_page_config(
            page_title="🔒 Repository Security Monitor",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Force refresh the monitor instance to pick up new methods
        if 'force_refresh' in st.session_state or 'monitor_instance' not in st.session_state:
            st.session_state.monitor_instance = RealtimeGitMonitor()
            if 'force_refresh' in st.session_state:
                del st.session_state.force_refresh
        
        self.monitor = st.session_state.monitor_instance
        
        # Style the app
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .alert-critical { border-left-color: #dc3545; }
        .alert-high { border-left-color: #fd7e14; }
        .alert-medium { border-left-color: #ffc107; }
        .alert-low { border-left-color: #28a745; }
        .status-active { color: #28a745; }
        .status-stopped { color: #dc3545; }
        </style>
        """, unsafe_allow_html=True)

    def run(self):
        """Main application entry point"""
        st.title("🔒 Real-time Git Repository Security Monitor")
        st.markdown("**Advanced security monitoring with comprehensive repository dashboard**")
        
        # Check if we have an active session
        if 'current_session' not in st.session_state:
            self.render_setup_view()
        else:
            self.render_monitoring_view()

    def render_setup_view(self):
        """Render the initial setup interface"""
        st.markdown("### 🚀 Start Repository Monitoring")
        
        # Emergency cleanup button (in case there are orphaned sessions)
        emergency_col1, emergency_col2, emergency_col3 = st.columns([1, 1, 1])
        
        with emergency_col2:
            if st.button("🔄 **Refresh Interface**", type="secondary", help="Refresh the monitoring interface"):
                st.session_state.force_refresh = True
                st.rerun()
        
        with emergency_col3:
            if st.button("🛑 **Stop All Monitoring**", type="secondary", help="Emergency stop for all active sessions"):
                with st.spinner("Stopping all monitoring sessions..."):
                    result = self.monitor.stop_all_monitoring()
                    if 'error' not in result:
                        st.success(f"✅ {result['message']}")
                    else:
                        st.error(f"❌ {result['error']}")
                st.rerun()
        
        # Show existing sessions
        available_sessions = self.monitor.list_available_sessions()
        
        if available_sessions:
            st.markdown("#### 📋 Available Sessions")
            st.info("Select a session to view or resume monitoring")
            
            # Create dropdown options
            session_options = []
            session_mapping = {}
            
            for session in available_sessions:
                # Extract repository name
                repo_name = session['repo_url'].split('/')[-1].replace('.git', '')
                
                # Format creation time
                created_time = session['created_at']
                try:
                    created_dt = datetime.fromisoformat(created_time)
                    created_str = created_dt.strftime("%m/%d %H:%M")
                except:
                    created_str = "Unknown"
                
                # Status indicator and summary
                status_icon = "🟢" if session['is_active'] else "⚪"
                status_text = "Active" if session['is_active'] else "Inactive"
                
                # Create dropdown label
                option_label = f"{status_icon} {repo_name} - {created_str} ({session['commits_processed']:,} commits, {session['total_alerts']} alerts)"
                
                session_options.append(option_label)
                session_mapping[option_label] = session
            
            # Dropdown selector
            selected_option = st.selectbox(
                "Choose a session:",
                options=["Select a session..."] + session_options,
                index=0,
                help="Select a session to view details and take actions"
            )
            
            # Show details and actions for selected session
            if selected_option != "Select a session...":
                selected_session = session_mapping[selected_option]
                
                # Session details
                with st.container():
                    st.markdown("---")
                    
                    # Session info in columns
                    info_col1, info_col2, info_col3 = st.columns([2, 1, 1])
                    
                    with info_col1:
                        st.markdown(f"**🔗 Repository:** {selected_session['repo_url']}")
                        repo_name = selected_session['repo_url'].split('/')[-1].replace('.git', '')
                        
                        # Format creation time
                        created_time = selected_session['created_at']
                        try:
                            created_dt = datetime.fromisoformat(created_time)
                            created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            created_str = created_time
                        st.markdown(f"**📅 Created:** {created_str}")
                    
                    with info_col2:
                        status = "🟢 Active" if selected_session['is_active'] else "⚪ Inactive"
                        st.markdown(f"**Status:** {status}")
                        st.markdown(f"**📊 Commits:** {selected_session['commits_processed']:,}")
                    
                    with info_col3:
                        st.markdown(f"**🚨 Alerts:** {selected_session['total_alerts']}")
                        if selected_session['is_active']:
                            st.success("**Currently Monitoring**")
                        else:
                            st.info("**Ready to Resume**")
                
                # Action buttons
                st.markdown("---")
                action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
                
                with action_col1:
                    if selected_session['is_active']:
                        if st.button(f"📊 **View Dashboard**", type="primary", key="view_selected"):
                            st.session_state.current_session = selected_session['session_name']
                            st.success(f"✅ Opening dashboard for {repo_name}")
                            st.rerun()
                    else:
                        if st.button(f"🚀 **Resume Monitoring**", type="primary", key="resume_selected"):
                            with st.spinner(f"Resuming monitoring for {repo_name}..."):
                                result = self.monitor.resume_session(selected_session['session_name'])
                            
                            if 'error' in result:
                                st.error(f"❌ {result['error']}")
                            else:
                                st.session_state.current_session = selected_session['session_name']
                                st.success(f"✅ Successfully resumed monitoring for {repo_name}")
                                st.rerun()
                
                with action_col2:
                    # Check if delete method exists and show appropriate button
                    if hasattr(self.monitor, 'delete_session'):
                        if st.button(f"🗑️ **Delete Session**", type="secondary", key="delete_selected"):
                            if st.session_state.get('confirm_delete', False):
                                with st.spinner(f"Deleting session for {repo_name}..."):
                                    result = self.monitor.delete_session(selected_session['session_name'])
                                
                                if 'error' in result:
                                    st.error(f"❌ {result['error']}")
                                else:
                                    st.success(f"✅ Session deleted: {repo_name}")
                                
                                if 'confirm_delete' in st.session_state:
                                    del st.session_state.confirm_delete
                                st.rerun()
                            else:
                                st.session_state.confirm_delete = True
                                st.warning("⚠️ Click again to confirm deletion")
                    else:
                        st.error("🔄 Delete function not available - please refresh interface")
                        st.caption("Click 'Refresh Interface' button above")
                
                with action_col3:
                    st.caption("💡 **Tip:** Active sessions show 🟢 and can be viewed immediately. Inactive sessions show ⚪ and need to be resumed first.")
            
            st.divider()
        
        st.markdown("#### 🆕 Start New Monitoring Session")
        st.info("Enter a Git repository URL to begin real-time security monitoring")
        
        with st.form("setup_form"):
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
        """Render comprehensive repository dashboard"""
        
        session_name = st.session_state.current_session
        session_info = self.monitor.get_session_status(session_name)
        
        if 'error' in session_info:
            st.error(f"❌ Session error: {session_info['error']}")
            if st.button("🔄 Restart"):
                del st.session_state.current_session
                st.rerun()
            return
        
        # Navigation and Control Header
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        
        with nav_col1:
            if st.button("🏠 **Back to Main**", type="secondary"):
                del st.session_state.current_session
                st.rerun()
        
        with nav_col2:
            st.markdown(f"## � Repository Dashboard")
            
        with nav_col3:
            if st.button("🗑️ **Stop & Close**", type="secondary"):
                if st.session_state.get('confirm_stop', False):
                    with st.spinner("Stopping monitoring and cleaning up..."):
                        result = self.monitor.close_session(session_name)
                        if 'error' not in result:
                            st.success("✅ Monitoring stopped and session closed")
                        else:
                            st.error(f"❌ {result['error']}")
                    del st.session_state.current_session
                    if 'confirm_stop' in st.session_state:
                        del st.session_state.confirm_stop
                    st.rerun()
                else:
                    st.session_state.confirm_stop = True
                    st.warning("⚠️ Click again to confirm stop & close")
        
        # Debug info (can be removed later)
        with st.expander("🔧 Debug Info", expanded=False):
            st.json(session_info)
            st.markdown(f"**Session Active:** {session_info.get('monitoring_active', False)}")
            st.markdown(f"**Thread Active:** {session_info.get('thread_active', False)}")
            st.markdown(f"**Status:** {session_info.get('status', 'unknown')}")
            st.markdown(f"**Historical Commits:** {session_info.get('historical_commits', 0)}")
            st.markdown(f"**Contributors Analyzed:** {session_info.get('contributors_analyzed', 0)}")
            st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Repository Info Panel
        with st.container():
            st.markdown("### 📦 Repository Information")
            repo_col1, repo_col2 = st.columns([2, 1])
            
            with repo_col1:
                st.markdown(f"**🔗 Repository:** {session_info['repo_url']}")
                
                # Extract repo name from URL for better display
                try:
                    repo_parts = session_info['repo_url'].split('/')
                    if len(repo_parts) >= 2:
                        owner = repo_parts[-2]
                        repo_name = repo_parts[-1].replace('.git', '')
                        st.markdown(f"**👤 Owner:** {owner}")
                        st.markdown(f"**📁 Repository:** {repo_name}")
                except:
                    pass
                    
                created_at = session_info.get('created_at', 'Unknown')
                if created_at != 'Unknown':
                    try:
                        created_dt = datetime.fromisoformat(created_at)
                        created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S")
                        st.markdown(f"**📅 Monitoring Started:** {created_str}")
                    except:
                        st.markdown(f"**📅 Monitoring Started:** {created_at}")
            
            with repo_col2:
                # Status indicator - check both session active and thread active
                is_active = session_info.get('monitoring_active', False) or session_info.get('thread_active', False)
                
                if is_active:
                    st.success("🟢 **ACTIVE MONITORING**")
                    st.caption("✅ Security scanning enabled")
                else:
                    st.error("🔴 **MONITORING STOPPED**")
                    st.caption("⚠️ No security scanning")
        
        st.divider()
        
        # Key Metrics Dashboard
        st.markdown("### 📈 Repository Analytics")
        
        # Primary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Show actual Git commits, not audit entries
            git_commits = session_info.get('git_commits', session_info['historical_commits'])
            st.metric(
                label="📦 Git Commits",
                value=f"{git_commits:,}",
                help="Actual Git repository commits (excluding merge duplicates)"
            )
        
        with col2:
            st.metric(
                label="👥 Contributors",
                value=f"{session_info['contributors_analyzed']:,}",
                help="Unique contributors identified from Git shortlog analysis"
            )
        
        with col3:
            st.metric(
                label="🚨 Security Alerts",
                value=f"{session_info['total_alerts']:,}",
                help="Total security anomalies detected"
            )
        
        with col4:
            st.metric(
                label="🔄 New Commits",
                value=f"{session_info['new_commits_processed']:,}",
                help="New commits processed since monitoring started"
            )
        
        # Secondary metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        uptime = session_info.get('uptime', '0 minutes')
        last_check = session_info.get('last_check_time', 'Never')
        
        with col5:
            st.metric(
                label="⏰ Monitoring Uptime",
                value=uptime,
                help="How long this monitoring session has been active"
            )
        
        with col6:
            st.metric(
                label="🔍 Last Check",
                value=last_check,
                help="When the system last checked for new commits"
            )
        
        with col7:
            # Calculate approximate commits per contributor
            commits_per_contributor = (
                session_info['historical_commits'] // max(session_info['contributors_analyzed'], 1)
                if session_info['contributors_analyzed'] > 0 else 0
            )
            st.metric(
                label="📊 Avg Commits/Contributor",
                value=f"{commits_per_contributor:,}",
                help="Average number of commits per contributor"
            )
        
        with col8:
            # Repository activity indicator
            if session_info['new_commits_processed'] > 0:
                activity_status = "🔥 High Activity"
            elif last_check != 'Never':
                activity_status = "✅ Active"
            else:
                activity_status = "⏸️ Initializing"
            
            st.metric(
                label="🎯 Repository Status",
                value=activity_status,
                help="Current activity level of the repository"
            )
        
        st.divider()
            
        with col2:
            last_check = session_info.get('last_check_time', 'Never')
            if last_check != 'Never':
                st.metric("🕐 Last Check", last_check)
                st.caption("Most recent scan")
            else:
                st.metric("🔄 Status", "Initializing")
                st.caption("Setting up monitoring")
            
        with col3:
            # Calculate alert rate
            new_commits = session_info['new_commits_processed']
            alerts = session_info['total_alerts']
            total_processed = max(1, session_info.get('historical_commits', 1) + new_commits)
            alert_rate = (alerts / total_processed) * 100
            st.metric("📊 Alert Rate", f"{alert_rate:.1f}%")
            st.caption("Percentage of flagged commits")
            
        with col4:
            # Security score based on alert rate
            if alert_rate < 2:
                security_score = "🟢 Excellent"
            elif alert_rate < 5:
                security_score = "🟡 Good"
            elif alert_rate < 10:
                security_score = "🟠 Moderate"
            else:
                security_score = "🔴 High Risk"
            
            st.metric("🛡️ Security Score", security_score)
            st.caption("Overall security assessment")
        
        st.divider()
        
        # Real-time monitoring status
        st.markdown("### ⚡ Real-Time Status")
        
        # Check if monitoring is actually active
        is_monitoring_active = session_info.get('monitoring_active', False) or session_info.get('thread_active', False)
        last_check = session_info.get('last_check_time', 'Never')
        
        if is_monitoring_active:
            if last_check != 'Never':
                st.success(f"🔍 **MONITORING ACTIVE** - Last scan completed at {last_check}")
                st.info("🔄 Automatically checking for new commits every 30 seconds")
            else:
                st.info("🔄 **INITIALIZING MONITORING** - Setting up real-time scanning...")
        else:
            st.error("⏸️ **MONITORING PAUSED** - No automatic security scanning")
        
        st.divider()
        
        # Security alerts section
        st.markdown("### 🚨 Security Analysis")
        
        recent_alerts = self.monitor.get_recent_alerts(session_name, limit=10)
        
        if recent_alerts:
            # Alert summary
            st.warning(f"⚠️ **{len(recent_alerts)} Security Events Detected**")
            
            # Alert breakdown by risk level
            risk_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
            for alert in recent_alerts:
                risk = alert.get('risk_level', 'Medium')
                if risk in risk_counts:
                    risk_counts[risk] += 1
            
            # Show risk distribution
            risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
            
            with risk_col1:
                if risk_counts['Critical'] > 0:
                    st.error(f"🔴 **{risk_counts['Critical']} Critical**")
                else:
                    st.success("🔴 **0 Critical**")
                    
            with risk_col2:
                if risk_counts['High'] > 0:
                    st.warning(f"🟠 **{risk_counts['High']} High**")
                else:
                    st.success("🟠 **0 High**")
                    
            with risk_col3:
                if risk_counts['Medium'] > 0:
                    st.info(f"🟡 **{risk_counts['Medium']} Medium**")
                else:
                    st.success("🟡 **0 Medium**")
                    
            with risk_col4:
                st.info(f"🟢 **{risk_counts['Low']} Low**")
            
            # Detailed alerts list
            st.markdown("#### 🔍 Recent Security Events")
            
            for i, alert in enumerate(reversed(recent_alerts[-5:])):  # Show last 5 alerts
                alert_num = len(recent_alerts) - i
                risk = alert.get('risk_level', 'Medium')
                risk_icons = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                risk_icon = risk_icons.get(risk, '🔵')
                
                with st.expander(f"{risk_icon} **Alert #{alert_num}:** {alert.get('anomaly_type', 'Security Event')} - {risk} Risk", expanded=i==0):
                    alert_col1, alert_col2 = st.columns([3, 1])
                    
                    with alert_col1:
                        st.markdown(f"**📝 Commit Hash:** `{alert.get('commit_hash', 'Unknown')}`")
                        st.markdown(f"**👤 Author:** {alert.get('author', 'Unknown')}")
                        
                        commit_msg = alert.get('message', 'No commit message')
                        if len(commit_msg) > 80:
                            commit_msg = commit_msg[:80] + "..."
                        st.markdown(f"**💬 Message:** {commit_msg}")
                        
                        explanation = alert.get('explanation', 'No detailed analysis available')
                        st.markdown(f"**🔍 Analysis:** {explanation}")
                        
                        # Show anomaly score
                        score = alert.get('anomaly_score', 0)
                        if score > 0:
                            st.markdown(f"**📊 Anomaly Score:** {score:.3f}")
                    
                    with alert_col2:
                        # Timestamp
                        timestamp = alert.get('timestamp', 'Unknown')
                        if timestamp != 'Unknown':
                            try:
                                alert_dt = datetime.fromisoformat(timestamp)
                                date_str = alert_dt.strftime("%Y-%m-%d")
                                time_str = alert_dt.strftime("%H:%M:%S")
                                st.markdown(f"**📅 Date:** {date_str}")
                                st.markdown(f"**🕐 Time:** {time_str}")
                            except:
                                st.markdown(f"**🕐 Time:** {timestamp}")
                        
                        # Risk indicator
                        st.markdown(f"**⚠️ Risk Level:**")
                        st.markdown(f"### {risk_icon} {risk}")
            
            if len(recent_alerts) > 5:
                st.info(f"📋 Showing 5 most recent alerts out of {len(recent_alerts)} total security events")
                
        else:
            # All clear status
            st.success("✅ **All Clear - No Security Anomalies Detected**")
            
            col_clear1, col_clear2 = st.columns(2)
            
            with col_clear1:
                st.info("🔍 **Active Monitoring Enabled**")
                st.markdown("""
                The system is continuously analyzing:
                - Commit timing patterns
                - Code change volumes
                - User behavior patterns
                """)
            
            with col_clear2:
                st.info("🛡️ **Security Status: Excellent**")
                st.markdown("""
                No suspicious activity detected:
                - All commits follow normal patterns
                - No unusual timing detected
                - User behavior is consistent
                """)
        
        st.divider()
        
        # Control Panel
        st.markdown("### 🎛️ Control Panel")
        
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            if st.button("🔄 **Force Refresh**", type="primary", help="Force update of all session data"):
                # Clear any streamlit caching
                st.cache_data.clear()
                st.rerun()
        
        with control_col2:
            if st.button("📊 **Export Report**"):
                st.info("📄 Detailed report export coming soon!")
        
        with control_col3:
            auto_refresh = st.checkbox("⚡ **Auto-refresh**", value=True, help="Automatically refresh every 30 seconds")
        
        # Auto refresh with countdown
        is_monitoring_active = session_info.get('monitoring_active', False) or session_info.get('thread_active', False)
        if auto_refresh and is_monitoring_active:
            with st.empty():
                for remaining in range(30, 0, -1):
                    st.caption(f"🔄 Next automatic refresh in {remaining} seconds...")
                    time.sleep(1)
            st.rerun()

if __name__ == "__main__":
    app = ComprehensiveRealtimeMonitoringApp()
    app.run()
