"""
Git Repository Security Analysis - Web-based Main Launcher
Unified web interface for selecting analysis modes
"""

import streamlit as st
import subprocess
import os
import shutil
import sys
from datetime import datetime
import time
import threading

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cleanup_utils import GitAnalysisCleanup

class WebMainLauncher:
    """Web-based main launcher for Git security analysis tools"""
    
    def __init__(self):
        self.project_path = os.path.dirname(os.path.abspath(__file__ + '/..'))
        self.python_exe = self._get_python_executable()
        self.cleanup = GitAnalysisCleanup(self.project_path)
        self.setup_page()
        self.cleanup.auto_cleanup_on_startup()
    
    def _get_python_executable(self):
        """Get the correct Python executable path"""
        venv_python = os.path.join(self.project_path, '.venv', 'Scripts', 'python.exe')
        if os.path.exists(venv_python):
            return venv_python
        return 'python'
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="🔍 Git Security Analysis Suite",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def cleanup_temp_repos(self):
        """Clean up temporary repository clones"""
        return self.cleanup.cleanup_temp_repos(show_message=False)
    
    def show_cleanup_status(self):
        """Show cleanup status messages"""
        self.cleanup.show_cleanup_status_streamlit()
    
    def launch_app_in_new_port(self, app_script, port_start=8505):
        """Launch Streamlit app in new port to avoid conflicts"""
        port = port_start
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                # Check if port is available
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('localhost', port))
                    if result != 0:  # Port is available
                        break
                port += 1
            except:
                port += 1
        
        # Launch app
        cmd = [
            self.python_exe, '-m', 'streamlit', 'run', 
            app_script, f'--server.port={port}', '--server.headless=false'
        ]
        
        def launch():
            subprocess.run(cmd, cwd=self.project_path)
        
        # Start in background thread
        thread = threading.Thread(target=launch, daemon=True)
        thread.start()
        
        return port
    
    def render_main_interface(self):
        """Render the main selection interface"""
        
        # Header
        st.markdown('<div class="main-header">🔍 Git Repository Security Analysis Suite</div>', unsafe_allow_html=True)
        
        # Subtitle and date
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h3>🎯 Choose Your Analysis Mode</h3>
                <p style="color: #666;">📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show cleanup status
        self.show_cleanup_status()
        
        # Main Analysis Options
        st.markdown("### 🎯 Choose Your Analysis Mode")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⚡ Real-time Monitoring**")
            if st.button("🌐 Launch Web Dashboard", key="realtime_web", use_container_width=True):
                try:
                    port = self.launch_app_in_new_port('ui/realtime_app.py', 8504)
                    st.success(f"✅ Started on port {port}")
                    st.info("� Continuous monitoring with live alerts")
                except Exception as e:
                    st.error(f"❌ Launch error: {e}")
            
            if st.button("💻 Launch CLI Monitor", key="realtime_cli", use_container_width=True):
                st.code("run_realtime_demo.bat", language="bash")
                st.info("💻 Terminal-based real-time monitoring")
        
        with col2:
            st.markdown("**📊 Standard Analysis**")
            if st.button("🖥️ Launch Web Analyzer", key="standard_web", use_container_width=True):
                try:
                    port = self.launch_app_in_new_port('ui/git_app.py', 8502)
                    st.success(f"✅ Started on port {port}")
                    st.info("📈 One-time repository analysis")
                except Exception as e:
                    st.error(f"❌ Launch error: {e}")
            
            if st.button("⌨️ Launch CLI Analyzer", key="standard_cli", use_container_width=True):
                st.code("run_git_demo.bat", language="bash")
                st.info("💻 Quick terminal analysis")
        
        with col3:
            st.markdown("**🛠️ Utilities**")
            if st.button("🧪 Run System Tests", key="test_system_util", use_container_width=True):
                with st.spinner("Testing..."):
                    try:
                        result = subprocess.run([
                            self.python_exe, 'test_realtime.py'
                        ], cwd=self.project_path, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            st.success("✅ All tests passed!")
                        else:
                            st.error("❌ Some tests failed")
                            st.text(result.stderr)
                    except Exception as e:
                        st.error(f"❌ Test error: {e}")
            
            if st.button("🧹 Clean Temp Files", key="cleanup", use_container_width=True):
                results = self.cleanup.cleanup_all(show_message=False)
                cleaned_count = sum(1 for cleaned, _ in results.values() if cleaned)
                if cleaned_count > 0:
                    st.success(f"✅ Cleaned {cleaned_count} categories")
                else:
                    st.info("ℹ️ Nothing to clean")
        
        st.divider()
        
        # System Status
        st.markdown("### 📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if os.path.exists(self.python_exe):
                st.success("✅ Python")
            else:
                st.error("❌ Python")
        
        with col2:
            try:
                import streamlit, pandas, numpy, sklearn
                st.success("✅ Dependencies")
            except ImportError:
                st.error("❌ Dependencies")
        
        with col3:
            git_available = subprocess.run(['git', '--version'], capture_output=True).returncode == 0
            if git_available:
                st.success("✅ Git")
            else:
                st.error("❌ Git")
        
        with col4:
            status = self.cleanup.get_status()
            temp_size = status['total_size_mb']
            
            if temp_size < 0.1:
                st.success("✅ Clean")
            else:
                st.warning(f"⚠️ {temp_size:.1f}MB")
        
        # Quick Info
        st.info("💡 **Tip:** Each mode opens in a new browser tab. Repository clones are automatically cleaned on page refresh.")
        
        # Footer
        st.markdown("---")
        st.markdown("**🔍 Git Security Analysis Suite** | Built with Streamlit & ML")
    
    def run(self):
        """Main application entry point"""
        self.render_main_interface()

def main():
    """Run the web-based main launcher"""
    launcher = WebMainLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
