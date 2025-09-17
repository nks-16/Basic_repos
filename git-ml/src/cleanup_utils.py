"""
Cleanup utilities for Git Security Analysis Suite
Handles automatic cleanup of temporary repositories and session data
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

class GitAnalysisCleanup:
    """Centralized cleanup utilities for the Git analysis system"""
    
    def __init__(self, project_root=None):
        """Initialize cleanup utility"""
        if project_root is None:
            # Try to find project root from current file location
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent
            # If we're in src/, go up one level
            if self.project_root.name == 'src':
                self.project_root = self.project_root.parent
        else:
            self.project_root = Path(project_root).resolve()
        
        self.temp_repos_path = self.project_root / 'temp_repos'
        self.sessions_path = self.project_root / 'sessions'
        self.logs_path = self.project_root / 'logs'
        
    def cleanup_temp_repos(self, show_message=True):
        """Clean up temporary repository clones"""
        cleaned = False
        error_msg = None
        
        if self.temp_repos_path.exists():
            try:
                # Windows-specific: Remove read-only attributes first
                if os.name == 'nt':
                    try:
                        # Use attrib command to remove read-only attributes
                        subprocess.run(['attrib', '-R', f'{self.temp_repos_path}\\*.*', '/S'], 
                                     capture_output=True, shell=True, timeout=30)
                    except Exception:
                        pass  # Ignore attrib errors, try normal cleanup
                
                # Try normal cleanup first
                shutil.rmtree(self.temp_repos_path)
                cleaned = True
                if show_message:
                    print(f"✅ Cleaned temporary repositories: {self.temp_repos_path}")
                    
            except PermissionError as pe:
                # Handle Windows permission errors
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.run(['rmdir', '/S', '/Q', str(self.temp_repos_path)], 
                                     shell=True, capture_output=True, timeout=30)
                        cleaned = True
                        if show_message:
                            print(f"✅ Cleaned temporary repositories (with admin): {self.temp_repos_path}")
                    else:
                        # Linux/Mac - try with force
                        subprocess.run(['rm', '-rf', str(self.temp_repos_path)], 
                                     capture_output=True, timeout=30)
                        cleaned = True
                        if show_message:
                            print(f"✅ Cleaned temporary repositories (forced): {self.temp_repos_path}")
                except Exception as fallback_error:
                    error_msg = f"❌ Error cleaning temp repos (permission denied): {pe}. Fallback failed: {fallback_error}"
                    if show_message:
                        print(error_msg)
                        print("💡 Tip: Try running as administrator or manually delete the temp_repos folder")
            except Exception as e:
                error_msg = f"❌ Error cleaning temp repos: {e}"
                if show_message:
                    print(error_msg)
        
        return cleaned, error_msg
    
    def cleanup_sessions(self, show_message=True):
        """Clean up session data"""
        cleaned = False
        error_msg = None
        
        if self.sessions_path.exists():
            try:
                shutil.rmtree(self.sessions_path)
                cleaned = True
                if show_message:
                    print(f"✅ Cleaned session data: {self.sessions_path}")
            except Exception as e:
                error_msg = f"❌ Error cleaning sessions: {e}"
                if show_message:
                    print(error_msg)
        
        return cleaned, error_msg
    
    def cleanup_logs(self, show_message=True):
        """Clean up log files"""
        cleaned = False
        error_msg = None
        
        if self.logs_path.exists():
            try:
                # Remove all log files but keep the directory
                for log_file in self.logs_path.glob("*.log"):
                    log_file.unlink()
                cleaned = True
                if show_message:
                    print(f"✅ Cleaned log files: {self.logs_path}")
            except Exception as e:
                error_msg = f"❌ Error cleaning logs: {e}"
                if show_message:
                    print(error_msg)
        
        return cleaned, error_msg
    
    def cleanup_all(self, show_message=True):
        """Clean up all temporary files and data"""
        results = {
            'temp_repos': self.cleanup_temp_repos(show_message),
            'sessions': self.cleanup_sessions(show_message),
            'logs': self.cleanup_logs(show_message)
        }
        
        total_cleaned = sum(1 for cleaned, _ in results.values() if cleaned)
        errors = [error for _, error in results.values() if error]
        
        if show_message:
            if total_cleaned > 0:
                print(f"🧹 Cleanup completed: {total_cleaned} categories cleaned")
            if errors:
                print(f"⚠️ {len(errors)} cleanup errors occurred")
        
        return results
    
    def get_temp_size(self):
        """Get size of temporary files in MB"""
        total_size = 0
        
        for path in [self.temp_repos_path, self.sessions_path, self.logs_path]:
            if path.exists():
                for item in path.rglob("*"):
                    if item.is_file():
                        try:
                            total_size += item.stat().st_size
                        except (OSError, PermissionError):
                            pass  # Skip files we can't access
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_status(self):
        """Get status of temporary files"""
        status = {
            'temp_repos_exists': self.temp_repos_path.exists(),
            'sessions_exists': self.sessions_path.exists(),
            'logs_exists': self.logs_path.exists(),
            'total_size_mb': self.get_temp_size()
        }
        return status
    
    def auto_cleanup_on_startup(self):
        """Automatically clean temp repos on startup (for use in Streamlit apps)"""
        try:
            # Import streamlit only if available
            import streamlit as st
            
            # Clean up temp repos
            cleaned, error = self.cleanup_temp_repos(show_message=False)
            
            if cleaned:
                if 'cleanup_message' not in st.session_state:
                    st.session_state.cleanup_message = "🧹 Cleaned up previous repository clones"
            elif error:
                if 'cleanup_error' not in st.session_state:
                    st.session_state.cleanup_error = error
                    
        except ImportError:
            # Not in Streamlit context, just do regular cleanup
            self.cleanup_temp_repos(show_message=True)
    
    def show_cleanup_status_streamlit(self):
        """Show cleanup status messages in Streamlit (call this in your main app)"""
        try:
            import streamlit as st
            
            if 'cleanup_message' in st.session_state:
                st.success(st.session_state.cleanup_message)
                del st.session_state.cleanup_message
            
            if 'cleanup_error' in st.session_state:
                st.warning(st.session_state.cleanup_error)
                del st.session_state.cleanup_error
                
        except ImportError:
            pass  # Not in Streamlit context

# Global cleanup instance
cleanup = GitAnalysisCleanup()

# Convenience functions
def cleanup_temp_repos():
    """Quick function to clean temp repos"""
    return cleanup.cleanup_temp_repos()

def cleanup_all():
    """Quick function to clean all temporary data"""
    return cleanup.cleanup_all()

def get_temp_status():
    """Quick function to get temporary file status"""
    return cleanup.get_status()

if __name__ == "__main__":
    """Command line interface for cleanup"""
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        
        if action == 'temp':
            cleanup.cleanup_temp_repos()
        elif action == 'sessions':
            cleanup.cleanup_sessions()
        elif action == 'logs':
            cleanup.cleanup_logs()
        elif action == 'all':
            cleanup.cleanup_all()
        elif action == 'status':
            status = cleanup.get_status()
            print(f"📊 Cleanup Status:")
            print(f"  Temp repos: {'✅' if not status['temp_repos_exists'] else '📁'}")
            print(f"  Sessions: {'✅' if not status['sessions_exists'] else '📁'}")
            print(f"  Logs: {'✅' if not status['logs_exists'] else '📁'}")
            print(f"  Total size: {status['total_size_mb']:.1f} MB")
        else:
            print("Usage: python cleanup_utils.py [temp|sessions|logs|all|status]")
    else:
        # Interactive mode
        print("🧹 Git Analysis Cleanup Utility")
        print("Available actions:")
        print("1. Clean temp repos (temp)")
        print("2. Clean sessions (sessions)")
        print("3. Clean logs (logs)")
        print("4. Clean all (all)")
        print("5. Show status (status)")
        
        choice = input("\nEnter action (or press Enter for 'all'): ").strip().lower()
        if not choice:
            choice = 'all'
            
        if choice in ['temp', 'sessions', 'logs', 'all', 'status']:
            if choice == 'temp':
                cleanup.cleanup_temp_repos()
            elif choice == 'sessions':
                cleanup.cleanup_sessions()
            elif choice == 'logs':
                cleanup.cleanup_logs()
            elif choice == 'all':
                cleanup.cleanup_all()
            elif choice == 'status':
                status = cleanup.get_status()
                print(f"\n📊 Cleanup Status:")
                print(f"  Temp repos: {'✅ Clean' if not status['temp_repos_exists'] else '📁 Has files'}")
                print(f"  Sessions: {'✅ Clean' if not status['sessions_exists'] else '📁 Has files'}")
                print(f"  Logs: {'✅ Clean' if not status['logs_exists'] else '📁 Has files'}")
                print(f"  Total size: {status['total_size_mb']:.1f} MB")
        else:
            print("Invalid choice.")
