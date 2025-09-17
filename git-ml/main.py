#!/usr/bin/env python3
"""
Git Repository Security Analysis - Main Menu
Unified launcher for all analysis modes and interfaces
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class GitSecurityLauncher:
    """Main launcher for Git security analysis tools"""
    
    def __init__(self):
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.python_exe = self._get_python_executable()
    
    def _get_python_executable(self):
        """Get the correct Python executable path"""
        venv_python = os.path.join(self.project_path, '.venv', 'Scripts', 'python.exe')
        if os.path.exists(venv_python):
            return venv_python
        return 'python'
    
    def print_header(self):
        """Print application header"""
        print("=" * 80)
        print("🔍 GIT REPOSITORY SECURITY ANALYSIS SUITE")
        print("=" * 80)
        print("🎯 Comprehensive security monitoring and anomaly detection for Git repositories")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def print_menu(self):
        """Print main menu options"""
        print("📋 SELECT ANALYSIS MODE:")
        print()
        
        print("🌟 REAL-TIME MONITORING (Recommended)")
        print("  1. 🌐 Real-time Web Interface    - Interactive monitoring dashboard")
        print("  2. 💻 Real-time Command Line     - Terminal-based live monitoring")
        print()
        
        print("📊 STANDARD ANALYSIS")
        print("  3. 🖥️  Standard Web Interface    - One-time repository analysis")
        print("  4. ⌨️  Standard Command Line     - Terminal-based analysis")
        print()
        
        print("🛠️  UTILITIES & TESTING")
        print("  5. 🧪 Test System Components    - Verify installation and functionality")
        print("  6. 📚 View Documentation        - Open README.md")
        print("  7. 📦 Install/Update Dependencies - Setup requirements")
        print()
        
        print("  0. 🚪 Exit")
        print()
        print("=" * 80)
    
    def launch_realtime_web(self):
        """Launch real-time web interface"""
        print("🚀 Starting Real-time Web Interface...")
        print("🌐 Opening interactive monitoring dashboard")
        print("⚡ Monitor Git repositories for security anomalies in real-time")
        print()
        
        try:
            cmd = [
                self.python_exe, '-m', 'streamlit', 'run', 
                'ui/realtime_app.py', '--server.port=8504', '--server.headless=false'
            ]
            subprocess.run(cmd, cwd=self.project_path)
        except KeyboardInterrupt:
            print("\n👋 Real-time monitoring stopped")
        except Exception as e:
            print(f"❌ Error launching real-time web interface: {e}")
            self._suggest_troubleshooting()
    
    def launch_realtime_cli(self):
        """Launch real-time command line interface"""
        print("🚀 Starting Real-time Command Line Interface...")
        print("💻 Interactive terminal-based monitoring")
        print("📊 Monitor repositories with live alerts")
        print()
        
        try:
            cmd = [self.python_exe, 'realtime_demo.py']
            subprocess.run(cmd, cwd=self.project_path)
        except KeyboardInterrupt:
            print("\n👋 Real-time CLI stopped")
        except Exception as e:
            print(f"❌ Error launching real-time CLI: {e}")
            self._suggest_troubleshooting()
    
    def launch_standard_web(self):
        """Launch standard web interface"""
        print("🚀 Starting Standard Web Interface...")
        print("🖥️  One-time repository analysis dashboard")
        print("📈 Analyze Git repositories for security patterns")
        print()
        
        try:
            cmd = [
                self.python_exe, '-m', 'streamlit', 'run', 
                'ui/git_app.py', '--server.port=8502', '--server.headless=false'
            ]
            subprocess.run(cmd, cwd=self.project_path)
        except KeyboardInterrupt:
            print("\n👋 Standard web interface stopped")
        except Exception as e:
            print(f"❌ Error launching standard web interface: {e}")
            self._suggest_troubleshooting()
    
    def launch_standard_cli(self):
        """Launch standard command line interface"""
        print("🚀 Starting Standard Command Line Interface...")
        print("⌨️  Terminal-based repository analysis")
        print("🔍 Analyze Git repositories from command line")
        print()
        
        try:
            cmd = [self.python_exe, 'git_demo.py']
            subprocess.run(cmd, cwd=self.project_path)
        except KeyboardInterrupt:
            print("\n👋 Standard CLI stopped")
        except Exception as e:
            print(f"❌ Error launching standard CLI: {e}")
            self._suggest_troubleshooting()
    
    def run_tests(self):
        """Run system component tests"""
        print("🧪 Testing System Components...")
        print("🔍 Verifying installation and functionality")
        print()
        
        try:
            cmd = [self.python_exe, 'test_realtime.py']
            subprocess.run(cmd, cwd=self.project_path)
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            self._suggest_troubleshooting()
        
        print("\n📊 Test completed!")
        input("Press Enter to continue...")
    
    def view_documentation(self):
        """Open README documentation"""
        print("📚 Opening Documentation...")
        
        readme_path = os.path.join(self.project_path, 'README.md')
        
        if os.path.exists(readme_path):
            try:
                # Try to open with default editor
                if sys.platform.startswith('win'):
                    os.startfile(readme_path)
                elif sys.platform.startswith('darwin'):
                    subprocess.run(['open', readme_path])
                else:
                    subprocess.run(['xdg-open', readme_path])
                
                print("✅ Documentation opened in default editor")
            except Exception as e:
                print(f"⚠️  Could not open automatically: {e}")
                print(f"📄 Please manually open: {readme_path}")
        else:
            print("❌ README.md not found")
        
        input("Press Enter to continue...")
    
    def install_dependencies(self):
        """Install or update dependencies"""
        print("📦 Installing/Updating Dependencies...")
        print("⏳ This may take a few minutes...")
        print()
        
        requirements = [
            'streamlit', 'pandas', 'numpy', 'scikit-learn', 
            'plotly', 'matplotlib', 'seaborn'
        ]
        
        try:
            for package in requirements:
                print(f"📥 Installing {package}...")
                cmd = [self.python_exe, '-m', 'pip', 'install', package]
                result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"⚠️  Warning installing {package}: {result.stderr}")
            
            print("\n🎉 All dependencies processed!")
            
        except Exception as e:
            print(f"❌ Error installing dependencies: {e}")
        
        input("Press Enter to continue...")
    
    def _suggest_troubleshooting(self):
        """Suggest troubleshooting steps"""
        print("\n🛠️  TROUBLESHOOTING:")
        print("1. Try running option 7 to install/update dependencies")
        print("2. Check that Git is installed and accessible")
        print("3. Ensure you have internet access for repository cloning")
        print("4. Run option 5 to test system components")
        print()
        input("Press Enter to continue...")
    
    def run(self):
        """Main application loop"""
        while True:
            try:
                self.print_header()
                self.print_menu()
                
                choice = input("🎯 Select option (0-7): ").strip()
                print()
                
                if choice == '0':
                    print("👋 Thank you for using Git Repository Security Analysis!")
                    print("🔍 Stay secure and monitor your repositories regularly")
                    break
                
                elif choice == '1':
                    self.launch_realtime_web()
                
                elif choice == '2':
                    self.launch_realtime_cli()
                
                elif choice == '3':
                    self.launch_standard_web()
                
                elif choice == '4':
                    self.launch_standard_cli()
                
                elif choice == '5':
                    self.run_tests()
                
                elif choice == '6':
                    self.view_documentation()
                
                elif choice == '7':
                    self.install_dependencies()
                
                else:
                    print("❌ Invalid option. Please select 0-7")
                    input("Press Enter to continue...")
                
                print()  # Add spacing between iterations
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    launcher = GitSecurityLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
