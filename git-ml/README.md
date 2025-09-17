
# Git Repository Security Analyzer

A comprehensive security analysis tool that analyzes real Git repositories for anomalous behavior patterns using machine learning. The system automatically learns optimal anomaly detection criteria from your repository's data patterns.

## 🎯 Overview

This tool analyzes actual Git repositories to detect security anomalies using:
- **🔍 Real Git Analysis**: Analyzes actual commit logs, contributor patterns, and code changes
- **🤖 ML-Driven Criteria Learning**: Automatically discovers optimal anomaly detection rules
- **📊 20+ Security Features**: Extracted from real Git commit data and contributor behavior
- **⚡ Smart Classification**: ML-powered anomaly categorization based on learned patterns
- **🌐 Interactive Interface**: Real-time analysis with detailed visualizations

## ✨ Key Features

### 🔍 **Git Repository Analysis**
- Clones and analyzes any public Git repository
- Extracts commit patterns, timing, and contributor behaviors
- Identifies suspicious activities like off-hours commits, large changes, sensitive file modifications
- Supports date range filtering for targeted analysis

### 🤖 **ML-Driven Anomaly Learning**
- **Automatic Criteria Discovery**: ML analyzes your repo to find optimal thresholds
- **Pattern-Based Detection**: Clusters similar anomalous behaviors
- **Adaptive Thresholds**: Learns repository-specific normal vs. anomalous patterns  
- **Smart Classification**: Categorizes anomalies by risk level and pattern type

### 🚨 **Intelligent Risk Assessment**
- **🔴 Critical**: Data exfiltration patterns, security breaches
- **🟡 High Risk**: Suspicious timing patterns, policy violations
- **🟠 Medium Risk**: Unusual but potentially explainable behavior  
- **🟢 Low Risk**: Minor deviations from normal patterns

### 📊 **Advanced Security Features**
- Commit frequency and timing analysis
- Large commit detection (potential data exfiltration)
- Off-hours and weekend activity monitoring
- Sensitive file modification tracking
- Rapid commit sequence detection (automation/bot detection)
- Repository access diversity analysis
- External domain activity monitoring

## 🚀 Quick Start

### 🌐 Web-based Main Launcher (Recommended)

The easiest way to access all tools is through the **Web-based Main Launcher**:

1. **Launch the main dashboard:**
   ```batch
   run_main_launcher.bat
   ```

2. **Open your browser to:** http://localhost:8501

3. **Choose your analysis mode from the web interface:**
   - **⚡ Real-time Monitoring**: Continuous monitoring with session management
   - **📊 Standard Analysis**: One-time repository analysis  
   - **🛠️ Utilities & Testing**: System verification and cleanup tools

**🎯 Key Benefits:**
- ✅ **No Terminal Commands**: All selections made through web interface
- ✅ **Automatic Cleanup**: Repository clones cleaned on every page refresh
- ✅ **Multi-mode Support**: Launch different tools without port conflicts
- ✅ **System Status**: Real-time environment and dependency monitoring
- ✅ **One-click Testing**: Built-in system verification and troubleshooting

### 📱 Alternative Launch Methods

**Option 1: Direct Batch Scripts**
**Option 1: Direct Batch Scripts**
```batch
# Real-time monitoring web interface
run_realtime_app.bat

# Standard analysis web interface  
run_git_app.bat

# Command-line demos
run_git_demo.bat
run_realtime_demo.bat
```

**Option 2: Manual Setup**
**Option 2: Manual Setup**
```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Start web applications
streamlit run ui/main_launcher.py --server.port 8501    # Main launcher
streamlit run ui/realtime_app.py --server.port 8504     # Real-time monitoring
streamlit run ui/git_app.py --server.port 8502          # Standard analysis

# 3. Or run CLI demos
python git_demo.py [repository_url]
python realtime_demo.py
```

**Option 3: Direct Access URLs**
- **🌐 Main Launcher**: http://localhost:8501
- **⚡ Real-time Monitor**: http://localhost:8504  
- **📊 Standard Analysis**: http://localhost:8502

## 📁 Project Structure

```
git-ml/
├── 📂 src/                    # Core application modules
│   ├── git_analyzer.py        # Git repository cloning and log extraction
│   ├── git_feature_extractor.py # Git-specific security feature extraction  
│   ├── ml_anomaly_criteria.py # ML-driven anomaly criteria learning
│   ├── anomaly_detector.py    # Traditional isolation forest (legacy)
│   └── utils.py              # Visualization & utility functions
│
├── 📂 ui/                     # Web interfaces
│   ├── realtime_app.py        # Real-time monitoring web interface
│   ├── git_app.py            # Standard Git analysis web app  
│   └── app.py                # Legacy synthetic data app
│
├── 📂 tests/                  # Unit tests
│   └── test_main.py          # Comprehensive test suite
│
├── 📄 main.py                # 🌟 UNIFIED LAUNCHER - Start here!
├── 📄 git_demo.py            # Standard Git repository analysis CLI
├── 📄 realtime_demo.py       # Real-time monitoring CLI
├── 📄 test_realtime.py       # Real-time system tests
├── 📄 run_main.bat           # Main launcher (Windows)
├── 📄 run_realtime_app.bat   # Real-time web app launcher
├── 📄 run_realtime_demo.bat  # Real-time CLI launcher  
├── 📄 run_git_app.bat        # Standard web app launcher
├── 📄 run_git_demo.bat       # Standard CLI launcher
├── 📄 requirements.txt      # Python dependencies
└── 📄 README.md             # This documentation
```

## 🔬 How It Works

### 🔍 **1. Repository Analysis**
The system clones any public Git repository and extracts:
- **Commit Metadata**: Author, timestamp, message, files changed
- **Activity Patterns**: Timing, frequency, code change volumes
- **Contributor Behavior**: User patterns across time and repositories
- **Security Indicators**: Sensitive file changes, unusual timing, large commits

### 🤖 **2. ML-Driven Learning**
Instead of using pre-defined rules, the system learns from your data:
- **Isolation Forest**: Identifies statistical outliers in contributor behavior
- **Clustering Analysis**: Groups similar anomalous patterns
- **Threshold Optimization**: Finds optimal detection thresholds per feature
- **Pattern Classification**: Discovers distinct anomaly types automatically

### � **3. Git-Specific Security Features**

#### **Behavioral Features**
- `commit_frequency`: How often a user commits
- `lines_changed_avg/max`: Average and maximum lines changed per commit
- `files_modified_avg/max`: File modification patterns
- `commit_message_length_avg`: Commit message thoroughness

#### **Timing Features**  
- `off_hours_commits`: Commits outside normal work hours
- `weekend_commits`: Weekend activity levels
- `unusual_time_patterns`: Irregular timing detection
- `rapid_commits_sequences`: Automated commit pattern detection

#### **Risk Features**
- `large_commits_count`: Commits with massive changes
- `sensitive_commits_count`: Modifications to sensitive files
- `external_domain_commits`: Activity from external email domains
- `repository_diversity`: Access patterns across multiple repos

#### **Activity Features**
- `days_active`: Number of active days in period
- `commits_per_day_avg/max`: Daily activity patterns
- `merge_commits/revert_commits`: Repository management actions

### 🎯 **4. Intelligent Classification**

The ML system automatically categorizes anomalies based on learned patterns:

**🔴 Critical Risk**
- Massive data downloads before contributor departure
- Commits to sensitive files from external domains
- Unusual access patterns to multiple repositories

**🟡 High Risk**  
- Consistent off-hours activity without business justification
- Large commits combined with sensitive file changes
- Rapid automated commit sequences

**🟠 Medium Risk**
- Unusual timing patterns that might be explainable
- Higher than normal activity levels
- Repository access diversity outside normal patterns

**🟢 Low Risk**
- Minor deviations from normal patterns
- Occasional off-hours commits
- Slightly elevated activity levels

## 🎮 How to Use

### 🚀 **Quick Start (Recommended)**

#### 🎯 **Unified Launcher**
```bash
# Windows: Double-click or run from command line
run_main.bat
# OR
python main.py
```

**Main Menu Options:**
- **🌐 Real-time Web Interface** - Interactive monitoring dashboard
- **💻 Real-time Command Line** - Terminal-based live monitoring  
- **🖥️ Standard Web Interface** - One-time repository analysis
- **⌨️ Standard Command Line** - Terminal-based analysis
- **🧪 Test System Components** - Verify installation
- **📚 View Documentation** - Open README.md
- **📦 Install/Update Dependencies** - Setup requirements

### ⚡ **Real-time Monitoring**

#### 🌐 **Web Interface**
```bash
# Start real-time monitoring web app  
run_realtime_app.bat
# OR
streamlit run ui/realtime_app.py --server.port=8503
```

**Real-time Workflow:**
1. **🔗 Start Session**: Enter Git repository URL to begin monitoring
2. **🤖 ML Learning**: System analyzes historical commits and learns normal patterns  
3. **⚡ Live Monitoring**: Continuously watches for new commits every 30 seconds
4. **🚨 Instant Alerts**: Real-time notifications when anomalies are detected
5. **📊 Session Management**: Monitor multiple repositories simultaneously
6. **🗑️ Close Session**: End monitoring and clean up all data

#### 🖥️ **Command Line Interface**
```bash
# Interactive monitoring
run_realtime_demo.bat
# OR
python realtime_demo.py [repo_url] [session_name]

# Available commands:
start <repo_url> [name]  - Start monitoring session
status                   - Show current session status  
alerts [limit]           - Show recent security alerts
sessions                 - List all active sessions
switch <session_name>    - Switch between sessions
close                    - Close current session (clears all data)
help                     - Show all commands
quit                     - Exit (monitoring continues in background)
```

### 🔍 **Standard Analysis Mode**

### 🌐 **Web Interface Workflow**

1. **� Repository Input**
   - Enter any public Git repository URL
   - Set analysis date range (default: last 3 months)  
   - Configure ML sensitivity settings

2. **🔍 Analysis Phase**
   - System clones repository and extracts commit logs
   - Processes contributor patterns and extracts security features
   - Shows repository statistics and contributor analysis

3. **🤖 ML Learning Phase**
   - ML algorithms analyze patterns to learn normal vs. anomalous behavior
   - System discovers optimal detection thresholds for your repository
   - Identifies distinct anomaly patterns through clustering

4. **� Detection Phase**
   - Applies learned criteria to detect anomalous contributors
   - Classifies anomalies by risk level and pattern type
   - Provides detailed explanations for each detection

5. **� Results Analysis**
   - Interactive visualizations of detected patterns
   - Detailed contributor profiles and risk assessments
   - Export capabilities for security team review

### 🖥️ **Command-Line Usage**

```bash
# Analyze specific repository
python git_demo.py "https://github.com/owner/repo.git"

# Use default demo repository
python git_demo.py
```

## 🎨 Visualizations & Analytics

### 📊 **Repository Health Dashboard**
- **Activity Timeline**: Daily commit patterns over time
- **Contributor Analysis**: Top contributors and their patterns
- **Risk Distribution**: Pie charts of detected anomaly types
- **Feature Importance**: Which security features are most indicative

### 📈 **Anomaly Investigation**
- **Individual Profiles**: Detailed analysis of each anomalous contributor
- **Pattern Explanations**: Why each anomaly was detected
- **Timeline Analysis**: When suspicious activities occurred
- **Confidence Scoring**: How certain the system is about each detection

## 🔧 Technical Architecture

### 🤖 **Machine Learning Pipeline**
1. **Data Ingestion**: Synthetic audit log generation with realistic patterns
## 🔧 Technical Architecture

### 🗂️ **Data Pipeline**
1. **Git Cloning**: Uses subprocess to clone repositories locally
2. **Log Extraction**: Parses `git log` output for detailed commit information
3. **Feature Engineering**: Transforms raw git data into security-relevant metrics
4. **ML Processing**: Applies unsupervised learning for anomaly detection
5. **Classification**: Uses learned patterns to categorize anomalies

### 🏗️ **System Components**
- **Backend**: Python with pandas, scikit-learn, subprocess for git operations
- **Frontend**: Streamlit with interactive Plotly visualizations  
- **ML Engine**: Isolation Forest + K-means clustering for pattern discovery
- **Git Integration**: Direct git command integration for repository analysis

## 🔒 Security & Privacy

### ✅ **Safe Analysis**
- Only analyzes **publicly accessible** repositories
- **No authentication** required - works with public repos only
- **Local processing** - all analysis happens on your machine
- **Temporary storage** - cloned repos are cleaned up automatically

### ⚠️ **Limitations**
- Cannot analyze private repositories without additional authentication
- Requires Git to be installed on the system
- Network access needed for repository cloning
- Analysis quality depends on repository history depth

## 🎯 Real-World Applications

### 🏢 **Enterprise Security**
- **Developer Behavior Monitoring**: Detect unusual contributor patterns
- **Insider Threat Detection**: Identify potentially malicious activities
- **Compliance Monitoring**: Ensure development practices meet security standards
- **Risk Assessment**: Evaluate repository security health

### 🎓 **Research & Education**
- **Security Analytics Learning**: Understand ML-driven security detection
- **Git Analytics**: Learn about repository analysis techniques
- **Anomaly Detection**: Practical application of unsupervised learning
- **DevOps Security**: Integration of security into development workflows

## 🤝 Contributing

This project welcomes contributions! Areas for enhancement:

- 🔧 **Additional ML algorithms** (SVM, neural networks)  
- 📊 **Enhanced visualizations** and dashboards
- 🔌 **GitHub API integration** for private repositories
- � **Performance optimization** for larger repositories
- 🧪 **Advanced feature engineering** techniques
- 📱 **Mobile-responsive interface** improvements

## 📄 License & Usage

This project is designed for **educational and practical security analysis** purposes.

- ✅ Educational use encouraged
- ✅ Security research and analysis permitted  
- ✅ Commercial evaluation allowed
- ⚠️ Respects repository privacy - public repos only
- ⚠️ Requires Git installation and network access

---

**🎉 Ready to analyze Git repository security? Run the application and discover potential security risks in real repositories!**

**🚀 Quick Start:**
- **Web Interface**: Run `run_git_app.bat` or `streamlit run ui/git_app.py`
- **Command Line**: Run `run_git_demo.bat` or `python git_demo.py`

For questions, issues, or contributions, please refer to the project documentation or create an issue in the repository.
