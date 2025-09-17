"""
Real-time Git Repository Security Monitor
Provides continuous monitoring of Git repositories for security anomalies
"""

import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add current directory to path to find sibling modules
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from git_analyzer import GitRepositoryAnalyzer
from git_feature_extractor import GitFeatureExtractor
from ml_anomaly_criteria import MLAnomalyCriteriaDetector

class RealtimeGitMonitor:
    """Real-time monitoring system for Git repository security"""
    
    def __init__(self):
        self.sessions = {}
        self.monitoring_threads = {}
        self.session_data_dir = "sessions"
        self.sessions_loaded = False  # Flag to prevent multiple loading
        os.makedirs(self.session_data_dir, exist_ok=True)
        
        # Load existing sessions on startup (only once)
        if not self.sessions_loaded:
            self._load_existing_sessions()
            self.sessions_loaded = True
    
    def start_monitoring_session(self, repo_url: str, session_name: str = None) -> Dict:
        """
        Start a new monitoring session for a repository
        
        Args:
            repo_url: Git repository URL to monitor
            session_name: Optional custom session name
        
        Returns:
            Session information and initial analysis results
        """
        if session_name is None:
            session_name = self._generate_session_name(repo_url)
        
        if session_name in self.sessions:
            return {
                "info": f"Session '{session_name}' already exists. Resuming monitoring...",
                "session_name": session_name,
                "existing": True
            }
        
        # Validate and fix repository URL
        validated_url = self._validate_and_fix_repo_url(repo_url)
        if validated_url.startswith("ERROR:"):
            return {"error": validated_url[6:]}  # Remove "ERROR:" prefix
        
        print(f"🚀 Starting monitoring session for: {validated_url}")
        
        # Step 1: Clone and analyze repository
        analyzer = GitRepositoryAnalyzer()
        try:
            print(f"📂 Setting up repository...")
            repo_path = analyzer.clone_repository(validated_url)
            print(f"🔍 Analyzing commit history...")
            commits_df = analyzer.extract_audit_logs(repo_path)  # Use audit format instead
            
            if commits_df.empty:
                return {"error": "❌ No commits found in repository - unable to establish baseline patterns"}
            
            print(f"✅ Found {len(commits_df)} commits for analysis")
                
        except Exception as e:
            return {"error": f"❌ Repository setup failed: {str(e)}"}
        
        # Step 2: Extract features from historical data
        print(f"🧮 Extracting security features...")
        feature_extractor = GitFeatureExtractor()
        
        features_df = feature_extractor.extract_features(commits_df)
        
        if features_df.empty:
            return {"error": "❌ No features could be extracted from repository - insufficient data for analysis"}
        
        # Step 3: Train ML model to learn repository-specific patterns
        print(f"🤖 Training ML model on repository patterns...")
        ml_detector = MLAnomalyCriteriaDetector(contamination=0.05)  # 5% for better sensitivity
        
        # Get numeric feature columns only (exclude metadata and string columns)
        exclude_columns = ['user_id', 'year_month', 'username', 'email', 'repository', 'commit_hash', 'domain', 'day_of_week']
        
        # More robust filtering - check both column name and data type
        feature_columns = []
        for col in features_df.columns:
            if col not in exclude_columns:
                # Check if column contains only numeric values
                try:
                    pd.to_numeric(features_df[col], errors='raise')
                    if features_df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'bool']:
                        feature_columns.append(col)
                    elif features_df[col].dtype == 'object':
                        # Try to convert object columns that might be numeric
                        numeric_series = pd.to_numeric(features_df[col], errors='coerce')
                        if not numeric_series.isna().all():  # If some values can be converted
                            features_df[col] = numeric_series.fillna(0)
                            feature_columns.append(col)
                except (ValueError, TypeError):
                    # Skip non-numeric columns silently
                    continue
        
        if not feature_columns:
            return {"error": "❌ No numeric features available for ML training - repository data incompatible"}
        
        print(f"✅ Using {len(feature_columns)} features for ML training")
        
        try:
            ml_results = ml_detector.learn_anomaly_criteria(features_df, feature_columns)
            print(f"🎯 ML model trained successfully - ready for real-time monitoring")
        except Exception as e:
            print(f"ERROR: ML training failed: {e}")
            return {"error": f"❌ ML training failed: {str(e)}"}
        
        # Step 4: Get comprehensive contributor analysis
        try:
            # Get detailed contributor stats from the repository
            contributor_stats = analyzer.get_contributor_stats()
            unique_contributors = len(contributor_stats) if not contributor_stats.empty else 0
            
            # Also get unique contributors from audit data
            audit_contributors = commits_df['email'].nunique() if 'email' in commits_df.columns else 0
            
            # Use the maximum to ensure we capture all contributors
            total_contributors = max(unique_contributors, audit_contributors)
            
            print(f"📊 Contributor Analysis:")
            print(f"   - Git shortlog contributors: {unique_contributors}")
            print(f"   - Audit data contributors: {audit_contributors}")
            print(f"   - Total unique contributors: {total_contributors}")
            
        except Exception as e:
            print(f"⚠️ Could not get detailed contributor stats: {e}")
            total_contributors = features_df['user_id'].nunique() if 'user_id' in features_df.columns else len(features_df)
        
        # Step 5: Get the latest commit hash to track new commits
        latest_commit = commits_df.iloc[0]['commit_hash'] if not commits_df.empty else None
        
        # Step 6: Create session data
        session_data = {
            'session_name': session_name,
            'repo_url': validated_url,
            'repo_path': repo_path,
            'created_at': datetime.now().isoformat(),
            'latest_commit_hash': latest_commit,
            'git_commits': len(analyzer.extract_git_logs(repo_path)),  # Actual Git commits
            'total_historical_commits': len(commits_df),  # Audit entries (may be more due to merges)
            'contributors_analyzed': total_contributors,
            'anomaly_rules': ml_results,
            'feature_extractor': feature_extractor,
            'ml_detector': ml_detector,
            'is_active': True,
            'alerts_count': 0,
            'new_commits_processed': 0
        }
        
        # Step 6: Save session and start monitoring
        self.sessions[session_name] = session_data
        self._save_session_data(session_name)
        
        # Step 7: Start background monitoring thread
        self._start_monitoring_thread(session_name)
        
        # Step 8: Return initial analysis results
        historical_anomalies = self._detect_historical_anomalies(features_df, ml_detector)
        
        return {
            'session_name': session_name,
            'status': 'active',
            'repo_url': validated_url,
            'analysis_complete': True,
            'historical_commits': len(commits_df),
            'contributors': features_df['user_id'].nunique() if 'user_id' in features_df.columns else len(features_df),
            'historical_anomalies': len(historical_anomalies),
            'anomaly_rules_learned': len(ml_results.get('thresholds', {})),
            'monitoring_started': True,
            'message': f'✅ Session started! Monitoring {validated_url} for real-time anomalies.'
        }
    
    def _validate_and_fix_repo_url(self, url: str) -> str:
        """
        Validate and fix common Git repository URL issues
        
        Args:
            url: Input URL to validate and fix
        
        Returns:
            Fixed URL or error message starting with "ERROR:"
        """
        url = url.strip()
        
        # Check for empty URL
        if not url:
            return "ERROR:Please provide a repository URL"
        
        # Fix common GitHub URL issues
        if "github.com" in url:
            # Convert GitHub web URLs to clone URLs
            if "/tree/" in url:
                # Extract repository part before /tree/
                base_url = url.split("/tree/")[0]
                if not base_url.endswith(".git"):
                    base_url += ".git"
                return base_url
            
            if "/blob/" in url:
                # Extract repository part before /blob/
                base_url = url.split("/blob/")[0]
                if not base_url.endswith(".git"):
                    base_url += ".git"
                return base_url
            
            # Ensure .git suffix for GitHub URLs
            if url.startswith("https://github.com/") and not url.endswith(".git"):
                return url + ".git"
        
        # Basic URL validation
        if not (url.startswith("https://") or url.startswith("http://") or url.startswith("git@")):
            return f"ERROR:Invalid URL format. Please use https:// or git@ URL"
        
        # Check for valid repository patterns
        valid_patterns = [
            "github.com",
            "gitlab.com", 
            "bitbucket.org",
            ".git"
        ]
        
        if not any(pattern in url for pattern in valid_patterns):
            return "ERROR:URL doesn't appear to be a valid Git repository. Please check the URL"
        
        return url
    
    def _start_monitoring_thread(self, session_name: str):
        """Start background thread to monitor for new commits"""
        if session_name in self.monitoring_threads:
            return
        
        def monitor_loop():
            while self.sessions.get(session_name, {}).get('is_active', False):
                try:
                    self._check_for_new_commits(session_name)
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"❌ Error in monitoring thread for {session_name}: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        self.monitoring_threads[session_name] = thread
    
    def _check_for_new_commits(self, session_name: str):
        """Check repository for new commits and analyze them"""
        session = self.sessions.get(session_name)
        if not session or not session['is_active']:
            return

        # Update last check time
        session['last_check_time'] = datetime.now().isoformat()

        try:
            # Validate repository path exists
            repo_path = session.get('repo_path')
            if not repo_path or not os.path.exists(repo_path) or not os.path.exists(os.path.join(repo_path, '.git')):
                print(f"⚠️ Repository path invalid, re-cloning...")
                # Re-clone repository
                analyzer = GitRepositoryAnalyzer()
                repo_path = analyzer.clone_repository(session['repo_url'])
                session['repo_path'] = repo_path
                
            # Pull latest changes
            analyzer = GitRepositoryAnalyzer()
            analyzer.repo_path = repo_path  # Set the repo path
            pull_success = analyzer.pull_repository(repo_path)
            
            if not pull_success:
                print(f"⚠️ Pull failed, repository may be corrupted")
                return

            # Get new commits since last check
            latest_hash = session.get('latest_commit_hash')
            if latest_hash:
                new_commits_df = analyzer.extract_audit_logs(
                    repo_path,
                    since_hash=latest_hash
                )
            else:
                # If no latest hash, get the most recent commit to establish baseline
                print(f"🔄 Establishing baseline for monitoring...")
                all_commits_df = analyzer.extract_audit_logs(repo_path)
                if not all_commits_df.empty:
                    session['latest_commit_hash'] = all_commits_df.iloc[0]['commit_hash']
                    self._save_session_data(session_name)
                return  # Skip this check cycle to establish baseline
            
            if new_commits_df.empty:
                # Silently continue - no need to spam console
                return  # No new commits
            
            print(f"🔍 Found {len(new_commits_df)} new commits in {session_name}")
            
            # Process each new commit
            for _, commit in new_commits_df.iterrows():
                alert = self._analyze_new_commit(session_name, commit)
                if alert:
                    self._send_realtime_alert(session_name, alert)
            
            # Update session with latest commit
            session['latest_commit_hash'] = new_commits_df.iloc[0]['commit_hash']
            session['new_commits_processed'] += len(new_commits_df)
            self._save_session_data(session_name)
            
        except Exception as e:
            print(f"❌ Error checking for new commits: {e}")
    
    def _analyze_new_commit(self, session_name: str, commit_data) -> Optional[Dict]:
        """Analyze a single new commit for anomalies"""
        session = self.sessions[session_name]
        
        # Create temporary DataFrame with just this commit
        temp_df = pd.DataFrame([commit_data])
        
        # Extract features for this commit
        features_df = session['feature_extractor'].extract_features(temp_df)
        
        if features_df.empty:
            return None
        
        # Use ML detector to check for anomalies
        ml_detector = session['ml_detector']
        anomalies = ml_detector.detect_anomalies_with_learned_rules(features_df, session['anomaly_rules'])
        
        if not anomalies.empty and len(anomalies) > 0:
            anomaly = anomalies.iloc[0]
            
            # Create alert
            alert = {
                'timestamp': datetime.now().isoformat(),
                'commit_hash': commit_data['commit_hash'][:8],
                'author': commit_data.get('username', 'Unknown'),
                'message': str(commit_data.get('commit_message', ''))[:100],
                'anomaly_type': anomaly.get('anomaly_type', 'Unknown'),
                'risk_level': anomaly.get('risk_level', 'Medium'),
                'anomaly_score': float(anomaly.get('anomaly_score', 0)),
                'contributing_features': anomaly.get('contributing_features', []),
                'explanation': anomaly.get('explanation', 'Unusual pattern detected')
            }
            
            session['alerts_count'] += 1
            return alert
        
        return None
    
    def _send_realtime_alert(self, session_name: str, alert: Dict):
        """Send real-time alert for anomalous commit"""
        print(f"\n🚨 ANOMALY DETECTED in {session_name}")
        print(f"   Commit: {alert['commit_hash']}")
        print(f"   Author: {alert['author']}")
        print(f"   Risk: {alert['risk_level']}")
        print(f"   Type: {alert['anomaly_type']}")
        print(f"   Score: {alert['anomaly_score']:.3f}")
        print(f"   Message: {alert['message']}")
        print(f"   Explanation: {alert['explanation']}")
        
        # Save alert to session
        alert_file = os.path.join(self.session_data_dir, f"{session_name}_alerts.json")
        alerts = []
        
        if os.path.exists(alert_file):
            with open(alert_file, 'r') as f:
                alerts = json.load(f)
        
        alerts.append(alert)
        
        with open(alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def get_session_status(self, session_name: str) -> Dict:
        """Get current status of a monitoring session"""
        session = self.sessions.get(session_name)
        if not session:
            return {"error": f"Session '{session_name}' not found"}
        
        # Format last check time
        last_check = session.get('last_check_time', 'Never')
        if last_check != 'Never' and isinstance(last_check, str):
            try:
                last_check_dt = datetime.fromisoformat(last_check)
                last_check = last_check_dt.strftime("%H:%M:%S")
            except:
                last_check = 'Unknown'
        
        return {
            'session_name': session_name,
            'repo_url': session['repo_url'],
            'status': 'active' if session['is_active'] else 'inactive',
            'created_at': session['created_at'],
            'uptime': self._calculate_uptime(session['created_at']),
            'git_commits': session.get('git_commits', session['total_historical_commits']),
            'historical_commits': session['total_historical_commits'],
            'new_commits_processed': session['new_commits_processed'],
            'total_alerts': session['alerts_count'],
            'contributors_analyzed': session.get('contributors_analyzed', 0),
            'monitoring_active': session['is_active'],
            'last_check_time': last_check,
            'thread_active': session_name in self.monitoring_threads and self.monitoring_threads[session_name].is_alive()
        }
    
    def get_recent_alerts(self, session_name: str, limit: int = 10) -> List[Dict]:
        """Get recent alerts for a session"""
        alert_file = os.path.join(self.session_data_dir, f"{session_name}_alerts.json")
        
        if not os.path.exists(alert_file):
            return []
        
        with open(alert_file, 'r') as f:
            alerts = json.load(f)
        
        return alerts[-limit:] if alerts else []
    
    def list_available_sessions(self) -> List[Dict]:
        """List all available sessions"""
        sessions_info = []
        
        for session_name, session_data in self.sessions.items():
            sessions_info.append({
                'session_name': session_name,
                'repo_url': session_data['repo_url'],
                'created_at': session_data['created_at'],
                'is_active': session_data['is_active'],
                'total_alerts': session_data['alerts_count'],
                'commits_processed': session_data['total_historical_commits'] + session_data['new_commits_processed']
            })
        
        return sorted(sessions_info, key=lambda x: x['created_at'], reverse=True)
    
    def resume_session(self, session_name: str) -> Dict:
        """Resume monitoring for an existing session"""
        if session_name not in self.sessions:
            return {"error": f"Session '{session_name}' not found"}
        
        session = self.sessions[session_name]
        
        # Check if already active
        if session['is_active']:
            return {"info": f"Session '{session_name}' is already active", "session_name": session_name}
        
        try:
            # Recreate the necessary components for monitoring
            print(f"🔄 Resuming session: {session_name}")
            
            # Recreate repository analyzer and setup
            analyzer = GitRepositoryAnalyzer()
            repo_url = session['repo_url']
            
            # Validate and setup repository
            validated_url = self._validate_and_fix_repo_url(repo_url)
            if validated_url.startswith("ERROR:"):
                return {"error": f"Repository validation failed: {validated_url[6:]}"}
            
            # Clone repository if needed
            print(f"📂 Setting up repository for monitoring...")
            repo_path = analyzer.clone_repository(validated_url)
            
            # Get latest commit hash for monitoring baseline
            print(f"🔍 Getting latest commit hash...")
            commits_df = analyzer.extract_audit_logs(repo_path)
            latest_commit_hash = commits_df.iloc[0]['commit_hash'] if not commits_df.empty else None
            
            # Reinitialize feature extractor and ML detector
            feature_extractor = GitFeatureExtractor()
            ml_detector = MLAnomalyCriteriaDetector(contamination=0.05)  # 5% for better sensitivity
            
            # Update session with the recreated components
            session['repo_path'] = repo_path
            session['latest_commit_hash'] = latest_commit_hash
            session['feature_extractor'] = feature_extractor
            session['ml_detector'] = ml_detector
            session['is_active'] = True
            session['resumed_at'] = datetime.now().isoformat()
            session['last_check_time'] = 'Never'  # Reset check time
            
            # Save updated session
            self._save_session_data(session_name)
            
            # Start the monitoring thread
            self._start_monitoring_thread(session_name)
            
            print(f"✅ Successfully resumed monitoring for: {session['repo_url']}")
            
            return {
                'session_name': session_name,
                'status': 'active',
                'message': f"Successfully resumed monitoring for {session['repo_url']}",
                'resumed': True
            }
            
        except Exception as e:
            print(f"❌ Error resuming session {session_name}: {e}")
            session['is_active'] = False
            return {"error": f"Failed to resume session: {str(e)}"}
    
    def close_session(self, session_name: str) -> Dict:
        """Close and cleanup a monitoring session"""
        if session_name not in self.sessions:
            return {"error": f"Session '{session_name}' not found"}
        
        session = self.sessions[session_name]
        
        try:
            # Stop monitoring
            session['is_active'] = False
            session['stopped_at'] = datetime.now().isoformat()
            
            # Save final session state before cleanup
            print(f"📄 Saving session data for {session_name}...")
            self._save_session_data(session_name)
            
        except Exception as e:
            print(f"❌ Error saving session data: {e}")
            print(f"Debug: Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Continue with cleanup even if save fails
        
        try:
            # Wait for thread to stop
            if session_name in self.monitoring_threads:
                thread = self.monitoring_threads[session_name]
                thread.join(timeout=5)
                if thread.is_alive():
                    print(f"⚠️ Warning: Thread for {session_name} did not stop gracefully")
                del self.monitoring_threads[session_name]
        except Exception as e:
            print(f"⚠️ Thread cleanup warning: {e}")
        
        try:
            # Cleanup repository files (optional - user might want to keep them)
            cleanup_repo = True  # Set to False if you want to keep repo files
            if cleanup_repo and 'repo_path' in session and os.path.exists(session['repo_path']):
                try:
                    import shutil
                    shutil.rmtree(session['repo_path'], ignore_errors=True)
                except Exception as e:
                    print(f"⚠️ Could not cleanup repo files: {e}")
        except Exception as e:
            print(f"⚠️ Repo cleanup warning: {e}")
        
        # Keep session files for history but mark as closed
        # Don't delete - just mark as inactive for future reference
        
        # Remove from active sessions memory
        total_alerts = session.get('alerts_count', 0)
        del self.sessions[session_name]
        
        return {
            'message': f'✅ Session "{session_name}" closed successfully',
            'total_alerts_generated': total_alerts,
            'cleanup_completed': True
        }
    
    def delete_session(self, session_name: str) -> Dict:
        """Permanently delete a session and all its data"""
        if session_name not in self.sessions:
            return {"error": f"Session '{session_name}' not found"}
        
        try:
            # First close the session if it's active
            if self.sessions[session_name]['is_active']:
                self.close_session(session_name)
            
            # Remove session files
            session_file = os.path.join(self.session_data_dir, f"{session_name}.json")
            alert_file = os.path.join(self.session_data_dir, f"{session_name}_alerts.json")
            
            for file_path in [session_file, alert_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove from memory if still there
            if session_name in self.sessions:
                del self.sessions[session_name]
            
            return {
                'message': f'✅ Session "{session_name}" deleted permanently',
                'deleted': True
            }
            
        except Exception as e:
            return {"error": f"Failed to delete session: {str(e)}"}
    
    def stop_all_monitoring(self) -> Dict:
        """Force stop all monitoring sessions and cleanup"""
        stopped_sessions = []
        
        # Stop all active sessions
        for session_name in list(self.sessions.keys()):
            if self.sessions[session_name]['is_active']:
                self.sessions[session_name]['is_active'] = False
                stopped_sessions.append(session_name)
        
        # Wait for all threads to stop
        for session_name, thread in list(self.monitoring_threads.items()):
            thread.join(timeout=3)
            del self.monitoring_threads[session_name]
        
        return {
            'message': f'✅ Stopped {len(stopped_sessions)} monitoring sessions',
            'stopped_sessions': stopped_sessions
        }
    
    def list_active_sessions(self) -> List[Dict]:
        """List all active monitoring sessions"""
        active_sessions = []
        
        for name, session in self.sessions.items():
            if session['is_active']:
                active_sessions.append({
                    'session_name': name,
                    'repo_url': session['repo_url'],
                    'created_at': session['created_at'],
                    'uptime': self._calculate_uptime(session['created_at']),
                    'alerts_count': session['alerts_count'],
                    'new_commits': session['new_commits_processed']
                })
        
        return active_sessions
    
    def _generate_session_name(self, repo_url: str) -> str:
        """Generate session name from repository URL"""
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{repo_name}_{timestamp}"
    
    def _save_session_data(self, session_name: str):
        """Save session data to disk"""
        session = self.sessions[session_name].copy()
        
        # Remove non-serializable objects
        session.pop('feature_extractor', None)
        session.pop('ml_detector', None)
        
        def convert_numpy_types(obj):
            """Recursively convert numpy and pandas types to JSON serializable types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif pd.isna(obj):
                return None
            elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
                # Handle numpy scalar types
                try:
                    return obj.item()
                except (ValueError, TypeError):
                    return str(obj)
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                # Handle custom objects by converting to dict
                try:
                    return convert_numpy_types(obj.__dict__)
                except:
                    return str(obj)
            else:
                return obj
        
        # Convert all numpy types recursively
        try:
            serializable_session = convert_numpy_types(session)
            
            session_file = os.path.join(self.session_data_dir, f"{session_name}.json")
            with open(session_file, 'w') as f:
                json.dump(serializable_session, f, indent=2, default=str)
                
        except Exception as e:
            print(f"❌ Error saving session {session_name}: {e}")
            # Fallback: save minimal session data
            try:
                minimal_session = {
                    'session_name': session.get('session_name', session_name),
                    'repo_url': session.get('repo_url', ''),
                    'created_at': session.get('created_at', ''),
                    'is_active': session.get('is_active', False),
                    'alerts_count': int(session.get('alerts_count', 0))
                }
                
                session_file = os.path.join(self.session_data_dir, f"{session_name}.json")
                with open(session_file, 'w') as f:
                    json.dump(minimal_session, f, indent=2)
                    
                print(f"✅ Saved minimal session data for {session_name}")
                
            except Exception as e2:
                print(f"❌ Failed to save even minimal session data: {e2}")
    
    def _load_existing_sessions(self):
        """Load existing sessions from disk on startup"""
        try:
            # Prevent multiple loading
            if self.sessions_loaded:
                return
                
            if not os.path.exists(self.session_data_dir):
                self.sessions_loaded = True
                return
            
            for filename in os.listdir(self.session_data_dir):
                if filename.endswith('.json') and filename not in ['any.json', 'nothing.json']:
                    session_name = filename[:-5]  # Remove .json extension
                    session_file = os.path.join(self.session_data_dir, filename)
                    
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        
                        # Restore session with default values for missing fields
                        # Preserve existing active status if session is already in memory
                        existing_session = self.sessions.get(session_name, {})
                        current_active_status = existing_session.get('is_active', False)
                        
                        # Check if there's actually a monitoring thread running for this session
                        has_active_thread = (session_name in self.monitoring_threads and 
                                           self.monitoring_threads[session_name].is_alive())
                        
                        # Use the most accurate active status
                        actual_active_status = current_active_status or has_active_thread
                        
                        self.sessions[session_name] = {
                            'repo_url': session_data.get('repo_url', ''),
                            'created_at': session_data.get('created_at', datetime.now().isoformat()),
                            'is_active': actual_active_status,  # Use actual active status
                            'git_commits': session_data.get('git_commits', session_data.get('total_historical_commits', 0)),
                            'total_historical_commits': session_data.get('total_historical_commits', 0),
                            'new_commits_processed': session_data.get('new_commits_processed', 0),
                            'alerts_count': session_data.get('alerts_count', 0),
                            'contributors_analyzed': session_data.get('contributors_analyzed', 0),
                            'last_check_time': session_data.get('last_check_time', 'Never'),
                            'anomalies_list': session_data.get('anomalies_list', []),
                            'feature_extractor': existing_session.get('feature_extractor'),  # Preserve existing components
                            'ml_detector': existing_session.get('ml_detector'),  # Preserve existing components
                        }
                        
                        print(f"Loaded session: {session_name}")
                        
                    except Exception as e:
                        print(f"Error loading session {session_name}: {e}")
                        continue
                        
            # Mark sessions as loaded
            self.sessions_loaded = True
            
        except Exception as e:
            print(f"Error loading existing sessions: {e}")
            self.sessions_loaded = True
    
    def _detect_historical_anomalies(self, features_df: pd.DataFrame, ml_detector) -> pd.DataFrame:
        """Detect anomalies in historical data"""
        try:
            print(f"🔍 Analyzing {len(features_df)} user-month records for historical anomalies...")
            
            # Get feature columns for ML
            feature_columns = [col for col in features_df.columns 
                             if col not in ['user_id', 'year_month', 'username', 'email']]
            
            # Apply the learned rules
            anomalies = ml_detector.detect_anomalies_with_learned_rules(features_df, {})
            
            if len(anomalies) > 0:
                print(f"🚨 Found {len(anomalies)} historical anomalies!")
                # Show top anomalies for debugging
                for i, (_, anomaly) in enumerate(anomalies.head(3).iterrows()):
                    user = anomaly.get('user_id', 'Unknown')
                    score = anomaly.get('anomaly_score', 0)
                    commits = anomaly.get('commit_frequency', 0)
                    print(f"   {i+1}. {user}: Score={score:.3f}, Commits={commits}")
            else:
                print(f"ℹ️  No historical anomalies detected with current sensitivity")
                
            return anomalies
        except Exception as e:
            print(f"❌ Error detecting historical anomalies: {e}")
            return pd.DataFrame()
    
    def _calculate_uptime(self, created_at: str) -> str:
        """Calculate session uptime"""
        created = datetime.fromisoformat(created_at)
        uptime = datetime.now() - created
        
        if uptime.days > 0:
            return f"{uptime.days}d {uptime.seconds//3600}h"
        elif uptime.seconds >= 3600:
            return f"{uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
        else:
            return f"{uptime.seconds//60}m {uptime.seconds%60}s"
