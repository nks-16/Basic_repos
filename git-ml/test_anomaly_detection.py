#!/usr/bin/env python3
"""
Test script to verify anomaly detection functionality
"""

import sys
import os
import pandas as pd

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.realtime_monitor import RealtimeGitMonitor
from src.git_analyzer import GitRepositoryAnalyzer
from src.git_feature_extractor import GitFeatureExtractor
from src.ml_anomaly_criteria import MLAnomalyCriteriaDetector

def test_webgoat_anomaly_detection():
    """Test anomaly detection on WebGoat repository"""
    
    print("🔍 Testing Anomaly Detection on WebGoat Repository")
    print("=" * 60)
    
    repo_url = "https://github.com/WebGoat/WebGoat.git"
    repo_path = "./temp_repos/WebGoat"
    
    # Check if repo exists
    if not os.path.exists(repo_path):
        print("❌ WebGoat repository not found. Please run the monitoring app first.")
        return
    
    print(f"📂 Using repository at: {repo_path}")
    
    # 1. Extract commits and convert to audit format
    analyzer = GitRepositoryAnalyzer()
    commits_df = analyzer.extract_audit_logs(repo_path, limit=500)
    print(f"📊 Extracted {len(commits_df)} commits")
    
    # 2. Extract features
    feature_extractor = GitFeatureExtractor()
    features_df = feature_extractor.extract_features(commits_df, group_by_month=True)
    print(f"🧮 Extracted features for {len(features_df)} user-month combinations")
    
    # Print feature statistics
    print(f"📈 Feature columns: {list(features_df.columns)}")
    print(f"📈 Total contributors: {features_df['user_id'].nunique()}")
    
    # 3. Check for obvious suspicious patterns manually
    print("\n🔍 Manual Pattern Analysis:")
    
    # Check for off-hours commits
    off_hours = features_df[features_df['off_hours_commits'] > 5]
    print(f"   • Users with >5 off-hours commits: {len(off_hours)}")
    
    # Check for weekend commits
    weekend = features_df[features_df['weekend_commits'] > 3]
    print(f"   • Users with >3 weekend commits: {len(weekend)}")
    
    # Check for high commit frequency
    high_freq = features_df[features_df['commit_frequency'] > 50]
    print(f"   • Users with >50 commits per month: {len(high_freq)}")
    
    # Check for rapid commits
    rapid = features_df[features_df['rapid_commits_sequences'] > 2]
    print(f"   • Users with >2 rapid commit sequences: {len(rapid)}")
    
    # 4. Train ML model and detect anomalies
    print("\n🤖 ML Anomaly Detection:")
    
    # Get numeric feature columns
    feature_columns = [col for col in features_df.columns 
                      if col not in ['user_id', 'year_month', 'username', 'email']]
    
    print(f"   • Using {len(feature_columns)} features for ML")
    
    # Train ML detector
    ml_detector = MLAnomalyCriteriaDetector(contamination=0.05)  # 5% contamination
    ml_results = ml_detector.learn_anomaly_criteria(features_df, feature_columns)
    
    print(f"   • Training stats: {ml_results.get('training_stats', {})}")
    
    # Apply detection
    anomalies = ml_detector.detect_anomalies_with_learned_rules(features_df, ml_results)
    
    print(f"   • ML detected anomalies: {len(anomalies)}")
    
    if len(anomalies) > 0:
        print("\n🚨 Detected Anomalies:")
        for idx, anomaly in anomalies.head(5).iterrows():
            user = anomaly.get('user_id', 'Unknown')
            score = anomaly.get('anomaly_score', 0)
            commits = anomaly.get('commit_frequency', 0)
            print(f"   • {user}: Score={score:.3f}, Commits={commits}")
    else:
        print("   • No anomalies detected with current settings")
        
        # Try with more sensitive settings
        print("\n🔍 Trying more sensitive detection (10% contamination):")
        ml_detector_sensitive = MLAnomalyCriteriaDetector(contamination=0.1)
        ml_results_sensitive = ml_detector_sensitive.learn_anomaly_criteria(features_df, feature_columns)
        anomalies_sensitive = ml_detector_sensitive.detect_anomalies_with_learned_rules(features_df, ml_results_sensitive)
        
        print(f"   • Sensitive ML detected anomalies: {len(anomalies_sensitive)}")
        
        if len(anomalies_sensitive) > 0:
            print("   🚨 Sensitive Anomalies:")
            for idx, anomaly in anomalies_sensitive.head(3).iterrows():
                user = anomaly.get('user_id', 'Unknown')
                score = anomaly.get('anomaly_score', 0)
                commits = anomaly.get('commit_frequency', 0)
                print(f"      • {user}: Score={score:.3f}, Commits={commits}")
    
    # 5. Test real-time detection with sample commit
    print("\n🔄 Testing Real-time Detection:")
    
    # Get a recent commit to test
    recent_commits = commits_df.head(1)
    if not recent_commits.empty:
        test_commit = recent_commits.iloc[0]
        print(f"   • Testing commit: {test_commit.get('commit_hash', 'Unknown')[:8]}")
        print(f"   • Author: {test_commit.get('username', 'Unknown')}")
        
        # Create feature for this commit
        temp_df = pd.DataFrame([test_commit])
        commit_features = feature_extractor.extract_features(temp_df, group_by_month=False)
        
        if not commit_features.empty:
            commit_anomalies = ml_detector.detect_anomalies_with_learned_rules(commit_features, ml_results)
            if len(commit_anomalies) > 0:
                print("   🚨 This commit would trigger an alert!")
            else:
                print("   ✅ This commit appears normal")
    
    print("\n✅ Anomaly detection test completed!")

if __name__ == "__main__":
    test_webgoat_anomaly_detection()
