#!/usr/bin/env python3
"""
Test script to verify anomaly detection with artificial suspicious data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.git_feature_extractor import GitFeatureExtractor
from src.ml_anomaly_criteria import MLAnomalyCriteriaDetector

def create_test_data_with_anomalies():
    """Create test data with both normal and suspicious patterns"""
    
    print("🧪 Creating test data with artificial anomalies...")
    
    # Normal users (80% of data)
    normal_users = []
    for i in range(80):
        user_data = {
            'user_id': f'normal_user_{i}',
            'year_month': '2025-09',
            'username': f'normal_{i}',
            'email': f'normal_{i}@company.com',
            'commit_frequency': np.random.randint(5, 25),
            'lines_changed_avg': np.random.uniform(10, 100),
            'lines_changed_max': np.random.uniform(50, 300),
            'files_modified_avg': np.random.uniform(1, 5),
            'files_modified_max': np.random.randint(1, 10),
            'commit_message_length_avg': np.random.uniform(30, 80),
            'merge_commits': np.random.randint(0, 3),
            'revert_commits': np.random.randint(0, 1),
            'large_commits_count': 0,
            'sensitive_commits_count': 0,
            'off_hours_commits': np.random.randint(0, 2),
            'weekend_commits': np.random.randint(0, 3),
            'days_active': np.random.randint(10, 22),
            'commits_per_day_avg': np.random.uniform(0.5, 2.0),
            'commits_per_day_max': np.random.randint(1, 5),
            'unusual_time_patterns': 0,
            'rapid_commits_sequences': 0,
            'unique_repositories': 1,
            'repository_diversity': 0.0,
            'external_domain_commits': 0,
            'total_actions': np.random.randint(5, 25)
        }
        normal_users.append(user_data)
    
    # Suspicious users (20% of data)
    suspicious_users = []
    
    # Type 1: Data exfiltration pattern
    for i in range(5):
        user_data = {
            'user_id': f'suspicious_exfil_{i}',
            'year_month': '2025-09',
            'username': f'exfil_{i}',
            'email': f'exfil_{i}@external.com',
            'commit_frequency': np.random.randint(80, 150),  # Very high
            'lines_changed_avg': np.random.uniform(200, 500),  # Large changes
            'lines_changed_max': np.random.uniform(1000, 2000),  # Very large
            'files_modified_avg': np.random.uniform(10, 20),  # Many files
            'files_modified_max': np.random.randint(25, 50),
            'commit_message_length_avg': np.random.uniform(10, 30),  # Short messages
            'merge_commits': 0,
            'revert_commits': 0,
            'large_commits_count': np.random.randint(5, 10),  # Many large commits
            'sensitive_commits_count': 0,
            'off_hours_commits': np.random.randint(15, 30),  # Many off-hours
            'weekend_commits': np.random.randint(10, 20),  # Many weekends
            'days_active': np.random.randint(25, 30),  # Very active
            'commits_per_day_avg': np.random.uniform(3.0, 6.0),  # High frequency
            'commits_per_day_max': np.random.randint(10, 20),
            'unusual_time_patterns': np.random.randint(5, 10),  # Unusual timing
            'rapid_commits_sequences': np.random.randint(8, 15),  # Rapid sequences
            'unique_repositories': np.random.randint(3, 8),  # Multiple repos
            'repository_diversity': np.random.uniform(0.7, 1.0),
            'external_domain_commits': np.random.randint(20, 40),  # External activity
            'total_actions': np.random.randint(80, 150)
        }
        suspicious_users.append(user_data)
    
    # Type 2: Privilege escalation pattern
    for i in range(3):
        user_data = {
            'user_id': f'suspicious_privesc_{i}',
            'year_month': '2025-09',
            'username': f'privesc_{i}',
            'email': f'privesc_{i}@company.com',
            'commit_frequency': np.random.randint(30, 60),
            'lines_changed_avg': np.random.uniform(50, 150),
            'lines_changed_max': np.random.uniform(200, 500),
            'files_modified_avg': np.random.uniform(3, 8),
            'files_modified_max': np.random.randint(5, 15),
            'commit_message_length_avg': np.random.uniform(15, 40),
            'merge_commits': np.random.randint(0, 2),
            'revert_commits': np.random.randint(2, 5),  # Many reverts (covering tracks)
            'large_commits_count': np.random.randint(2, 5),
            'sensitive_commits_count': np.random.randint(3, 8),  # Sensitive changes
            'off_hours_commits': np.random.randint(8, 15),
            'weekend_commits': np.random.randint(5, 12),
            'days_active': np.random.randint(15, 25),
            'commits_per_day_avg': np.random.uniform(1.5, 3.0),
            'commits_per_day_max': np.random.randint(5, 12),
            'unusual_time_patterns': np.random.randint(3, 7),
            'rapid_commits_sequences': np.random.randint(3, 8),
            'unique_repositories': 1,
            'repository_diversity': 0.0,
            'external_domain_commits': 0,
            'total_actions': np.random.randint(30, 60)
        }
        suspicious_users.append(user_data)
    
    # Type 3: Insider threat pattern
    for i in range(2):
        user_data = {
            'user_id': f'suspicious_insider_{i}',
            'year_month': '2025-09', 
            'username': f'insider_{i}',
            'email': f'insider_{i}@company.com',
            'commit_frequency': np.random.randint(40, 80),
            'lines_changed_avg': np.random.uniform(100, 300),
            'lines_changed_max': np.random.uniform(500, 1000),
            'files_modified_avg': np.random.uniform(5, 12),
            'files_modified_max': np.random.randint(10, 25),
            'commit_message_length_avg': np.random.uniform(20, 50),
            'merge_commits': np.random.randint(1, 4),
            'revert_commits': np.random.randint(1, 3),
            'large_commits_count': np.random.randint(3, 7),
            'sensitive_commits_count': np.random.randint(1, 4),
            'off_hours_commits': np.random.randint(12, 25),  # Working odd hours
            'weekend_commits': np.random.randint(8, 15),
            'days_active': np.random.randint(20, 30),
            'commits_per_day_avg': np.random.uniform(2.0, 4.0),
            'commits_per_day_max': np.random.randint(7, 15),
            'unusual_time_patterns': np.random.randint(4, 8),
            'rapid_commits_sequences': np.random.randint(5, 10),
            'unique_repositories': np.random.randint(2, 5),  # Access to multiple repos
            'repository_diversity': np.random.uniform(0.4, 0.8),
            'external_domain_commits': np.random.randint(5, 15),
            'total_actions': np.random.randint(40, 80)
        }
        suspicious_users.append(user_data)
    
    # Combine all data
    all_users = normal_users + suspicious_users
    df = pd.DataFrame(all_users)
    
    print(f"✅ Created test dataset:")
    print(f"   • Total users: {len(df)}")
    print(f"   • Normal users: {len(normal_users)}")
    print(f"   • Suspicious users: {len(suspicious_users)}")
    print(f"   • Expected anomaly rate: {len(suspicious_users)/len(df)*100:.1f}%")
    
    return df

def test_anomaly_detection_with_artificial_data():
    """Test anomaly detection with artificial suspicious patterns"""
    
    print("🔍 Testing Anomaly Detection with Artificial Data")
    print("=" * 60)
    
    # Create test data
    test_df = create_test_data_with_anomalies()
    
    # Get feature columns
    feature_columns = [col for col in test_df.columns 
                      if col not in ['user_id', 'year_month', 'username', 'email']]
    
    print(f"\n🤖 Training ML model with {len(feature_columns)} features...")
    
    # Test different contamination rates
    contamination_rates = [0.05, 0.1, 0.15, 0.2]
    
    for contamination in contamination_rates:
        print(f"\n📊 Testing with {contamination*100}% contamination rate:")
        
        # Train ML detector
        ml_detector = MLAnomalyCriteriaDetector(contamination=contamination)
        ml_results = ml_detector.learn_anomaly_criteria(test_df, feature_columns)
        
        # Detect anomalies
        anomalies = ml_detector.detect_anomalies_with_learned_rules(test_df, ml_results)
        
        print(f"   • Detected anomalies: {len(anomalies)}")
        print(f"   • Detection rate: {len(anomalies)/len(test_df)*100:.1f}%")
        
        if len(anomalies) > 0:
            # Check if we detected the artificial suspicious users
            detected_suspicious = 0
            false_positives = 0
            
            for _, anomaly in anomalies.iterrows():
                user_id = anomaly['user_id']
                if 'suspicious' in user_id:
                    detected_suspicious += 1
                else:
                    false_positives += 1
            
            print(f"   • Correctly detected suspicious users: {detected_suspicious}/10")
            print(f"   • False positives (normal users flagged): {false_positives}")
            
            # Show top detections
            print(f"   🚨 Top detections:")
            for i, (_, anomaly) in enumerate(anomalies.head(5).iterrows()):
                user = anomaly['user_id']
                score = anomaly.get('anomaly_score', 0)
                commits = anomaly.get('commit_frequency', 0)
                off_hours = anomaly.get('off_hours_commits', 0)
                is_suspicious = '✓' if 'suspicious' in user else '✗'
                print(f"      {i+1}. {user}: Score={score:.3f}, Commits={commits}, Off-hours={off_hours} [{is_suspicious}]")
    
    print(f"\n✅ Artificial anomaly detection test completed!")
    
    return contamination_rates[-1]  # Return best contamination rate

if __name__ == "__main__":
    test_anomaly_detection_with_artificial_data()
