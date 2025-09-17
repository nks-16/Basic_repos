#!/usr/bin/env python3
"""
Simple verification test for anomaly detection
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.ml_anomaly_criteria import MLAnomalyCriteriaDetector

def test_detection_capability():
    """Simple test to verify anomaly detection works"""
    
    print("🔍 ANOMALY DETECTION VERIFICATION TEST")
    print("=" * 50)
    
    # Create simple test data
    print("📊 Creating test dataset...")
    
    # Normal users (low activity, business hours)
    normal_data = {
        'user_id': ['normal_1', 'normal_2', 'normal_3', 'normal_4', 'normal_5'],
        'commit_frequency': [10, 15, 8, 12, 20],
        'off_hours_commits': [0, 1, 0, 0, 2],
        'weekend_commits': [1, 0, 2, 1, 0],
        'lines_changed_avg': [50, 75, 40, 60, 80],
        'rapid_commits_sequences': [0, 0, 1, 0, 0],
        'external_domain_commits': [0, 0, 0, 0, 0]
    }
    
    # Suspicious users (high activity, odd hours, large changes)
    suspicious_data = {
        'user_id': ['SUSPICIOUS_1', 'SUSPICIOUS_2'],
        'commit_frequency': [150, 200],  # Very high
        'off_hours_commits': [25, 30],   # Many late night commits
        'weekend_commits': [15, 20],     # Working weekends
        'lines_changed_avg': [500, 800], # Large changes
        'rapid_commits_sequences': [10, 15], # Rapid sequences
        'external_domain_commits': [20, 35]  # External activity
    }
    
    # Combine datasets
    df_normal = pd.DataFrame(normal_data)
    df_suspicious = pd.DataFrame(suspicious_data)
    df_combined = pd.concat([df_normal, df_suspicious], ignore_index=True)
    
    print(f"   • Normal users: {len(df_normal)}")
    print(f"   • Suspicious users: {len(df_suspicious)}")
    print(f"   • Total: {len(df_combined)}")
    
    # Test with different contamination rates
    feature_cols = ['commit_frequency', 'off_hours_commits', 'weekend_commits', 
                   'lines_changed_avg', 'rapid_commits_sequences', 'external_domain_commits']
    
    for contamination in [0.1, 0.2, 0.3]:
        print(f"\n🤖 Testing with {contamination*100}% contamination rate:")
        
        try:
            # Train detector
            detector = MLAnomalyCriteriaDetector(contamination=contamination)
            rules = detector.learn_anomaly_criteria(df_combined, feature_cols)
            
            # Detect anomalies
            anomalies = detector.detect_anomalies_with_learned_rules(df_combined, rules)
            
            print(f"   ✅ Detected {len(anomalies)} anomalies")
            
            if len(anomalies) > 0:
                print("   🚨 Flagged users:")
                for _, anomaly in anomalies.iterrows():
                    user = anomaly['user_id']
                    is_correct = "✓ CORRECT" if "SUSPICIOUS" in user else "✗ FALSE POSITIVE"
                    commits = anomaly.get('commit_frequency', 0)
                    off_hours = anomaly.get('off_hours_commits', 0)
                    print(f"      • {user}: {commits} commits, {off_hours} off-hours [{is_correct}]")
            else:
                print("   ⚠️  No anomalies detected")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Verification test completed!")

if __name__ == "__main__":
    test_detection_capability()
