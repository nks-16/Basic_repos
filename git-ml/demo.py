"""
Demo script for GitHub Security Anomaly Detection
This script demonstrates the core functionality without the UI
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import GitHubDataGenerator
from src.feature_extractor import GitHubFeatureExtractor
from src.anomaly_detector import GitHubAnomalyDetector
from src.utils import calculate_anomaly_statistics, generate_anomaly_report

def run_demo():
    """Run the complete anomaly detection demo"""
    
    print("🔍 GitHub Security Anomaly Detection Demo")
    print("=" * 50)
    
    # Step 1: Generate synthetic data
    print("\n1️⃣ Generating synthetic audit log data...")
    generator = GitHubDataGenerator(num_users=500, seed=42)
    audit_logs = generator.generate_monthly_data(year=2023, months=[10, 11, 12])
    
    print(f"   ✅ Generated {len(audit_logs):,} audit log entries")
    print(f"   📊 Unique users: {audit_logs['user_id'].nunique()}")
    print(f"   🎯 Unique actions: {audit_logs['action'].nunique()}")
    
    # Step 2: Extract security features
    print("\n2️⃣ Extracting security features...")
    extractor = GitHubFeatureExtractor()
    features = extractor.extract_features(audit_logs)
    ml_data, ml_features = extractor.prepare_for_ml(features)
    
    print(f"   ✅ Extracted features for {len(features):,} user-month combinations")
    print(f"   🔢 Feature count: {len(ml_features)}")
    
    # Step 3: Train anomaly detection model
    print("\n3️⃣ Training Isolation Forest model...")
    train_data = ml_data[ml_data['year_month'] == '2023-10']
    test_data = ml_data[ml_data['year_month'].isin(['2023-11', '2023-12'])]
    
    detector = GitHubAnomalyDetector(contamination=0.009, random_state=42)
    training_results = detector.train(train_data, ml_features)
    
    print(f"   ✅ Model trained on {training_results['num_samples']} samples")
    print(f"   🎯 Training anomalies: {training_results['num_anomalies']} ({training_results['anomaly_rate']:.2%})")
    
    # Step 4: Detect anomalies
    print("\n4️⃣ Detecting anomalies in test data...")
    predictions = detector.predict(test_data)
    
    summary = detector.get_anomaly_summary(predictions)
    
    print(f"   ✅ Analyzed {summary['total_samples']} users")
    print(f"   🚨 Found {summary['total_anomalies']} anomalies ({summary['anomaly_rate']:.2%})")
    
    # Step 5: Analyze results by type
    print("\n5️⃣ Anomaly classification results:")
    if summary['anomaly_types']:
        for anomaly_type, count in summary['anomaly_types'].items():
            emoji = {"suspicious": "🔴", "risky_behavior": "🟡", "misuse": "🟠", "unknown": "⚫"}.get(anomaly_type, "⚫")
            type_name = anomaly_type.replace('_', ' ').title()
            print(f"   {emoji} {type_name}: {count} users")
    
    # Step 6: Show top anomalous users
    print("\n6️⃣ Top 10 most anomalous users:")
    anomalies = predictions[predictions['is_anomaly']].sort_values('anomaly_score').head(10)
    
    for idx, (_, row) in enumerate(anomalies.iterrows(), 1):
        username = row['username']
        user_id = row['user_id']
        score = row['anomaly_score']
        anomaly_type = row['anomaly_type'].replace('_', ' ').title()
        in_org = "Yes" if row['in_org'] == 1 else "No"
        
        print(f"   {idx:2d}. {username} ({user_id}) - Score: {score:.4f}")
        print(f"       Type: {anomaly_type}, In Org: {in_org}")
    
    # Step 7: Feature importance
    print("\n7️⃣ Top contributing features:")
    importance = detector.get_feature_importance(predictions)
    top_features = importance.head(10)
    
    for idx, (_, row) in enumerate(top_features.iterrows(), 1):
        feature = row['feature']
        importance_score = row['importance']
        print(f"   {idx:2d}. {feature}: {importance_score:.3f}")
    
    # Step 8: Generate detailed report
    print("\n8️⃣ Generating detailed report...")
    report = generate_anomaly_report(predictions, ml_features)
    
    # Save report
    report_file = "anomaly_detection_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"   ✅ Report saved to {report_file}")
    
    # Step 9: Export results
    print("\n9️⃣ Exporting results...")
    results_file = "anomaly_results.csv"
    predictions.to_csv(results_file, index=False)
    
    print(f"   ✅ Results exported to {results_file}")
    
    print("\n🎉 Demo completed successfully!")
    print(f"📊 Summary: Found {summary['total_anomalies']} anomalies out of {summary['total_samples']} users")
    print("🌐 For interactive analysis, run: streamlit run ui/app.py")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()
