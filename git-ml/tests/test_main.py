"""
Unit tests for GitHub Security Anomaly Detection
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import GitHubDataGenerator
from feature_extractor import GitHubFeatureExtractor
from anomaly_detector import GitHubAnomalyDetector

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = GitHubDataGenerator(num_users=10, seed=42)
    
    def test_user_generation(self):
        """Test user profile generation"""
        self.assertEqual(len(self.generator.users), 10)
        
        # Check user structure
        user = self.generator.users[0]
        required_fields = ['user_id', 'username', 'email', 'type', 'in_org']
        for field in required_fields:
            self.assertIn(field, user)
    
    def test_audit_log_generation(self):
        """Test audit log generation"""
        from datetime import datetime
        
        start_date = datetime(2023, 11, 1)
        end_date = datetime(2023, 11, 30)
        
        logs = self.generator.generate_audit_logs(start_date, end_date)
        
        self.assertIsInstance(logs, pd.DataFrame)
        self.assertGreater(len(logs), 0)
        
        # Check required columns
        required_cols = ['timestamp', 'user_id', 'action', 'ip_address', 'repository']
        for col in required_cols:
            self.assertIn(col, logs.columns)

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = GitHubFeatureExtractor()
        
        # Create sample data
        generator = GitHubDataGenerator(num_users=5, seed=42)
        self.audit_logs = generator.generate_monthly_data(year=2023, months=[11])
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        features = self.extractor.extract_features(self.audit_logs)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Check some key features exist
        self.assertIn('in_org', features.columns)
        self.assertIn('sum.git.clone', features.columns)
        self.assertIn('count.unique_ips_used', features.columns)
    
    def test_feature_statistics(self):
        """Test feature statistics calculation"""
        features = self.extractor.extract_features(self.audit_logs)
        stats = self.extractor.get_feature_statistics(features)
        
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertIn('Feature', stats.columns)
        self.assertIn('Min', stats.columns)
        self.assertIn('Max', stats.columns)

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        # Generate test data
        generator = GitHubDataGenerator(num_users=50, seed=42)
        audit_logs = generator.generate_monthly_data(year=2023, months=[10, 11])
        
        # Extract features
        extractor = GitHubFeatureExtractor()
        self.features = extractor.extract_features(audit_logs)
        self.ml_data, self.ml_features = extractor.prepare_for_ml(self.features)
        
        self.detector = GitHubAnomalyDetector(contamination=0.1, random_state=42)
    
    def test_model_training(self):
        """Test model training"""
        train_data = self.ml_data[self.ml_data['year_month'] == '2023-10']
        
        results = self.detector.train(train_data, self.ml_features)
        
        self.assertTrue(self.detector.is_trained)
        self.assertIn('num_samples', results)
        self.assertIn('num_anomalies', results)
        self.assertGreater(results['num_samples'], 0)
    
    def test_anomaly_prediction(self):
        """Test anomaly prediction"""
        # Train model first
        train_data = self.ml_data[self.ml_data['year_month'] == '2023-10']
        self.detector.train(train_data, self.ml_features)
        
        # Test prediction
        test_data = self.ml_data[self.ml_data['year_month'] == '2023-11']
        predictions = self.detector.predict(test_data)
        
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertIn('anomaly_score', predictions.columns)
        self.assertIn('is_anomaly', predictions.columns)
        self.assertIn('anomaly_type', predictions.columns)
    
    def test_anomaly_classification(self):
        """Test anomaly classification logic"""
        # Create a test row with suspicious pattern
        test_row = pd.Series({
            'sum.git.clone': 200,
            'sum.repo.download_zip': 15,
            'in_org': 0,  # User left company
            'count.unique_ips_used': 5
        })
        
        classification = self.detector._classify_anomaly(test_row)
        self.assertEqual(classification, 'suspicious')

if __name__ == '__main__':
    unittest.main()
