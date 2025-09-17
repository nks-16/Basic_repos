"""
Anomaly Detector for GitHub Security Anomaly Detection
Implements Isolation Forest as described in the research paper
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle
import os

class GitHubAnomalyDetector:
    def __init__(self, contamination: float = 0.009, random_state: int = 42):
        """
        Initialize the anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies (0.9% as per paper)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Anomaly classification thresholds and patterns
        self.classification_rules = {
            'misuse': {
                'conditions': [
                    ('sum.git.clone', '>', 50),
                    ('count.unique_ips_used', '>', 10),
                    ('sum.git.push', '>', 30)
                ],
                'description': 'Large numbers of git clones/pushes + unique IP addresses'
            },
            'risky_behavior': {
                'conditions': [
                    ('sum.protected_branch.destroy', '>', 5),
                    ('sum.protected_branch.policy_override', '>', 10),
                    ('sum.repository_branch_protection_evaluation.disable', '>', 3)
                ],
                'description': 'Large number of branch protection changes or bypasses'
            },
            'suspicious': {
                'conditions': [
                    ('sum.git.clone', '>', 100),
                    ('sum.repo.download_zip', '>', 10),
                    ('in_org', '==', 0)  # User no longer with company
                ],
                'description': 'Large numbers of git clones/downloads + user left company'
            }
        }
    
    def train(self, feature_data: pd.DataFrame, feature_names: List[str]) -> Dict:
        """
        Train the isolation forest model
        
        Args:
            feature_data: DataFrame with features for training
            feature_names: List of feature column names
            
        Returns:
            Dictionary with training results
        """
        
        self.feature_names = feature_names
        
        # Prepare training data
        X_train = feature_data[feature_names].fillna(0)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train isolation forest
        self.isolation_forest.fit(X_train_scaled)
        
        # Generate training scores for analysis
        train_scores = self.isolation_forest.decision_function(X_train_scaled)
        train_predictions = self.isolation_forest.predict(X_train_scaled)
        
        self.is_trained = True
        
        # Calculate training statistics
        num_anomalies = np.sum(train_predictions == -1)
        anomaly_rate = num_anomalies / len(train_predictions)
        
        training_results = {
            'num_samples': len(X_train),
            'num_features': len(feature_names),
            'num_anomalies': num_anomalies,
            'anomaly_rate': anomaly_rate,
            'contamination': self.contamination,
            'score_range': (train_scores.min(), train_scores.max()),
            'feature_names': feature_names
        }
        
        return training_results
    
    def predict(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in new data
        
        Args:
            feature_data: DataFrame with features for prediction
            
        Returns:
            DataFrame with anomaly scores and predictions
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare prediction data
        X_pred = feature_data[self.feature_names].fillna(0)
        
        # Scale features using fitted scaler
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        anomaly_scores = self.isolation_forest.decision_function(X_pred_scaled)
        predictions = self.isolation_forest.predict(X_pred_scaled)
        
        # Create results dataframe
        results = feature_data.copy()
        results['anomaly_score'] = anomaly_scores
        results['is_anomaly'] = (predictions == -1)
        results['anomaly_rank'] = results['anomaly_score'].rank(ascending=True)
        
        # Classify anomalies
        results['anomaly_type'] = results.apply(
            lambda row: self._classify_anomaly(row) if row['is_anomaly'] else 'normal',
            axis=1
        )
        
        return results
    
    def _classify_anomaly(self, row: pd.Series) -> str:
        """
        Classify anomaly type based on feature patterns
        
        Args:
            row: Feature row to classify
            
        Returns:
            Anomaly type classification
        """
        
        classifications = []
        
        for anomaly_type, rules in self.classification_rules.items():
            matches = 0
            total_conditions = len(rules['conditions'])
            
            for feature, operator, threshold in rules['conditions']:
                if feature in row.index:
                    value = row[feature]
                    
                    if operator == '>' and value > threshold:
                        matches += 1
                    elif operator == '==' and value == threshold:
                        matches += 1
                    elif operator == '<' and value < threshold:
                        matches += 1
            
            # If at least 2/3 of conditions are met, classify as this type
            if matches >= max(1, total_conditions * 0.67):
                classifications.append(anomaly_type)
        
        if len(classifications) == 0:
            return 'unknown'
        elif len(classifications) == 1:
            return classifications[0]
        else:
            # If multiple classifications, prioritize by severity
            priority = ['suspicious', 'risky_behavior', 'misuse', 'unknown']
            for p in priority:
                if p in classifications:
                    return p
            return classifications[0]
    
    def get_anomaly_summary(self, results: pd.DataFrame) -> Dict:
        """
        Generate summary of anomaly detection results
        
        Args:
            results: DataFrame with anomaly detection results
            
        Returns:
            Summary statistics
        """
        
        anomalies = results[results['is_anomaly']]
        
        summary = {
            'total_samples': len(results),
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(results),
            'score_range': (results['anomaly_score'].min(), results['anomaly_score'].max()),
            'anomaly_types': anomalies['anomaly_type'].value_counts().to_dict(),
            'top_anomalous_features': self._get_top_anomalous_features(anomalies)
        }
        
        return summary
    
    def _get_top_anomalous_features(self, anomalies: pd.DataFrame) -> Dict:
        """Get features that contribute most to anomalies"""
        
        if len(anomalies) == 0:
            return {}
        
        feature_contributions = {}
        
        for feature in self.feature_names:
            if feature in anomalies.columns:
                mean_value = anomalies[feature].mean()
                if mean_value > 0:
                    feature_contributions[feature] = mean_value
        
        # Sort by contribution
        sorted_features = sorted(feature_contributions.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:10])  # Top 10 features
    
    def get_feature_importance(self, feature_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature importance based on anomaly contribution
        
        Args:
            feature_data: DataFrame with features and anomaly results
            
        Returns:
            DataFrame with feature importance scores
        """
        
        if 'is_anomaly' not in feature_data.columns:
            raise ValueError("Feature data must contain anomaly predictions")
        
        anomalies = feature_data[feature_data['is_anomaly']]
        normal = feature_data[~feature_data['is_anomaly']]
        
        importance_data = []
        
        for feature in self.feature_names:
            if feature in feature_data.columns:
                anomaly_mean = anomalies[feature].mean() if len(anomalies) > 0 else 0
                normal_mean = normal[feature].mean() if len(normal) > 0 else 0
                
                # Calculate difference in means as importance
                importance = abs(anomaly_mean - normal_mean)
                
                importance_data.append({
                    'feature': feature,
                    'importance': importance,
                    'anomaly_mean': anomaly_mean,
                    'normal_mean': normal_mean,
                    'difference': anomaly_mean - normal_mean
                })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'classification_rules': self.classification_rules
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a previously trained model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        self.classification_rules = model_data['classification_rules']
        self.is_trained = True

if __name__ == "__main__":
    # Example usage
    from data_generator import GitHubDataGenerator
    from feature_extractor import GitHubFeatureExtractor
    
    # Generate sample data
    print("Generating sample data...")
    generator = GitHubDataGenerator(num_users=500)
    audit_logs = generator.generate_monthly_data(year=2023, months=[10, 11])
    
    # Extract features
    print("Extracting features...")
    extractor = GitHubFeatureExtractor()
    features = extractor.extract_features(audit_logs)
    ml_data, ml_features = extractor.prepare_for_ml(features)
    
    # Split data for training and testing
    train_data = ml_data[ml_data['year_month'] == '2023-10']
    test_data = ml_data[ml_data['year_month'] == '2023-11']
    
    # Train model
    print("Training anomaly detector...")
    detector = GitHubAnomalyDetector(contamination=0.009)
    training_results = detector.train(train_data, ml_features)
    
    print(f"Training completed:")
    print(f"- Samples: {training_results['num_samples']}")
    print(f"- Features: {training_results['num_features']}")
    print(f"- Anomalies detected: {training_results['num_anomalies']}")
    print(f"- Anomaly rate: {training_results['anomaly_rate']:.3f}")
    
    # Make predictions
    print("\nMaking predictions on test data...")
    predictions = detector.predict(test_data)
    
    # Get summary
    summary = detector.get_anomaly_summary(predictions)
    print(f"Test results:")
    print(f"- Total samples: {summary['total_samples']}")
    print(f"- Anomalies found: {summary['total_anomalies']}")
    print(f"- Anomaly types: {summary['anomaly_types']}")
