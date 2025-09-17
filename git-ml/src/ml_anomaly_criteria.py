"""
ML-Driven Anomaly Criteria Detector
Automatically learns optimal anomaly detection rules from data patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MLAnomalyCriteriaDetector:
    def __init__(self, contamination: float = 0.01):
        """
        Initialize the ML-driven anomaly criteria detector
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.isolation_forest = None
        self.feature_thresholds = {}
        self.anomaly_profiles = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def learn_anomaly_criteria(self, feature_data: pd.DataFrame, 
                              feature_names: List[str]) -> Dict:
        """
        Learn optimal anomaly detection criteria from data
        
        Args:
            feature_data: DataFrame with features
            feature_names: List of feature column names
            
        Returns:
            Dictionary with learned criteria
        """
        
        # Prepare data
        print(f"DEBUG: Preparing ML data with features: {feature_names}")
        print(f"DEBUG: Input DataFrame shape: {feature_data.shape}")
        print(f"DEBUG: Feature columns dtypes: {feature_data[feature_names].dtypes.to_dict()}")
        
        # Ensure all features are numeric and handle any remaining issues
        X = feature_data[feature_names].copy()
        
        # Convert any remaining object columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"DEBUG: Converting object column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Fill any NaN values
        X = X.fillna(0)
        
        # Ensure all values are finite
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"DEBUG: Final ML input data shape: {X.shape}")
        print(f"DEBUG: Final ML input dtypes: {X.dtypes.to_dict()}")
        
        if X.empty:
            raise ValueError("No data available for ML training after preprocessing")
        
        # Check for any remaining non-numeric values
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                print(f"ERROR: Column {col} is still non-numeric: {X[col].dtype}")
                print(f"Sample values: {X[col].head().tolist()}")
                raise ValueError(f"Column {col} contains non-numeric data: {X[col].dtype}")
        
        try:
            X_scaled = self.scaler.fit_transform(X)
        except Exception as scale_error:
            print(f"ERROR: Scaling failed: {scale_error}")
            print(f"Input data types: {X.dtypes.to_dict()}")
            print(f"Input data sample:\n{X.head()}")
            raise ValueError(f"Failed to scale features: {scale_error}")
        
        # Step 1: Train isolation forest to identify initial anomalies
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_predictions = self.isolation_forest.fit_predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        # Add results to dataframe
        feature_data_copy = feature_data.copy()
        feature_data_copy['anomaly_score'] = anomaly_scores
        feature_data_copy['is_anomaly'] = (anomaly_predictions == -1)
        
        # Step 2: Analyze anomalous patterns to derive rules
        anomalies = feature_data_copy[feature_data_copy['is_anomaly']]
        normal = feature_data_copy[~feature_data_copy['is_anomaly']]
        
        # Step 3: Statistical analysis for thresholds
        self.feature_thresholds = self._calculate_optimal_thresholds(
            anomalies, normal, feature_names
        )
        
        # Step 4: Cluster anomalies to find distinct patterns
        self.anomaly_profiles = self._discover_anomaly_profiles(
            anomalies, feature_names
        )
        
        # Step 5: Generate interpretable rules
        criteria = self._generate_anomaly_rules()
        
        self.is_trained = True
        
        return {
            'criteria': criteria,
            'feature_thresholds': self.feature_thresholds,
            'anomaly_profiles': self.anomaly_profiles,
            'training_stats': {
                'total_samples': len(feature_data),
                'anomalies_found': len(anomalies),
                'anomaly_rate': len(anomalies) / len(feature_data),
                'num_profiles': len(self.anomaly_profiles)
            }
        }
    
    def detect_anomalies_with_learned_rules(self, features_df: pd.DataFrame, learned_rules: Dict) -> pd.DataFrame:
        """
        Apply learned anomaly detection rules to new data
        
        Args:
            features_df: DataFrame with extracted features
            learned_rules: Previously learned anomaly detection rules
            
        Returns:
            DataFrame with detected anomalies and explanations
        """
        if features_df.empty or not learned_rules:
            return pd.DataFrame()
        
        try:
            # Use isolation forest if available
            if hasattr(self, 'isolation_forest') and self.isolation_forest is not None:
                model = self.isolation_forest
                
                # Get available features
                feature_names = [col for col in features_df.columns if col.startswith(('commit_', 'lines_', 'files_', 'off_hours', 'weekend', 'large_', 'sensitive_'))]
                
                if not feature_names:
                    return pd.DataFrame()
                
                # Predict anomalies
                X = features_df[feature_names].fillna(0)
                X_scaled = self.scaler.transform(X) if hasattr(self, 'scaler') and self.scaler else X
                
                anomaly_scores = model.decision_function(X_scaled)
                predictions = model.predict(X_scaled)
                
                # Filter anomalies (prediction = -1 means anomaly)
                anomaly_mask = predictions == -1
                
                if not anomaly_mask.any():
                    return pd.DataFrame()
                
                anomaly_df = features_df[anomaly_mask].copy()
                anomaly_df['anomaly_score'] = anomaly_scores[anomaly_mask]
                
                # Add detailed analysis
                for idx, row in anomaly_df.iterrows():
                    classification = self._analyze_individual_anomaly(row, feature_names)
                    
                    for key, value in classification.items():
                        anomaly_df.at[idx, key] = value
                
                return anomaly_df.reset_index(drop=True)
                
        except Exception as e:
            print(f"Error applying learned rules: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame()
    
    def _analyze_individual_anomaly(self, row: pd.Series, feature_names: List[str]) -> Dict:
        """Analyze individual anomaly and provide explanation"""
        
        classification = {
            'anomaly_type': 'Unknown',
            'risk_level': 'Medium',
            'explanation': 'Unusual pattern detected',
            'contributing_features': []
        }
        
        # Analyze specific patterns
        contributing_factors = []
        
        # Check timing anomalies
        if row.get('off_hours_commits', 0) > 0.5:
            contributing_factors.append('High off-hours activity')
            classification['anomaly_type'] = 'Timing Anomaly'
        
        if row.get('weekend_commits', 0) > 0.3:
            contributing_factors.append('Frequent weekend commits')
            classification['anomaly_type'] = 'Timing Anomaly'
        
        # Check commit size anomalies
        if row.get('lines_changed_max', 0) > 1000:
            contributing_factors.append('Very large commits')
            classification['anomaly_type'] = 'Size Anomaly'
            classification['risk_level'] = 'High'
        
        # Check frequency anomalies
        if row.get('commit_frequency', 0) > 10:
            contributing_factors.append('Extremely high commit frequency')
            classification['anomaly_type'] = 'Frequency Anomaly'
        
        # Check sensitive file access
        if row.get('sensitive_commits_count', 0) > 0:
            contributing_factors.append('Modified sensitive files')
            classification['risk_level'] = 'High'
            classification['anomaly_type'] = 'Security Risk'
        
        # Set contributing features
        classification['contributing_features'] = contributing_factors
        
        # Generate explanation
        if contributing_factors:
            classification['explanation'] = f"Detected: {', '.join(contributing_factors)}"
        
        # Adjust risk level based on multiple factors
        if len(contributing_factors) >= 3:
            classification['risk_level'] = 'Critical'
        elif len(contributing_factors) >= 2:
            classification['risk_level'] = 'High'
        
        return classification
    
    def _calculate_optimal_thresholds(self, anomalies: pd.DataFrame, 
                                    normal: pd.DataFrame, 
                                    feature_names: List[str]) -> Dict:
        """Calculate optimal thresholds for each feature"""
        
        thresholds = {}
        
        for feature in feature_names:
            if feature in anomalies.columns and feature in normal.columns:
                anomaly_values = anomalies[feature].fillna(0)
                normal_values = normal[feature].fillna(0)
                
                if len(anomaly_values) > 0 and len(normal_values) > 0:
                    # Calculate percentile-based thresholds
                    anomaly_median = anomaly_values.median()
                    normal_95th = normal_values.quantile(0.95)
                    normal_99th = normal_values.quantile(0.99)
                    
                    # Use the threshold that best separates anomalies from normal
                    if anomaly_median > normal_99th:
                        threshold = normal_99th
                        confidence = 'high'
                    elif anomaly_median > normal_95th:
                        threshold = normal_95th
                        confidence = 'medium'
                    else:
                        # Use anomaly 25th percentile as threshold
                        threshold = anomaly_values.quantile(0.25)
                        confidence = 'low'
                    
                    thresholds[feature] = {
                        'threshold': threshold,
                        'confidence': confidence,
                        'anomaly_median': anomaly_median,
                        'normal_95th': normal_95th,
                        'normal_99th': normal_99th,
                        'separation_score': anomaly_median - normal_95th
                    }
        
        return thresholds
    
    def _discover_anomaly_profiles(self, anomalies: pd.DataFrame, 
                                 feature_names: List[str]) -> Dict:
        """Discover distinct anomaly profiles using clustering"""
        
        if len(anomalies) < 3:
            return {}
        
        # Prepare data for clustering
        X_anomalies = anomalies[feature_names].fillna(0)
        X_anomalies_scaled = self.scaler.transform(X_anomalies)
        
        profiles = {}
        
        # Try different numbers of clusters
        best_clusters = 2
        best_score = -1
        
        for n_clusters in range(2, min(6, len(anomalies) // 2 + 1)):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_anomalies_scaled)
                
                # Calculate silhouette score
                score = silhouette_score(X_anomalies_scaled, cluster_labels)
                
                if score > best_score:
                    best_score = score
                    best_clusters = n_clusters
            except:
                continue
        
        # Perform final clustering
        try:
            kmeans = KMeans(n_clusters=best_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_anomalies_scaled)
            
            # Analyze each cluster
            for cluster_id in range(best_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = X_anomalies[cluster_mask]
                
                if len(cluster_data) > 0:
                    profile = self._analyze_cluster_profile(cluster_data, cluster_id, feature_names)
                    profiles[f'profile_{cluster_id}'] = profile
                    
        except Exception as e:
            # Fallback: create single profile
            profiles['profile_0'] = self._analyze_cluster_profile(X_anomalies, 0, feature_names)
        
        return profiles
    
    def _analyze_cluster_profile(self, cluster_data: pd.DataFrame, 
                               cluster_id: int, feature_names: List[str]) -> Dict:
        """Analyze a specific anomaly cluster to create a profile"""
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'key_features': {},
            'pattern_description': '',
            'risk_level': 'medium'
        }
        
        # Find the most distinctive features for this cluster
        feature_stats = []
        
        for feature in feature_names:
            if feature in cluster_data.columns:
                values = cluster_data[feature].fillna(0)
                if len(values) > 0 and values.std() > 0:
                    feature_stats.append({
                        'feature': feature,
                        'mean': values.mean(),
                        'max': values.max(),
                        'std': values.std(),
                        'non_zero_ratio': (values > 0).mean()
                    })
        
        # Sort by importance (high mean + high variability)
        feature_stats.sort(key=lambda x: x['mean'] * (1 + x['std']), reverse=True)
        
        # Take top 3-5 most important features
        top_features = feature_stats[:5]
        
        for feat_stat in top_features:
            if feat_stat['mean'] > 0:  # Only include meaningful features
                profile['key_features'][feat_stat['feature']] = {
                    'typical_value': feat_stat['mean'],
                    'max_observed': feat_stat['max'],
                    'variability': feat_stat['std']
                }
        
        # Generate pattern description
        profile['pattern_description'] = self._generate_pattern_description(profile['key_features'])
        
        # Assess risk level based on feature severity
        profile['risk_level'] = self._assess_risk_level(profile['key_features'])
        
        return profile
    
    def _generate_pattern_description(self, key_features: Dict) -> str:
        """Generate human-readable description of anomaly pattern"""
        
        if not key_features:
            return "Unusual behavior pattern detected"
        
        descriptions = []
        
        # Define feature descriptions
        feature_descriptions = {
            'commit_frequency': 'high commit activity',
            'large_commits_count': 'large code changes',
            'off_hours_commits': 'commits outside normal hours',
            'weekend_commits': 'weekend activity',
            'sensitive_commits_count': 'sensitive file modifications',
            'rapid_commits_sequences': 'automated commit patterns',
            'external_domain_commits': 'commits from external domains',
            'lines_changed_max': 'massive code changes',
            'repository_diversity': 'access to many repositories',
            'unusual_time_patterns': 'irregular timing patterns'
        }
        
        for feature, stats in key_features.items():
            if feature in feature_descriptions and stats['typical_value'] > 1:
                descriptions.append(feature_descriptions[feature])
        
        if descriptions:
            return f"Pattern characterized by: {', '.join(descriptions[:3])}"
        else:
            return "Unusual behavior pattern detected"
    
    def _assess_risk_level(self, key_features: Dict) -> str:
        """Assess risk level based on key features"""
        
        high_risk_features = [
            'sensitive_commits_count', 'external_domain_commits',
            'off_hours_commits', 'large_commits_count'
        ]
        
        medium_risk_features = [
            'rapid_commits_sequences', 'weekend_commits',
            'unusual_time_patterns', 'repository_diversity'
        ]
        
        high_risk_score = sum(1 for feat in high_risk_features 
                             if feat in key_features and key_features[feat]['typical_value'] > 2)
        
        medium_risk_score = sum(1 for feat in medium_risk_features 
                               if feat in key_features and key_features[feat]['typical_value'] > 1)
        
        if high_risk_score >= 2:
            return 'critical'
        elif high_risk_score >= 1 or medium_risk_score >= 3:
            return 'high'
        elif medium_risk_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_anomaly_rules(self) -> Dict:
        """Generate interpretable anomaly detection rules"""
        
        rules = {
            'threshold_rules': [],
            'pattern_rules': [],
            'composite_rules': []
        }
        
        # Generate threshold rules
        for feature, threshold_info in self.feature_thresholds.items():
            if threshold_info['confidence'] in ['high', 'medium']:
                rules['threshold_rules'].append({
                    'feature': feature,
                    'operator': '>',
                    'threshold': threshold_info['threshold'],
                    'confidence': threshold_info['confidence'],
                    'description': f"{feature} exceeds {threshold_info['threshold']:.2f}"
                })
        
        # Generate pattern rules from profiles
        for profile_name, profile in self.anomaly_profiles.items():
            if profile['key_features']:
                pattern_rule = {
                    'profile_name': profile_name,
                    'conditions': [],
                    'risk_level': profile['risk_level'],
                    'description': profile['pattern_description']
                }
                
                for feature, stats in profile['key_features'].items():
                    if stats['typical_value'] > 0:
                        pattern_rule['conditions'].append({
                            'feature': feature,
                            'operator': '>',
                            'value': stats['typical_value'] * 0.7,  # Slightly lower threshold
                            'typical': stats['typical_value']
                        })
                
                rules['pattern_rules'].append(pattern_rule)
        
        # Generate composite rules (combinations)
        rules['composite_rules'] = self._generate_composite_rules()
        
        return rules
    
    def _generate_composite_rules(self) -> List[Dict]:
        """Generate composite rules that combine multiple features"""
        
        composite_rules = []
        
        # Rule 1: High activity + off hours = suspicious
        if ('commit_frequency' in self.feature_thresholds and 
            'off_hours_commits' in self.feature_thresholds):
            
            composite_rules.append({
                'name': 'suspicious_activity',
                'conditions': [
                    ('commit_frequency', '>', self.feature_thresholds['commit_frequency']['threshold']),
                    ('off_hours_commits', '>', 2)
                ],
                'description': 'High commit activity during off hours',
                'risk_level': 'high'
            })
        
        # Rule 2: Large commits + sensitive files = data exfiltration risk
        if ('large_commits_count' in self.feature_thresholds and 
            'sensitive_commits_count' in self.feature_thresholds):
            
            composite_rules.append({
                'name': 'data_exfiltration_risk',
                'conditions': [
                    ('large_commits_count', '>', 1),
                    ('sensitive_commits_count', '>', 0)
                ],
                'description': 'Large commits involving sensitive files',
                'risk_level': 'critical'
            })
        
        # Rule 3: Automated patterns = bot or script abuse
        if 'rapid_commits_sequences' in self.feature_thresholds:
            composite_rules.append({
                'name': 'automation_abuse',
                'conditions': [
                    ('rapid_commits_sequences', '>', 2),
                    ('commit_frequency', '>', 10)
                ],
                'description': 'Automated commit patterns detected',
                'risk_level': 'medium'
            })
        
        return composite_rules
    
    def classify_anomaly_with_learned_rules(self, user_data: pd.Series) -> Dict:
        """Classify an anomaly using learned rules"""
        
        if not self.is_trained:
            return {'type': 'unknown', 'risk_level': 'medium', 'reasons': []}
        
        classification = {
            'type': 'unknown',
            'risk_level': 'low',
            'reasons': [],
            'matched_rules': [],
            'confidence': 0.0
        }
        
        total_score = 0
        max_risk_level = 'low'
        
        # Check threshold rules
        for rule in self.feature_thresholds.values():
            if 'threshold' in rule:
                feature_name = None
                for fname, rdata in self.feature_thresholds.items():
                    if rdata == rule:
                        feature_name = fname
                        break
                
                if feature_name and feature_name in user_data.index:
                    value = user_data[feature_name]
                    if value > rule['threshold']:
                        classification['matched_rules'].append(f"{feature_name} threshold exceeded")
                        classification['reasons'].append(f"{feature_name}: {value:.2f} > {rule['threshold']:.2f}")
                        total_score += 1
        
        # Check pattern matches
        best_profile_match = None
        best_match_score = 0
        
        for profile_name, profile in self.anomaly_profiles.items():
            match_score = 0
            total_conditions = len(profile['key_features'])
            
            for feature, stats in profile['key_features'].items():
                if feature in user_data.index:
                    value = user_data[feature]
                    threshold = stats['typical_value'] * 0.7
                    
                    if value >= threshold:
                        match_score += 1
            
            if total_conditions > 0:
                match_ratio = match_score / total_conditions
                if match_ratio > best_match_score:
                    best_match_score = match_ratio
                    best_profile_match = profile
        
        # Apply best matching profile
        if best_profile_match and best_match_score >= 0.5:
            classification['type'] = best_profile_match.get('pattern_description', 'pattern_match')
            classification['risk_level'] = best_profile_match.get('risk_level', 'medium')
            classification['reasons'].append(f"Matches anomaly pattern (confidence: {best_match_score:.2f})")
            total_score += best_match_score * 2
        
        # Calculate final confidence
        classification['confidence'] = min(total_score / 3.0, 1.0)  # Normalize to 0-1
        
        # Determine final risk level
        risk_levels = ['low', 'medium', 'high', 'critical']
        if classification['confidence'] > 0.8:
            classification['risk_level'] = 'critical'
        elif classification['confidence'] > 0.6:
            classification['risk_level'] = 'high'
        elif classification['confidence'] > 0.3:
            classification['risk_level'] = 'medium'
        
        return classification

if __name__ == "__main__":
    # Example usage would go here
    pass
