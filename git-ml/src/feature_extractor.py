"""
Feature Extractor for GitHub Security Anomaly Detection
Implements the 31 security-related features from the research paper
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class GitHubFeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor with the 31 features from the paper"""
        
        # Binary features
        self.binary_features = ['in_org']
        
        # Sum features (frequency of occurrence)
        self.sum_features = [
            'sum.codespaces.policy_group_deleted',
            'sum.codespaces.policy_group_updated',
            'sum.environment.remove_protection_rule',
            'sum.environment.update_protection_rule',
            'sum.git.clone',
            'sum.git.push',
            'sum.hook.create',
            'sum.integration_installation.create',
            'sum.ip_allow_list.disable',
            'sum.ip_allow_list.disable_for_installed_apps',
            'sum.ip_allow_list_entry.create',
            'sum.oauth_application.create',
            'sum.org.add_outside_collaborator',
            'sum.org.recovery_codes_downloaded',
            'sum.org.recovery_code_used',
            'sum.org.recovery_codes_printed',
            'sum.org.recovery_codes_viewed',
            'sum.personal_access_token.request_created',
            'sum.personal_access_token.access_granted',
            'sum.protected_branch.destroy',
            'sum.protected_branch.policy_override',
            'sum.public_key.create',
            'sum.repo.access',
            'sum.repo.download_zip',
            'sum.repository_branch_protection_evaluation.disable',
            'sum.repository_ruleset.destroy',
            'sum.repository_ruleset.update',
            'sum.repository_secret_scanning_push_protection.disable',
            'sum.secret_scanning_push_protection.bypass',
            'sum.ssh_certificate_authority.create',
            'sum.ssh_certificate_requirement.disable'
        ]
        
        # Count features (unique counts)
        self.count_features = [
            'count.unique_ips_used',
            'count.unique_repos_accessed'
        ]
        
        # Days features (number of days with activity)
        self.days_features = [
            'days.git.clone',
            'days.repo.download_zip'
        ]
        
        self.all_features = (self.binary_features + self.sum_features + 
                           self.count_features + self.days_features)
    
    def extract_features(self, audit_logs: pd.DataFrame, 
                        group_by_month: bool = True) -> pd.DataFrame:
        """
        Extract security features from audit logs
        
        Args:
            audit_logs: Raw audit log data
            group_by_month: Whether to group by user-month or user only
            
        Returns:
            DataFrame with extracted features
        """
        if group_by_month:
            return self._extract_monthly_features(audit_logs)
        else:
            return self._extract_user_features(audit_logs)
    
    def _extract_monthly_features(self, audit_logs: pd.DataFrame) -> pd.DataFrame:
        """Extract features grouped by user and month"""
        
        # Ensure timestamp is datetime
        audit_logs['timestamp'] = pd.to_datetime(audit_logs['timestamp'])
        audit_logs['year_month'] = audit_logs['timestamp'].dt.to_period('M')
        
        # Group by user and month
        grouped = audit_logs.groupby(['user_id', 'year_month'])
        
        feature_data = []
        
        for (user_id, year_month), group in grouped:
            features = self._calculate_features_for_group(group, user_id, year_month)
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def _extract_user_features(self, audit_logs: pd.DataFrame) -> pd.DataFrame:
        """Extract features grouped by user only"""
        
        grouped = audit_logs.groupby('user_id')
        
        feature_data = []
        
        for user_id, group in grouped:
            features = self._calculate_features_for_group(group, user_id, None)
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def _calculate_features_for_group(self, group: pd.DataFrame, 
                                    user_id: str, year_month) -> Dict:
        """Calculate all features for a user group"""
        
        features = {
            'user_id': user_id,
            'year_month': str(year_month) if year_month else 'all'
        }
        
        # Binary feature: in_org
        features['in_org'] = int(group['in_org'].iloc[0])
        
        # Sum features - count occurrences of each action
        for sum_feature in self.sum_features:
            action_name = sum_feature.replace('sum.', '')
            count = len(group[group['action'] == action_name])
            features[sum_feature] = count
        
        # Count features
        features['count.unique_ips_used'] = group['ip_address'].nunique()
        features['count.unique_repos_accessed'] = group['repository'].nunique()
        
        # Days features - count unique days with specific actions
        group['date'] = group['timestamp'].dt.date
        
        git_clone_days = len(group[group['action'] == 'git.clone']['date'].unique())
        features['days.git.clone'] = git_clone_days
        
        repo_download_days = len(group[group['action'] == 'repo.download_zip']['date'].unique())
        features['days.repo.download_zip'] = repo_download_days
        
        # Add metadata
        features['total_actions'] = len(group)
        features['username'] = group['username'].iloc[0]
        features['user_type'] = group['user_type'].iloc[0]
        
        return features
    
    def get_feature_statistics(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature statistics as shown in the research paper
        
        Args:
            feature_df: DataFrame with extracted features
            
        Returns:
            DataFrame with statistical summary
        """
        
        numeric_features = [f for f in self.all_features if f != 'in_org']
        stats_data = []
        
        for feature in numeric_features:
            if feature in feature_df.columns:
                values = feature_df[feature]
                non_zero_values = values[values > 0]
                
                if len(non_zero_values) > 0:
                    stats = {
                        'Feature': feature,
                        'Min': values.min(),
                        '25%': values.quantile(0.25),
                        '50%': values.quantile(0.50),
                        '75%': values.quantile(0.75),
                        '95%': values.quantile(0.95),
                        '99%': values.quantile(0.99),
                        'Max': values.max(),
                        'Non_Zero_Count': len(non_zero_values),
                        'Total_Count': len(values)
                    }
                    stats_data.append(stats)
        
        return pd.DataFrame(stats_data)
    
    def prepare_for_ml(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature data for machine learning
        
        Args:
            feature_df: DataFrame with extracted features
            
        Returns:
            Tuple of (prepared_data, feature_names)
        """
        
        # Select only the ML features
        ml_features = [f for f in self.all_features if f in feature_df.columns]
        
        # Create ML-ready dataframe
        ml_data = feature_df[['user_id', 'year_month'] + ml_features].copy()
        
        # Fill any NaN values with 0
        ml_data[ml_features] = ml_data[ml_features].fillna(0)
        
        return ml_data, ml_features
    
    def identify_outliers(self, feature_df: pd.DataFrame, 
                         percentile_threshold: float = 0.99) -> pd.DataFrame:
        """
        Identify statistical outliers in features
        
        Args:
            feature_df: DataFrame with extracted features
            percentile_threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        
        outlier_df = feature_df.copy()
        numeric_features = [f for f in self.all_features if f != 'in_org']
        
        for feature in numeric_features:
            if feature in feature_df.columns:
                threshold = feature_df[feature].quantile(percentile_threshold)
                outlier_df[f'{feature}_outlier'] = feature_df[feature] > threshold
        
        return outlier_df

if __name__ == "__main__":
    # Example usage
    from data_generator import GitHubDataGenerator
    
    # Generate sample data
    generator = GitHubDataGenerator(num_users=100)
    audit_logs = generator.generate_monthly_data(year=2023, months=[11])
    
    # Extract features
    extractor = GitHubFeatureExtractor()
    features = extractor.extract_features(audit_logs)
    
    print(f"Extracted features for {len(features)} user-month combinations")
    print(f"Feature columns: {len(extractor.all_features)}")
    
    # Get statistics
    stats = extractor.get_feature_statistics(features)
    print("\nFeature statistics:")
    print(stats.head())
    
    # Prepare for ML
    ml_data, ml_features = extractor.prepare_for_ml(features)
    print(f"\nML-ready data shape: {ml_data.shape}")
