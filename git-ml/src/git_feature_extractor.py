"""
Git-based Feature Extractor for Security Anomaly Detection
Extracts security-relevant features from real Git repository data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class GitFeatureExtractor:
    def __init__(self):
        """Initialize the Git-based feature extractor"""
        
        # Git-specific security features
        self.git_features = [
            'commit_frequency',
            'lines_changed_avg',
            'lines_changed_max',
            'large_commits_count',
            'sensitive_commits_count',
            'off_hours_commits',
            'weekend_commits',
            'merge_commits',
            'revert_commits',
            'files_modified_avg',
            'files_modified_max',
            'commit_message_length_avg',
            'unique_repositories',
            'days_active',
            'commits_per_day_avg',
            'commits_per_day_max',
            'external_domain_commits',
            'rapid_commits_sequences',
            'unusual_time_patterns',
            'repository_diversity'
        ]
    
    def extract_features(self, audit_logs: pd.DataFrame, 
                        group_by_month: bool = True) -> pd.DataFrame:
        """
        Extract security features from Git audit logs
        
        Args:
            audit_logs: Git audit log data
            group_by_month: Whether to group by user-month or user only
            
        Returns:
            DataFrame with extracted features
        """
        # Debug: Print column information
        print(f"DEBUG: Input DataFrame columns: {list(audit_logs.columns)}")
        print(f"DEBUG: DataFrame shape: {audit_logs.shape}")
        if not audit_logs.empty:
            print(f"DEBUG: Sample data:")
            print(audit_logs.head(2).to_string())
        
        # Handle column name compatibility
        if 'user_id' not in audit_logs.columns:
            if 'username' in audit_logs.columns:
                print("DEBUG: Using 'username' as 'user_id'")
                audit_logs = audit_logs.copy()
                audit_logs['user_id'] = audit_logs['username']
            elif 'email' in audit_logs.columns:
                print("DEBUG: Using 'email' as 'user_id'")
                audit_logs = audit_logs.copy()
                audit_logs['user_id'] = audit_logs['email']
            elif 'author_name' in audit_logs.columns:
                print("DEBUG: Using 'author_name' as 'user_id'")
                audit_logs = audit_logs.copy()
                audit_logs['user_id'] = audit_logs['author_name']
            elif 'author_email' in audit_logs.columns:
                print("DEBUG: Using 'author_email' as 'user_id'")
                audit_logs = audit_logs.copy()
                audit_logs['user_id'] = audit_logs['author_email']
            else:
                raise ValueError(f"No user identifier column found. Available columns: {list(audit_logs.columns)}")
        
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
            features = self._calculate_git_features(group, user_id, year_month)
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def _extract_user_features(self, audit_logs: pd.DataFrame) -> pd.DataFrame:
        """Extract features grouped by user only"""
        
        grouped = audit_logs.groupby('user_id')
        
        feature_data = []
        
        for user_id, group in grouped:
            features = self._calculate_git_features(group, user_id, None)
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def _calculate_git_features(self, group: pd.DataFrame, 
                               user_id: str, year_month) -> Dict:
        """Calculate Git-specific security features for a user group"""
        
        features = {
            'user_id': user_id,
            'year_month': str(year_month) if year_month else 'all'
        }
        
        # Basic commit statistics
        push_commits = group[group['action'] == 'git.push']
        total_commits = len(push_commits)
        features['commit_frequency'] = total_commits
        
        if total_commits > 0:
            # Lines changed analysis
            if 'lines_added' in push_commits.columns and 'lines_deleted' in push_commits.columns:
                lines_changed = push_commits['lines_added'].fillna(0) + push_commits['lines_deleted'].fillna(0)
                features['lines_changed_avg'] = lines_changed.mean()
                features['lines_changed_max'] = lines_changed.max()
            else:
                features['lines_changed_avg'] = 0
                features['lines_changed_max'] = 0
            
            # Files modified analysis
            if 'files_modified' in push_commits.columns:
                files_modified = push_commits['files_modified'].fillna(0)
                features['files_modified_avg'] = files_modified.mean()
                features['files_modified_max'] = files_modified.max()
            else:
                features['files_modified_avg'] = 0
                features['files_modified_max'] = 0
            
            # Commit message analysis
            if 'commit_message' in push_commits.columns:
                message_lengths = push_commits['commit_message'].fillna('').str.len()
                features['commit_message_length_avg'] = message_lengths.mean()
            else:
                features['commit_message_length_avg'] = 0
            
            # Merge and revert commits
            if 'is_merge' in push_commits.columns:
                features['merge_commits'] = push_commits['is_merge'].fillna(False).sum()
            else:
                features['merge_commits'] = 0
            
            if 'is_revert' in push_commits.columns:
                features['revert_commits'] = push_commits['is_revert'].fillna(False).sum()
            else:
                features['revert_commits'] = 0
        else:
            # No commits - set defaults
            for feature in ['lines_changed_avg', 'lines_changed_max', 'files_modified_avg', 
                          'files_modified_max', 'commit_message_length_avg', 'merge_commits', 'revert_commits']:
                features[feature] = 0
        
        # Special action counts
        features['large_commits_count'] = len(group[group['action'] == 'repo.large_commit'])
        features['sensitive_commits_count'] = len(group[group['action'] == 'repo.sensitive_change'])
        features['off_hours_commits'] = len(group[group['action'] == 'repo.off_hours_commit'])
        
        # Time-based analysis
        if len(group) > 0:
            timestamps = pd.to_datetime(group['timestamp'])
            
            # Weekend commits
            weekend_commits = timestamps[timestamps.dt.weekday >= 5]
            features['weekend_commits'] = len(weekend_commits)
            
            # Days active
            unique_days = timestamps.dt.date.nunique()
            features['days_active'] = unique_days
            
            # Commits per day statistics
            if unique_days > 0:
                commits_per_day = len(group) / unique_days
                features['commits_per_day_avg'] = commits_per_day
                
                # Find maximum commits in a single day
                daily_commits = timestamps.dt.date.value_counts()
                features['commits_per_day_max'] = daily_commits.max() if len(daily_commits) > 0 else 0
            else:
                features['commits_per_day_avg'] = 0
                features['commits_per_day_max'] = 0
            
            # Time pattern analysis
            features['unusual_time_patterns'] = self._analyze_time_patterns(timestamps)
            features['rapid_commits_sequences'] = self._detect_rapid_sequences(timestamps)
        else:
            features.update({
                'weekend_commits': 0,
                'days_active': 0,
                'commits_per_day_avg': 0,
                'commits_per_day_max': 0,
                'unusual_time_patterns': 0,
                'rapid_commits_sequences': 0
            })
        
        # Repository and domain analysis
        if 'repository' in group.columns:
            features['unique_repositories'] = group['repository'].nunique()
            features['repository_diversity'] = self._calculate_repo_diversity(group)
        else:
            features['unique_repositories'] = 0
            features['repository_diversity'] = 0
        
        if 'domain' in group.columns:
            external_domains = group[~group['domain'].isin(['github.com', 'localhost', 'company.com'])]
            features['external_domain_commits'] = len(external_domains)
        else:
            features['external_domain_commits'] = 0
        
        # Add metadata
        features['total_actions'] = len(group)
        if 'username' in group.columns:
            features['username'] = group['username'].iloc[0]
        if 'email' in group.columns:
            features['email'] = group['email'].iloc[0]
        
        return features
    
    def _analyze_time_patterns(self, timestamps: pd.Series) -> int:
        """Detect unusual time patterns in commits"""
        
        if len(timestamps) < 5:
            return 0
        
        unusual_count = 0
        
        # Check for commits at very unusual hours (2-5 AM)
        very_late_commits = timestamps[timestamps.dt.hour.between(2, 5)]
        unusual_count += len(very_late_commits)
        
        # Check for rapid succession commits (within 1 minute)
        sorted_times = timestamps.sort_values()
        time_diffs = sorted_times.diff()
        rapid_commits = time_diffs[time_diffs < pd.Timedelta(minutes=1)]
        unusual_count += len(rapid_commits)
        
        return unusual_count
    
    def _detect_rapid_sequences(self, timestamps: pd.Series) -> int:
        """Detect sequences of rapid commits (potential automation)"""
        
        if len(timestamps) < 3:
            return 0
        
        sorted_times = timestamps.sort_values()
        time_diffs = sorted_times.diff()
        
        # Look for sequences where commits are within 30 seconds of each other
        rapid_mask = time_diffs < pd.Timedelta(seconds=30)
        
        # Count sequences of 3 or more rapid commits
        sequences = 0
        current_sequence = 0
        
        for is_rapid in rapid_mask:
            if is_rapid:
                current_sequence += 1
            else:
                if current_sequence >= 2:  # 3+ commits in sequence
                    sequences += 1
                current_sequence = 0
        
        # Check final sequence
        if current_sequence >= 2:
            sequences += 1
        
        return sequences
    
    def _calculate_repo_diversity(self, group: pd.DataFrame) -> float:
        """Calculate diversity score for repository access"""
        
        if 'repository' not in group.columns:
            return 0.0
        
        repo_counts = group['repository'].value_counts()
        
        if len(repo_counts) <= 1:
            return 0.0
        
        # Calculate Shannon diversity index
        proportions = repo_counts / len(group)
        diversity = -sum(p * np.log(p) for p in proportions if p > 0)
        
        return diversity
    
    def get_feature_statistics(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature statistics for Git features"""
        
        numeric_features = [f for f in self.git_features if f in feature_df.columns]
        stats_data = []
        
        for feature in numeric_features:
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
                    'Total_Count': len(values),
                    'Mean': values.mean(),
                    'Std': values.std()
                }
                stats_data.append(stats)
        
        return pd.DataFrame(stats_data)
    
    def prepare_for_ml(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature data for machine learning"""
        
        # Select available Git features
        available_features = [f for f in self.git_features if f in feature_df.columns]
        
        # Create ML-ready dataframe
        ml_data = feature_df[['user_id', 'year_month'] + available_features].copy()
        
        # Fill any NaN values with 0
        ml_data[available_features] = ml_data[available_features].fillna(0)
        
        return ml_data, available_features

if __name__ == "__main__":
    # Example usage would go here
    pass
