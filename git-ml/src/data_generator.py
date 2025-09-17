"""
Data Generator for GitHub Security Anomaly Detection
Simulates realistic GitHub audit log data based on the research paper
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
from typing import List, Dict, Tuple

class GitHubDataGenerator:
    def __init__(self, num_users: int = 6000, seed: int = 42):
        """
        Initialize the GitHub audit log data generator
        
        Args:
            num_users: Number of users to simulate (default 6000 as per paper)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        self.fake = Faker()
        Faker.seed(seed)
        
        self.num_users = num_users
        
        # GitHub audit log actions from the research paper
        self.security_actions = [
            'codespaces.policy_group_deleted',
            'codespaces.policy_group_updated',
            'environment.remove_protection_rule',
            'environment.update_protection_rule',
            'git.clone',
            'git.push',
            'hook.create',
            'integration_installation.create',
            'ip_allow_list.disable',
            'ip_allow_list.disable_for_installed_apps',
            'ip_allow_list_entry.create',
            'oauth_application.create',
            'org.add_outside_collaborator',
            'org.recovery_codes_downloaded',
            'org.recovery_code_used',
            'org.recovery_codes_printed',
            'org.recovery_codes_viewed',
            'personal_access_token.request_created',
            'personal_access_token.access_granted',
            'protected_branch.destroy',
            'protected_branch.policy_override',
            'public_key.create',
            'repo.access',
            'repo.download_zip',
            'repository_branch_protection_evaluation.disable',
            'repository_ruleset.destroy',
            'repository_ruleset.update',
            'repository_secret_scanning_push_protection.disable',
            'secret_scanning_push_protection.bypass',
            'ssh_certificate_authority.create',
            'ssh_certificate_requirement.disable'
        ]
        
        # Generate user profiles
        self.users = self._generate_user_profiles()
        
    def _generate_user_profiles(self) -> List[Dict]:
        """Generate user profiles with different behavior patterns"""
        users = []
        
        for i in range(self.num_users):
            user_type = np.random.choice(['normal', 'power_user', 'anomalous'], 
                                       p=[0.85, 0.12, 0.03])
            
            user = {
                'user_id': f"user_{i:05d}",
                'username': self.fake.user_name(),
                'email': self.fake.email(),
                'type': user_type,
                'in_org': np.random.choice([True, False], p=[0.95, 0.05]),
                'join_date': self.fake.date_between(start_date='-2y', end_date='-30d'),
                'base_ip': self.fake.ipv4(),
                'departments': ['engineering', 'data', 'security', 'devops'],
                'department': np.random.choice(['engineering', 'data', 'security', 'devops'])
            }
            users.append(user)
            
        return users
    
    def _generate_ip_addresses(self, user: Dict, num_ips: int) -> List[str]:
        """Generate IP addresses for a user based on their behavior type"""
        if user['type'] == 'anomalous':
            # Anomalous users might use many different IPs
            return [self.fake.ipv4() for _ in range(min(num_ips, 50))]
        else:
            # Normal users typically use 1-3 IPs
            base_ip_parts = user['base_ip'].split('.')
            ips = [user['base_ip']]
            
            for _ in range(min(num_ips - 1, 2)):
                new_ip = f"{base_ip_parts[0]}.{base_ip_parts[1]}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
                ips.append(new_ip)
                
            return ips
    
    def _generate_repositories(self, num_repos: int) -> List[str]:
        """Generate repository names"""
        repo_names = []
        for i in range(num_repos):
            repo_name = f"{self.fake.word()}-{self.fake.word()}-{i}"
            repo_names.append(repo_name)
        return repo_names
    
    def generate_audit_logs(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate GitHub audit logs for the specified date range
        
        Args:
            start_date: Start date for log generation
            end_date: End date for log generation
            
        Returns:
            DataFrame with audit log entries
        """
        logs = []
        repositories = self._generate_repositories(1000)  # Generate pool of repos
        
        current_date = start_date
        while current_date <= end_date:
            # Generate logs for each user for this day
            for user in self.users:
                num_actions = self._get_daily_actions_count(user, current_date)
                
                if num_actions == 0:
                    continue
                    
                # Generate actions for this user on this day
                for _ in range(num_actions):
                    action = self._select_action_for_user(user)
                    
                    log_entry = {
                        'timestamp': current_date + timedelta(
                            hours=np.random.randint(0, 24),
                            minutes=np.random.randint(0, 60),
                            seconds=np.random.randint(0, 60)
                        ),
                        'user_id': user['user_id'],
                        'username': user['username'],
                        'action': action,
                        'ip_address': np.random.choice(self._generate_ip_addresses(user, 5)),
                        'repository': np.random.choice(repositories),
                        'user_type': user['type'],
                        'in_org': user['in_org']
                    }
                    logs.append(log_entry)
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(logs)
    
    def _get_daily_actions_count(self, user: Dict, date: datetime) -> int:
        """Get number of actions for a user on a given day"""
        base_prob = 0.3  # 30% chance of any activity
        
        if user['type'] == 'normal':
            if np.random.random() < base_prob:
                return np.random.poisson(2)  # Average 2 actions
        elif user['type'] == 'power_user':
            if np.random.random() < base_prob * 2:
                return np.random.poisson(8)  # Average 8 actions
        elif user['type'] == 'anomalous':
            if np.random.random() < base_prob * 3:
                return np.random.poisson(20)  # Average 20 actions (anomalous)
                
        return 0
    
    def _select_action_for_user(self, user: Dict) -> str:
        """Select an appropriate action based on user type"""
        if user['type'] == 'anomalous':
            # Anomalous users more likely to perform suspicious actions
            weights = np.ones(len(self.security_actions))
            # Increase probability for suspicious actions
            suspicious_indices = [i for i, action in enumerate(self.security_actions) 
                                if 'git.clone' in action or 'repo.download_zip' in action 
                                or 'protected_branch' in action]
            for idx in suspicious_indices:
                weights[idx] *= 5
            weights = weights / weights.sum()
            return np.random.choice(self.security_actions, p=weights)
        else:
            # Normal users follow typical patterns
            return np.random.choice(self.security_actions)
    
    def generate_monthly_data(self, year: int = 2023, months: List[int] = None) -> pd.DataFrame:
        """
        Generate audit log data for specified months
        
        Args:
            year: Year to generate data for
            months: List of months (1-12), if None generates all 12 months
            
        Returns:
            DataFrame with complete audit log data
        """
        if months is None:
            months = list(range(1, 13))
            
        all_logs = []
        
        for month in months:
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - timedelta(days=1)
                
            month_logs = self.generate_audit_logs(start_date, end_date)
            month_logs['month'] = month
            month_logs['year'] = year
            
            all_logs.append(month_logs)
        
        return pd.concat(all_logs, ignore_index=True)

if __name__ == "__main__":
    # Example usage
    generator = GitHubDataGenerator(num_users=1000)
    
    # Generate data for November 2023
    data = generator.generate_monthly_data(year=2023, months=[11])
    
    print(f"Generated {len(data)} audit log entries")
    print(f"Unique users: {data['user_id'].nunique()}")
    print(f"Unique actions: {data['action'].nunique()}")
    print("\nSample data:")
    print(data.head())
