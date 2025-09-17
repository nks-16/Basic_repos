"""
Git Repository Analyzer for Security Anomaly Detection
Extracts real commit logs and contributor data from Git repositories
"""

import pandas as pd
import numpy as np
import subprocess
import json
import requests
import os
import shutil
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import re
import os

class GitRepositoryAnalyzer:
    def __init__(self):
        """Initialize the Git repository analyzer"""
        self.repo_path = None
        self.repo_url = None
        
    def clone_repository(self, git_url: str, local_path: str = None) -> str:
        """Clone a Git repository from URL"""
        
        if local_path is None:
            # Generate local path from repo name
            repo_name = git_url.split('/')[-1].replace('.git', '')
            local_path = f"./temp_repos/{repo_name}"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Check if repository already exists and is valid
            if os.path.exists(local_path):
                if os.path.exists(os.path.join(local_path, '.git')):
                    print(f"📁 Repository already exists at {local_path}")
                    print(f"🔄 Pulling latest changes...")
                    
                    # Try to pull latest changes
                    try:
                        result = subprocess.run(
                            ['git', 'pull'], 
                            cwd=local_path, 
                            capture_output=True, 
                            text=True,
                            timeout=60
                        )
                        if result.returncode == 0:
                            print(f"✅ Repository updated successfully")
                            self.repo_path = local_path
                            self.repo_url = git_url
                            return local_path
                        else:
                            print(f"⚠️ Pull failed, will re-clone: {result.stderr}")
                    except Exception as pull_error:
                        print(f"⚠️ Pull failed, will re-clone: {pull_error}")
                
                # Remove existing repository if pull failed or invalid
                print(f"🗑️ Removing existing repository to re-clone...")
                try:
                    if os.name == 'nt':
                        # Windows-specific: Remove read-only attributes
                        try:
                            subprocess.run(['attrib', '-R', f'{local_path}\\*.*', '/S'], 
                                         capture_output=True, shell=True)
                        except:
                            pass  # Ignore attrib errors
                    shutil.rmtree(local_path)
                except Exception as cleanup_error:
                    # Try alternative cleanup methods silently
                    try:
                        if os.name == 'nt':  # Windows
                            subprocess.run(['rmdir', '/S', '/Q', local_path], 
                                         shell=True, capture_output=True)
                        else:  # Linux/Mac
                            subprocess.run(['rm', '-rf', local_path], 
                                         capture_output=True)
                    except Exception as alt_cleanup_error:
                        print(f"ERROR: Failed to cleanup existing repository: {alt_cleanup_error}")
                        raise Exception(f"Cannot remove existing repository at {local_path}")
            
            # Clone the repository
            print(f"📥 Cloning repository from {git_url}...")
            result = subprocess.run(
                ['git', 'clone', git_url, local_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Repository cloned successfully to {local_path}")
                self.repo_path = local_path
                self.repo_url = git_url
                return local_path
            else:
                raise Exception(f"Git clone failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Git clone timeout - repository too large or network issues")
        except Exception as e:
            raise Exception(f"Failed to clone repository: {str(e)}")
    
    def pull_repository(self, repo_path: str) -> bool:
        """
        Pull latest changes from remote repository
        
        Args:
            repo_path: Path to local repository
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ['git', 'pull'], 
                cwd=repo_path, 
                capture_output=True, 
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Error pulling repository: {e}")
            return False
    
    def extract_git_logs(self, repo_path: str, since_hash: str = None, limit: int = 1000) -> pd.DataFrame:
        """
        Extract commit logs from git repository
        
        Args:
            repo_path: Path to the cloned repository
            since_hash: Only get commits after this hash (for incremental updates)
            limit: Maximum number of commits to extract
            
        Returns:
            DataFrame with commit information
        """
        try:
            # First get basic commit info
            cmd = [
                'git', 'log', 
                '--pretty=format:%H|%an|%ae|%at|%s',
                f'--max-count={limit}'
            ]
            
            # Add since parameter if provided
            if since_hash:
                cmd.extend([f'{since_hash}..HEAD'])
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"Git log failed: {result.stderr}")
                return pd.DataFrame()
            
            if not result.stdout.strip():
                print("DEBUG: No commits found in repository")
                return pd.DataFrame()
            
            # Parse git log output
            commits = []
            for line_num, line in enumerate(result.stdout.strip().split('\n')):
                if not line.strip():
                    continue
                
                parts = line.split('|')
                if len(parts) >= 5:
                    try:
                        commit_hash = parts[0].strip()
                        timestamp_int = int(parts[3].strip())
                        
                        commits.append({
                            'commit_hash': commit_hash,
                            'author': parts[1].strip(),
                            'email': parts[2].strip(),
                            'timestamp': timestamp_int,
                            'message': parts[4].strip() if len(parts) > 4 else '',
                            'lines_changed': 0,  # Will be filled later
                            'files_changed': 0   # Will be filled later
                        })
                    except (ValueError, IndexError) as parse_error:
                        print(f"DEBUG: Skipping malformed line {line_num}: {parse_error}")
                        continue
            
            if not commits:
                print("DEBUG: No valid commits could be parsed")
                return pd.DataFrame()
            
            print(f"DEBUG: Successfully parsed {len(commits)} commits")
            
            # Convert to DataFrame
            df = pd.DataFrame(commits)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Get file change statistics for each commit
            for i, commit_hash in enumerate(df['commit_hash']):
                try:
                    # Get file stats for this commit
                    stat_cmd = ['git', 'show', '--stat', '--format=', commit_hash]
                    stat_result = subprocess.run(
                        stat_cmd,
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if stat_result.returncode == 0:
                        # Parse file change stats
                        lines_changed = 0
                        files_changed = 0
                        for stat_line in stat_result.stdout.strip().split('\n'):
                            if '|' in stat_line and ('+' in stat_line or '-' in stat_line):
                                files_changed += 1
                                # Extract numbers from lines like " file.txt | 10 ++++------"
                                parts = stat_line.split('|')
                                if len(parts) > 1:
                                    try:
                                        changes_part = parts[1].strip()
                                        numbers = ''.join(c for c in changes_part if c.isdigit())
                                        if numbers:
                                            lines_changed += int(numbers)
                                    except:
                                        pass
                        
                        df.loc[i, 'lines_changed'] = lines_changed
                        df.loc[i, 'files_changed'] = max(1, files_changed)  # At least 1 file changed
                    else:
                        df.loc[i, 'lines_changed'] = 1  # Default fallback
                        df.loc[i, 'files_changed'] = 1
                        
                except Exception as stat_error:
                    print(f"DEBUG: Could not get stats for commit {commit_hash}: {stat_error}")
                    df.loc[i, 'lines_changed'] = 1
                    df.loc[i, 'files_changed'] = 1
            
            return df.sort_values('timestamp', ascending=False).reset_index(drop=True)
            
        except Exception as e:
            print(f"Error extracting git logs: {e}")
            return pd.DataFrame()
    
    def extract_audit_logs(self, repo_path: str = None, since_hash: str = None, limit: int = 1000) -> pd.DataFrame:
        """
        Extract git logs and convert to security audit format
        
        Args:
            repo_path: Path to repository (uses self.repo_path if None)
            since_hash: Only get commits after this hash
            limit: Maximum number of commits
            
        Returns:
            DataFrame in security audit format with user_id column
        """
        if repo_path is None:
            repo_path = self.repo_path
            
        if repo_path is None:
            raise ValueError("No repository path provided")
        
        # Get raw git logs
        git_logs_df = self.extract_git_logs(repo_path, since_hash, limit)
        
        if git_logs_df.empty:
            return pd.DataFrame()
        
        # Convert DataFrame back to list of dicts for _convert_to_audit_format
        git_logs = []
        for _, row in git_logs_df.iterrows():
            # Handle timestamp properly
            timestamp = row['datetime'] if 'datetime' in row else pd.to_datetime(row['timestamp'], unit='s')
            
            git_logs.append({
                'commit_hash': row['commit_hash'],
                'author_name': row.get('author', ''),
                'author_email': row.get('email', ''),
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp),
                'message': row.get('message', ''),
                'lines_added': row.get('lines_changed', 0) // 2,  # Approximate
                'lines_deleted': row.get('lines_changed', 0) // 2,  # Approximate
                'files_changed': [f"file_{i}" for i in range(max(1, row.get('files_changed', 1)))]  # Placeholder
            })
        
        # Get file stats (placeholder for now)
        file_stats = {'total_files': len(git_logs), 'file_types': {}, 'sensitive_files': []}
        
        # Convert to audit format
        print(f"DEBUG: About to convert {len(git_logs)} git commits to audit format")
        audit_df = self._convert_to_audit_format(git_logs, file_stats)
        print(f"DEBUG: Audit format conversion result - columns: {list(audit_df.columns)}")
        print(f"DEBUG: Audit format shape: {audit_df.shape}")
        
        return audit_df
    
    def _enrich_commit_data(self, df: pd.DataFrame, repo_path: str):
        """Add additional statistics to commit data"""
        try:
            # Add basic statistics that we can calculate from the data
            df['lines_changed'] = 0  # Default value
            df['files_changed'] = 0  # Default value
            df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
            df['weekday'] = pd.to_datetime(df['timestamp'], unit='s').dt.weekday
            df['is_weekend'] = df['weekday'].isin([5, 6])
            df['is_off_hours'] = df['hour'].isin(list(range(22, 24)) + list(range(0, 6)))
            
        except Exception as e:
            print(f"Warning: Could not enrich commit data: {e}")
            # Continue without enrichment
    
    def _parse_git_logs(self, log_output: str) -> List[Dict]:
        """Parse git log output into structured data"""
        
        logs = []
        lines = log_output.strip().split('\n')
        
        current_commit = None
        
        for line in lines:
            if '|' in line and not line.startswith(' '):
                # Commit line
                parts = line.split('|')
                if len(parts) >= 7:
                    current_commit = {
                        'commit_hash': parts[0],
                        'author_name': parts[1],
                        'author_email': parts[2],
                        'timestamp': parts[3],
                        'message': parts[4],
                        'committer_name': parts[5],
                        'committer_email': parts[6],
                        'files_changed': [],
                        'lines_added': 0,
                        'lines_deleted': 0
                    }
                    logs.append(current_commit)
            elif current_commit and line.strip():
                # File change line
                if 'files changed' in line:
                    # Parse summary line: "X files changed, Y insertions(+), Z deletions(-)"
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:
                        current_commit['lines_added'] = int(numbers[1]) if len(numbers) > 1 else 0
                        current_commit['lines_deleted'] = int(numbers[2]) if len(numbers) > 2 else 0
        
        return logs
    
    def _get_file_change_stats(self) -> Dict:
        """Get detailed file change statistics"""
        
        try:
            # Get file types and change patterns
            cmd = [
                'git', '-C', self.repo_path, 'log',
                '--pretty=format:', '--name-only'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            files = [f for f in result.stdout.split('\n') if f.strip()]
            
            # Analyze file patterns
            file_stats = {
                'total_files': len(set(files)),
                'file_types': {},
                'sensitive_files': []
            }
            
            # Categorize files
            sensitive_patterns = [
                r'\.env', r'\.key', r'\.pem', r'\.p12', r'\.jks',
                r'config\.json', r'secrets?\.', r'password',
                r'\.ssh/', r'\.aws/', r'\.docker/'
            ]
            
            for file_path in set(files):
                # File extension
                ext = os.path.splitext(file_path)[1].lower()
                file_stats['file_types'][ext] = file_stats['file_types'].get(ext, 0) + 1
                
                # Check for sensitive files
                for pattern in sensitive_patterns:
                    if re.search(pattern, file_path, re.IGNORECASE):
                        file_stats['sensitive_files'].append(file_path)
                        break
            
            return file_stats
            
        except Exception:
            return {'total_files': 0, 'file_types': {}, 'sensitive_files': []}
    
    def _convert_to_audit_format(self, git_logs: List[Dict], file_stats: Dict) -> pd.DataFrame:
        """Convert git logs to security audit format"""
        
        audit_entries = []
        
        for commit in git_logs:
            # Extract user info
            author_email = commit['author_email']
            author_name = commit['author_name']
            timestamp = pd.to_datetime(commit['timestamp'])
            
            # Determine user ID (use email domain + name)
            domain = author_email.split('@')[-1] if '@' in author_email else 'unknown'
            user_id = f"{author_name.replace(' ', '_').lower()}@{domain}"
            
            # Generate multiple audit entries per commit based on activities
            
            # 1. Code commit entry
            audit_entries.append({
                'timestamp': timestamp,
                'user_id': user_id,
                'username': author_name,
                'email': author_email,
                'action': 'git.push',
                'repository': self.repo_url or self.repo_path,
                'commit_hash': commit['commit_hash'],
                'lines_added': commit['lines_added'],
                'lines_deleted': commit['lines_deleted'],
                'commit_message': commit['message'],
                'is_merge': 'merge' in commit['message'].lower(),
                'is_revert': 'revert' in commit['message'].lower(),
                'files_modified': len(commit['files_changed']),
                'domain': domain,
                'hour': timestamp.hour,
                'day_of_week': timestamp.day_name()
            })
            
            # 2. If large commit, add potential risk entry
            if commit['lines_added'] + commit['lines_deleted'] > 1000:
                audit_entries.append({
                    'timestamp': timestamp,
                    'user_id': user_id,
                    'username': author_name,
                    'email': author_email,
                    'action': 'repo.large_commit',
                    'repository': self.repo_url or self.repo_path,
                    'commit_hash': commit['commit_hash'],
                    'lines_changed': commit['lines_added'] + commit['lines_deleted'],
                    'domain': domain,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.day_name()
                })
            
            # 3. If commit affects sensitive files
            commit_msg_lower = commit['message'].lower()
            if any(word in commit_msg_lower for word in ['secret', 'password', 'key', 'token', 'config']):
                audit_entries.append({
                    'timestamp': timestamp,
                    'user_id': user_id,
                    'username': author_name,
                    'email': author_email,
                    'action': 'repo.sensitive_change',
                    'repository': self.repo_url or self.repo_path,
                    'commit_hash': commit['commit_hash'],
                    'commit_message': commit['message'],
                    'domain': domain,
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.day_name()
                })
            
            # 4. Off-hours commit (potential risk)
            if timestamp.hour < 6 or timestamp.hour > 22 or timestamp.weekday() > 4:
                audit_entries.append({
                    'timestamp': timestamp,
                    'user_id': user_id,
                    'username': author_name,
                    'email': author_email,
                    'action': 'repo.off_hours_commit',
                    'repository': self.repo_url or self.repo_path,
                    'commit_hash': commit['commit_hash'],
                    'hour': timestamp.hour,
                    'day_of_week': timestamp.day_name(),
                    'domain': domain
                })
        
        return pd.DataFrame(audit_entries)
    
    def get_contributor_stats(self) -> pd.DataFrame:
        """Get contributor statistics for the repository"""
        
        if not self.repo_path:
            raise Exception("No repository cloned. Call clone_repository first.")
        
        try:
            # Get contributor statistics
            cmd = [
                'git', '-C', self.repo_path, 'shortlog',
                '-sne', '--all'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            contributors = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # Parse format: "  123  Author Name <email@domain.com>"
                    match = re.match(r'\s*(\d+)\s+(.+?)\s+<(.+)>', line)
                    if match:
                        commit_count, name, email = match.groups()
                        contributors.append({
                            'name': name,
                            'email': email,
                            'commit_count': int(commit_count),
                            'domain': email.split('@')[-1] if '@' in email else 'unknown',
                            'user_id': f"{name.replace(' ', '_').lower()}@{email.split('@')[-1] if '@' in email else 'unknown'}"
                        })
            
            return pd.DataFrame(contributors)
            
        except Exception as e:
            raise Exception(f"Failed to get contributor stats: {str(e)}")
    
    def cleanup(self):
        """Clean up temporary repository files"""
        if self.repo_path and os.path.exists(self.repo_path):
            import shutil
            try:
                shutil.rmtree(self.repo_path)
            except Exception:
                pass  # Ignore cleanup errors

if __name__ == "__main__":
    # Example usage
    analyzer = GitRepositoryAnalyzer()
    
    try:
        # Test with a public repository
        repo_url = "https://github.com/octocat/Hello-World.git"
        print(f"Cloning {repo_url}...")
        
        local_path = analyzer.clone_repository(repo_url)
        print(f"Repository cloned to: {local_path}")
        
        # Extract audit logs
        audit_logs = analyzer.extract_git_logs()
        print(f"Extracted {len(audit_logs)} audit log entries")
        
        # Get contributor stats
        contributors = analyzer.get_contributor_stats()
        print(f"Found {len(contributors)} contributors")
        
        print("\nSample audit logs:")
        print(audit_logs.head())
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        analyzer.cleanup()
