#!/usr/bin/env python3
"""
Cleanup script to remove unwanted files from the git-ml project
"""

import os
import shutil
import glob

def cleanup_project():
    """Remove unwanted files and directories"""
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Cleaning up project at: {project_root}")
    
    # Files and directories to remove
    cleanup_targets = [
        # Python cache files
        "__pycache__",
        "src/__pycache__",
        
        # Temporary directories
        "temp_repos",
        "sessions",
        
        # Test files
        "git_demo.py",
        "test_setup.py", 
        "test_monitoring.py",
        "quick_test.py",
        
        # Duplicate batch files
        "run_comprehensive_monitor_new.bat",
        "run_realtime_improved.bat", 
        "run_simple_monitor.bat",
        
        # Temporary documentation
        "FIXES_APPLIED.md",
        "INTERFACE_IMPROVEMENTS.md",
        "USER_FRIENDLY_FIXES.md"
    ]
    
    removed_count = 0
    
    for target in cleanup_targets:
        target_path = os.path.join(project_root, target)
        
        try:
            if os.path.isfile(target_path):
                os.remove(target_path)
                print(f"✅ Removed file: {target}")
                removed_count += 1
            elif os.path.isdir(target_path):
                shutil.rmtree(target_path)
                print(f"✅ Removed directory: {target}")
                removed_count += 1
            else:
                print(f"⚠️  Not found: {target}")
        except Exception as e:
            print(f"❌ Error removing {target}: {e}")
    
    # Remove any .pyc files recursively
    pyc_files = glob.glob(os.path.join(project_root, "**", "*.pyc"), recursive=True)
    for pyc_file in pyc_files:
        try:
            os.remove(pyc_file)
            print(f"✅ Removed .pyc file: {os.path.relpath(pyc_file, project_root)}")
            removed_count += 1
        except Exception as e:
            print(f"❌ Error removing {pyc_file}: {e}")
    
    # Remove empty __pycache__ directories
    pycache_dirs = glob.glob(os.path.join(project_root, "**", "__pycache__"), recursive=True)
    for pycache_dir in pycache_dirs:
        try:
            if os.path.exists(pycache_dir) and not os.listdir(pycache_dir):
                os.rmdir(pycache_dir)
                print(f"✅ Removed empty __pycache__: {os.path.relpath(pycache_dir, project_root)}")
                removed_count += 1
        except Exception as e:
            print(f"❌ Error removing {pycache_dir}: {e}")
    
    print(f"\n🧹 Cleanup complete! Removed {removed_count} items.")
    
    # Show remaining project structure
    print("\n📁 Clean project structure:")
    for root, dirs, files in os.walk(project_root):
        # Skip .venv and .git directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        level = root.replace(project_root, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Only show first level of files to avoid clutter
        if level < 2:
            subindent = '  ' * (level + 1)
            for file in sorted(files):
                if not file.startswith('.'):
                    print(f"{subindent}{file}")

if __name__ == "__main__":
    cleanup_project()
