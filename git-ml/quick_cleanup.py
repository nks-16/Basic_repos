import os
import shutil
import stat

def handle_remove_readonly(func, path, exc):
    """Handle readonly files during removal"""
    if os.path.exists(path):
        os.chmod(path, stat.S_IWRITE)
        func(path)

# Project root
root = r"c:\Users\User\Desktop\Projects\git-ml"

# Remove cache directories
cache_dirs = [
    os.path.join(root, "__pycache__"),
    os.path.join(root, "src", "__pycache__")
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir, onerror=handle_remove_readonly)
            print(f"✅ Removed: {cache_dir}")
        except Exception as e:
            print(f"❌ Failed to remove {cache_dir}: {e}")

# Remove temp directories  
temp_dirs = [
    os.path.join(root, "temp_repos"),
    os.path.join(root, "sessions")
]

for temp_dir in temp_dirs:
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
            print(f"✅ Removed: {temp_dir}")
        except Exception as e:
            print(f"❌ Failed to remove {temp_dir}: {e}")

# Remove specific unwanted files
unwanted_files = [
    "git_demo.py",
    "test_setup.py", 
    "test_monitoring.py",
    "quick_test.py",
    "FIXES_APPLIED.md",
    "INTERFACE_IMPROVEMENTS.md", 
    "USER_FRIENDLY_FIXES.md"
]

for filename in unwanted_files:
    filepath = os.path.join(root, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"✅ Removed file: {filename}")
        except Exception as e:
            print(f"❌ Failed to remove {filename}: {e}")

print("\n🧹 Cleanup completed!")
