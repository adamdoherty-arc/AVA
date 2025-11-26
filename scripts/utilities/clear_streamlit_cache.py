"""
Clear Streamlit cache directories
Run this if the 7-Day Scanner still shows old cached data
"""
import os
import shutil

cache_dirs = [
    '.streamlit',
    '__pycache__',
]

user_cache = os.path.expanduser('~/.streamlit')

print("=" * 80)
print("CLEARING STREAMLIT CACHE")
print("=" * 80)

# Clear local cache directories
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"[OK] Removed: {cache_dir}/")
        except Exception as e:
            print(f"[WARN] Could not remove {cache_dir}: {e}")

# Clear user cache
if os.path.exists(user_cache):
    try:
        shutil.rmtree(user_cache)
        print(f"[OK] Removed: {user_cache}")
    except Exception as e:
        print(f"[WARN] Could not remove user cache: {e}")

print("\n" + "=" * 80)
print("CACHE CLEARED")
print("=" * 80)
print("\nNext steps:")
print("1. Restart Streamlit dashboard")
print("2. Navigate to 7-Day Scanner page")
print("3. Should now show 333 opportunities!")
