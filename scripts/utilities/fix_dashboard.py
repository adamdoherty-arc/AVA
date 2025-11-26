#!/usr/bin/env python3
# Script to fix the broken dashboard.py by removing bad lines 1317-1467

with open('dashboard.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find line 1316 (0-indexed = 1315) and line 1468 (0-indexed = 1467)
# Keep everything before 1317 and everything after 1467
cleaned_lines = lines[:1316] + lines[1467:]

with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"Removed lines 1317-1467 ({1467-1316} lines)")
print("Dashboard.py fixed!")
