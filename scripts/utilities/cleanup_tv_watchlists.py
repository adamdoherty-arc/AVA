"""
Script to clean up TradingView Watchlists page
Remove positions and trade history sections (lines 1170-1397)
"""

# Read the file
with open('dashboard.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in file: {len(lines)}")

# Lines to delete: 1170-1397 (0-indexed: 1169-1396)
start_line = 1169  # Line 1170 in 1-indexed
end_line = 1397    # Line 1398 in 1-indexed (exclusive)

print(f"Deleting lines {start_line+1} to {end_line}")
print(f"Number of lines to delete: {end_line - start_line}")

# Show what we're deleting (first and last few lines)
print("\nFirst 3 lines to delete:")
for i in range(start_line, min(start_line + 3, end_line)):
    print(f"  {i+1}: {lines[i][:80]}")

print(f"\n... ({end_line - start_line - 6} lines) ...\n")

print("Last 3 lines to delete:")
for i in range(max(start_line, end_line - 3), end_line):
    print(f"  {i+1}: {lines[i][:80]}")

# Delete the lines
new_lines = lines[:start_line] + lines[end_line:]

print(f"\nNew total lines: {len(new_lines)}")
print(f"Lines deleted: {len(lines) - len(new_lines)}")

# Write back
with open('dashboard.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("\n✓ File updated successfully!")
print("TradingView Watchlists page now only contains watchlist management")
