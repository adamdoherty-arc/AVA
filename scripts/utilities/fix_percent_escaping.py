"""Fix the percent sign escaping in LIKE clauses"""

# Read the file
with open('src/espn_kalshi_matcher.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix: Escape % in LIKE clauses
# Change LIKE 'pattern%' to LIKE 'pattern%%'
content = content.replace("LIKE 'KXNFLGAME%'", "LIKE 'KXNFLGAME%%'")
content = content.replace("LIKE 'KXNCAAFGAME%'", "LIKE 'KXNCAAFGAME%%'")
content = content.replace("LIKE 'KXNBAGAME%'", "LIKE 'KXNBAGAME%%'")

# Write the fixed file
with open('src/espn_kalshi_matcher.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed espn_kalshi_matcher.py")
print("  - Escaped % signs in LIKE clauses")
print("  - Changed LIKE 'KXNFLGAME%' to LIKE 'KXNFLGAME%%'")
print("  - Changed LIKE 'KXNCAAFGAME%' to LIKE 'KXNCAAFGAME%%'")
