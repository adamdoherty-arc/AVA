"""
Test fuzzy matching scores
"""
from thefuzz import fuzz, process

# Test team names
test_names = [
    "Florida State Seminoles",
    "NC State Wolfpack",
    "Ohio State Buckeyes",
    "Georgia Bulldogs",
    "Texas A&M Aggies",
    "Missouri Tigers",
]

# Elo ratings keys
elo_keys = [
    "Georgia", "Alabama", "Ohio State", "Michigan", "Texas",
    "Florida State", "Penn State", "LSU", "Clemson", "Notre Dame",
    "Oregon", "USC", "Oklahoma", "Tennessee", "Auburn",
    "Florida", "Texas A&M", "Ole Miss", "Mississippi State",
    "Arkansas", "Kentucky", "South Carolina", "Missouri",
    "Wisconsin", "Iowa", "Oklahoma State", "Michigan State",
    "Minnesota", "Nebraska", "Northwestern", "Illinois",
    "Indiana", "Purdue", "Maryland", "Rutgers", "Vanderbilt",
    "North Carolina", "Virginia Tech", "Pittsburgh", "Louisville",
    "NC State", "Wake Forest", "Virginia", "Duke",
    "Georgia Tech", "Boston College", "Syracuse", "Baylor",
    "TCU", "Kansas State", "Texas Tech", "West Virginia",
    "Iowa State", "Kansas"
]

print("=" * 100)
print("FUZZY MATCHING TEST")
print("=" * 100)

for test_name in test_names:
    print(f"\nTest: '{test_name}'")
    print("-" * 80)

    # Get top 3 matches
    results = process.extract(test_name, elo_keys, scorer=fuzz.token_sort_ratio, limit=3)

    for match, score in results:
        print(f"  {score:3d}% - '{match}'")

    # Get best match
    best = process.extractOne(test_name, elo_keys, scorer=fuzz.token_sort_ratio)
    if best:
        print(f"  Best: {best[1]}% - '{best[0]}'")
