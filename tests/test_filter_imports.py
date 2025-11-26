"""
Test script to verify betting filter imports
"""

import sys
import pandas as pd
from datetime import datetime

print("Testing betting filter imports...")

# Test individual filter imports
try:
    from src.betting.filters import (
        BaseBettingFilter,
        ConfidenceFilter,
        ExpectedValueFilter,
        DateRangeFilter,
        GameStatusFilter,
        SportFilter,
        SortFilter,
        BettingFilterPanel
    )
    print("[OK] All filter imports successful")
except Exception as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

# Test filter instantiation
try:
    confidence_filter = ConfidenceFilter()
    ev_filter = ExpectedValueFilter()
    date_filter = DateRangeFilter()
    status_filter = GameStatusFilter()
    sport_filter = SportFilter()
    sort_filter = SortFilter()
    print("[OK] All filters instantiated successfully")
except Exception as e:
    print(f"[ERROR] Instantiation error: {e}")
    sys.exit(1)

# Test BettingFilterPanel
try:
    panel = BettingFilterPanel()
    panel.add_filter(confidence_filter)
    panel.add_filter(ev_filter)
    panel.add_filter(date_filter)
    print(f"[OK] BettingFilterPanel created with {panel.get_filter_count()} filters")
except Exception as e:
    print(f"[ERROR] Panel creation error: {e}")
    sys.exit(1)

# Test filter application on sample data
try:
    sample_data = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team C'],
        'away_team': ['Team X', 'Team Y', 'Team Z'],
        'confidence': [65, 75, 55],
        'ev': [5.0, 8.0, 2.0],
        'game_date': pd.date_range(start='2025-01-01', periods=3),
        'status': ['upcoming', 'live', 'upcoming'],
        'sport': ['NFL', 'NCAA Football', 'NFL']
    })

    # Test confidence filter
    filtered = confidence_filter.apply(sample_data, 60)
    assert len(filtered) == 2, "Confidence filter failed"
    print("[OK] Confidence filter apply() works")

    # Test EV filter
    filtered = ev_filter.apply(sample_data, 4.0)
    assert len(filtered) == 2, "EV filter failed"
    print("[OK] EV filter apply() works")

    # Test status filter
    filtered = status_filter.apply(sample_data, "Upcoming")
    assert len(filtered) == 2, "Status filter failed"
    print("[OK] Status filter apply() works")

    # Test sport filter
    filtered = sport_filter.apply(sample_data, "NFL")
    assert len(filtered) == 2, "Sport filter failed"
    print("[OK] Sport filter apply() works")

    print("\n[SUCCESS] All filter tests passed!")

except Exception as e:
    print(f"[ERROR] Filter test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nFilter library is ready for integration into sports betting pages.")
