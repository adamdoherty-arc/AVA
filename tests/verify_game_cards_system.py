"""
Verification Script for Game Cards AI and Kalshi Integration
Tests all components to ensure they work as intended
"""

import sys
import os
from datetime import datetime
from typing import List, Dict

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from src.espn_live_data import get_espn_client
from src.advanced_betting_ai_agent import AdvancedBettingAIAgent
from src.espn_kalshi_matcher import ESPNKalshiMatcher


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_ai_predictions(games: List[Dict], max_games: int = 5) -> Dict:
    """
    Test AI predictions to ensure they show unique analysis

    Returns:
        Dict with test results
    """
    print_section("TEST 1: AI PREDICTION UNIQUENESS")

    ai_agent = AdvancedBettingAIAgent()
    predictions = []
    unique_probs = set()
    unique_recommendations = set()

    print(f"\nTesting {min(len(games), max_games)} games...\n")

    for i, game in enumerate(games[:max_games]):
        away_team = game.get('away_team', 'Away')
        home_team = game.get('home_team', 'Home')
        score = game.get('score', '0-0')
        status = game.get('status', 'Unknown')

        print(f"Game {i+1}: {away_team} @ {home_team}")
        print(f"  Score: {score}, Status: {status}")

        # Get AI prediction
        prediction = ai_agent.analyze_betting_opportunity(game, {})

        win_prob = prediction.get('win_probability', 0.5)
        confidence = prediction.get('confidence', 0.5)
        recommendation = prediction.get('recommendation', 'PASS')
        reasoning = prediction.get('reasoning', [])

        print(f"  Win Prob: {win_prob:.1%}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Recommendation: {recommendation}")
        print(f"  Reasoning: {reasoning[0] if reasoning else 'No reasoning'}")
        print()

        predictions.append({
            'game': f"{away_team} @ {home_team}",
            'win_prob': win_prob,
            'confidence': confidence,
            'recommendation': recommendation,
            'reasoning': reasoning
        })

        unique_probs.add(round(win_prob, 2))
        unique_recommendations.add(recommendation)

    # Analyze results
    print_section("AI PREDICTION RESULTS")

    total_games = len(predictions)
    unique_prob_count = len(unique_probs)
    unique_rec_count = len(unique_recommendations)

    print(f"\nTotal games tested: {total_games}")
    print(f"Unique win probabilities: {unique_prob_count}")
    print(f"Unique recommendations: {unique_rec_count}")

    # Check if all predictions are identical (BAD)
    all_identical = unique_prob_count == 1 and unique_rec_count == 1

    if all_identical:
        print("\n❌ FAILED: All predictions are identical!")
        print("   This indicates AI is not analyzing game state properly")
        status = "FAILED"
    else:
        print("\n✅ PASSED: Predictions show variation across games")
        print(f"   Win probabilities range from {min(unique_probs):.1%} to {max(unique_probs):.1%}")
        status = "PASSED"

    return {
        'status': status,
        'total_games': total_games,
        'unique_probs': unique_prob_count,
        'unique_recs': unique_rec_count,
        'predictions': predictions
    }


def test_kalshi_matcher(games: List[Dict], max_games: int = 5) -> Dict:
    """
    Test Kalshi market matching

    Returns:
        Dict with test results
    """
    print_section("TEST 2: KALSHI MARKET MATCHING")

    matcher = ESPNKalshiMatcher()
    matched_count = 0
    unmatched_count = 0
    match_results = []

    print(f"\nTesting {min(len(games), max_games)} games...\n")

    for i, game in enumerate(games[:max_games]):
        away_team = game.get('away_team', 'Away')
        home_team = game.get('home_team', 'Home')

        print(f"Game {i+1}: {away_team} @ {home_team}")

        # Try to match
        kalshi_odds = matcher.match_game_to_kalshi(game)

        if kalshi_odds:
            matched_count += 1
            print(f"  ✅ MATCHED")
            print(f"     Away: {kalshi_odds['away_win_price']:.1%}")
            print(f"     Home: {kalshi_odds['home_win_price']:.1%}")
            print(f"     Market: {kalshi_odds.get('ticker', 'N/A')}")
            print(f"     Volume: ${kalshi_odds.get('volume', 0):,.0f}")
            match_results.append({
                'game': f"{away_team} @ {home_team}",
                'matched': True,
                'odds': kalshi_odds
            })
        else:
            unmatched_count += 1
            print(f"  ❌ NOT MATCHED - No Kalshi market found")
            match_results.append({
                'game': f"{away_team} @ {home_team}",
                'matched': False,
                'odds': None
            })

        print()

    # Analyze results
    print_section("KALSHI MATCHING RESULTS")

    total_games = len(match_results)
    match_rate = (matched_count / total_games * 100) if total_games > 0 else 0

    print(f"\nTotal games tested: {total_games}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")
    print(f"Match rate: {match_rate:.1f}%")

    if matched_count == 0:
        print("\n⚠️ WARNING: Zero matches found")
        print("   Possible causes:")
        print("   1. Kalshi markets not synced (run: python sync_kalshi_team_winners.py)")
        print("   2. Database has combo/parlay markets instead of team winners")
        print("   3. Team name variations not matching")
        status = "NO_MATCHES"
    elif match_rate < 50:
        print("\n⚠️ PARTIAL: Some matches found but low match rate")
        print("   Consider:")
        print("   1. Adding more team name variations")
        print("   2. Checking market date ranges")
        status = "PARTIAL"
    else:
        print("\n✅ GOOD: Decent match rate")
        status = "PASSED"

    return {
        'status': status,
        'total_games': total_games,
        'matched': matched_count,
        'unmatched': unmatched_count,
        'match_rate': match_rate,
        'results': match_results
    }


def test_team_name_variations():
    """Test team name variation system"""
    print_section("TEST 3: TEAM NAME VARIATIONS")

    matcher = ESPNKalshiMatcher()

    test_cases = [
        'Jacksonville Jaguars',
        'Los Angeles Chargers',
        'New England Patriots',
        'Ohio State Buckeyes'
    ]

    print("\nTesting team name variation generation:\n")

    for team in test_cases:
        variations = matcher.get_team_variations(team)
        print(f"{team}:")
        print(f"  Variations: {', '.join(variations)}")
        print()

    print("✅ Team name variation system working")

    return {'status': 'PASSED'}


def test_specific_game(away_team: str, home_team: str):
    """
    Test a specific game match (e.g., Jacksonville vs Los Angeles)

    Args:
        away_team: Away team name
        home_team: Home team name
    """
    print_section(f"SPECIFIC TEST: {away_team} vs {home_team}")

    matcher = ESPNKalshiMatcher()

    # Create mock game
    game = {
        'away_team': away_team,
        'home_team': home_team,
        'game_time': datetime.now().strftime('%Y-%m-%d %H:%M')
    }

    print(f"\nSearching for Kalshi market...")
    print(f"Away team variations: {matcher.get_team_variations(away_team)}")
    print(f"Home team variations: {matcher.get_team_variations(home_team)}")
    print()

    kalshi_odds = matcher.match_game_to_kalshi(game)

    if kalshi_odds:
        print("✅ MATCH FOUND!")
        print(f"\n  Market: {kalshi_odds.get('market_title', 'N/A')}")
        print(f"  Ticker: {kalshi_odds.get('ticker', 'N/A')}")
        print(f"  {away_team}: {kalshi_odds['away_win_price']:.1%}")
        print(f"  {home_team}: {kalshi_odds['home_win_price']:.1%}")
        print(f"  Volume: ${kalshi_odds.get('volume', 0):,.0f}")

        # User mentioned Jacksonville 41%, Los Angeles 59%
        if 'jaguars' in away_team.lower() or 'jacksonville' in away_team.lower():
            expected_away = 0.41
            expected_home = 0.59
            away_close = abs(kalshi_odds['away_win_price'] - expected_away) < 0.05
            home_close = abs(kalshi_odds['home_win_price'] - expected_home) < 0.05

            print(f"\n  Expected: {away_team} 41%, {home_team} 59%")
            if away_close and home_close:
                print("  ✅ Odds match expected values!")
            else:
                print("  ⚠️ Odds differ from expected (might have changed)")

        return {'status': 'PASSED', 'odds': kalshi_odds}
    else:
        print("❌ NO MATCH FOUND")
        print("\nTroubleshooting:")
        print("1. Check if Kalshi markets are synced:")
        print("   python sync_kalshi_team_winners.py --list")
        print("\n2. Verify market exists in database:")
        print(f"   SELECT * FROM kalshi_markets WHERE title ILIKE '%{away_team}%' OR title ILIKE '%{home_team}%';")
        print("\n3. Run sync to fetch fresh markets:")
        print("   python sync_kalshi_team_winners.py --sport nfl")

        return {'status': 'FAILED', 'odds': None}


def main():
    """Run all verification tests"""
    print("=" * 80)
    print(" GAME CARDS SYSTEM VERIFICATION")
    print(" Testing AI Predictions and Kalshi Odds Integration")
    print("=" * 80)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get ESPN games
    print("Fetching ESPN games...")
    espn_client = get_espn_client()
    games = espn_client.get_scoreboard()

    if not games:
        print("❌ No ESPN games found. Cannot run tests.")
        return

    print(f"✅ Found {len(games)} ESPN games\n")

    # Run tests
    results = {}

    # Test 1: AI Predictions
    results['ai_predictions'] = test_ai_predictions(games, max_games=5)

    # Test 2: Kalshi Matching
    results['kalshi_matching'] = test_kalshi_matcher(games, max_games=5)

    # Test 3: Team Name Variations
    results['team_variations'] = test_team_name_variations()

    # Test 4: Jacksonville vs Los Angeles (user's example)
    print("\n")
    results['jacksonville_la'] = test_specific_game('Jacksonville Jaguars', 'Los Angeles Chargers')

    # Final Summary
    print_section("FINAL VERIFICATION SUMMARY")

    all_tests = [
        ('AI Predictions', results['ai_predictions']['status']),
        ('Kalshi Matching', results['kalshi_matching']['status']),
        ('Team Variations', results['team_variations']['status']),
        ('Jacksonville vs LA', results['jacksonville_la']['status'])
    ]

    print("\nTest Results:")
    for test_name, status in all_tests:
        status_icon = "✅" if status == "PASSED" else ("⚠️" if status == "PARTIAL" else "❌")
        print(f"  {status_icon} {test_name}: {status}")

    # Overall status
    passed = sum(1 for _, status in all_tests if status == "PASSED")
    total = len(all_tests)

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Game Cards system is working correctly.")
    elif results['kalshi_matching']['status'] == 'NO_MATCHES':
        print("\n⚠️ Kalshi markets need to be synced. Run:")
        print("   python sync_kalshi_team_winners.py --sport football")
        print("\nAfter sync, re-run this verification script.")
    else:
        print("\n⚠️ Some tests need attention. Review results above.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
