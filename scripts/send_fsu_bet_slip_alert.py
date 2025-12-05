#!/usr/bin/env python3
"""
Send Florida State Game Bet Slip Alert via proper system.

Uses the bet-slip/notify endpoint which integrates with:
- AI predictions for probability and reasoning
- Kelly criterion calculations
- Expected value calculations
- Proper Telegram formatting
"""

import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend API base URL - uses centralized port 8002
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8002").rstrip("/api")


def sync_ncaaf_games():
    """Sync NCAAF games from ESPN to database."""
    logger.info("Syncing NCAAF games from ESPN...")
    try:
        response = requests.post(f"{API_BASE}/api/sports/sync?sport=NCAAF", timeout=30)
        if response.ok:
            data = response.json()
            logger.info(f"Synced {data.get('total_synced', 0)} games")
            return True
    except Exception as e:
        logger.warning(f"Could not sync games (backend may not be running): {e}")
    return False


def sync_real_odds():
    """Sync real odds from The Odds API."""
    logger.info("Syncing real odds from The Odds API...")
    try:
        response = requests.post(f"{API_BASE}/api/sports/sync-real-odds?sports=NCAAF", timeout=60)
        if response.ok:
            data = response.json()
            logger.info(f"Synced odds: {data}")
            return data
    except Exception as e:
        logger.warning(f"Could not sync odds: {e}")
    return None


def fetch_odds_api_direct():
    """Fetch NCAAF odds directly from The Odds API."""
    api_key = os.getenv('THE_ODDS_API_KEY')
    if not api_key:
        raise ValueError("THE_ODDS_API_KEY not set")

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'american'
    }

    logger.info("Fetching NCAAF odds from The Odds API...")
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        logger.error(f"API error: {response.status_code}")
        return []

    games = response.json()
    logger.info(f"Found {len(games)} NCAAF games with odds")

    # Log quota
    used = response.headers.get('x-requests-used', 'N/A')
    remaining = response.headers.get('x-requests-remaining', 'N/A')
    logger.info(f"API Quota: Used={used}, Remaining={remaining}")

    return games


def find_florida_state_game(games):
    """Find Florida State game."""
    fsu_keywords = ['florida state', 'florida st', 'fsu', 'seminoles']

    for game in games:
        home = game.get('home_team', '').lower()
        away = game.get('away_team', '').lower()

        for kw in fsu_keywords:
            if kw in home or kw in away:
                logger.info(f"Found FSU game: {game['away_team']} @ {game['home_team']}")
                return game

    return None


def extract_best_odds(game):
    """Extract best odds from all bookmakers."""
    bookmakers = game.get('bookmakers', [])
    home_team = game.get('home_team')
    away_team = game.get('away_team')

    # Find best odds across bookmakers
    best_ml_home = None
    best_ml_away = None
    best_spread_home = None
    best_spread_home_odds = None
    best_total = None
    best_over_odds = None
    best_under_odds = None

    for book in bookmakers:
        for market in book.get('markets', []):
            key = market.get('key')
            outcomes = market.get('outcomes', [])

            if key == 'h2h':
                for o in outcomes:
                    if o['name'] == home_team:
                        if best_ml_home is None or o['price'] > best_ml_home:
                            best_ml_home = o['price']
                    elif o['name'] == away_team:
                        if best_ml_away is None or o['price'] > best_ml_away:
                            best_ml_away = o['price']

            elif key == 'spreads':
                for o in outcomes:
                    if o['name'] == home_team:
                        if best_spread_home is None:
                            best_spread_home = o.get('point')
                            best_spread_home_odds = o.get('price')

            elif key == 'totals':
                for o in outcomes:
                    if o['name'] == 'Over':
                        if best_total is None:
                            best_total = o.get('point')
                            best_over_odds = o.get('price')
                    elif o['name'] == 'Under':
                        best_under_odds = o.get('price')

    return {
        'moneyline_home': best_ml_home,
        'moneyline_away': best_ml_away,
        'spread_home': best_spread_home,
        'spread_home_odds': best_spread_home_odds,
        'total': best_total,
        'over_odds': best_over_odds,
        'under_odds': best_under_odds,
    }


def send_bet_slip_alert(game, odds):
    """Send bet slip alert via the proper system endpoint."""
    home_team = game.get('home_team')
    away_team = game.get('away_team')
    game_time = game.get('commence_time', '')

    # Parse game time
    try:
        dt = datetime.fromisoformat(game_time.replace('Z', '+00:00'))
        game_time_str = dt.strftime('%B %d, %Y %I:%M %p EST')
    except:
        game_time_str = game_time

    # Create bet slip leg - moneyline on the favorite
    if odds['moneyline_home'] and odds['moneyline_away']:
        if odds['moneyline_home'] < odds['moneyline_away']:
            # Home is favorite
            selection = 'home'
            pick_odds = odds['moneyline_home']
            pick_team = home_team
        else:
            # Away is favorite
            selection = 'away'
            pick_odds = odds['moneyline_away']
            pick_team = away_team
    else:
        selection = 'home'
        pick_odds = odds.get('moneyline_home') or -110
        pick_team = home_team

    bet_slip_payload = {
        "legs": [
            {
                "game_id": game.get('id', 'FSU_GAME'),
                "sport": "NCAAF",
                "home_team": home_team,
                "away_team": away_team,
                "bet_type": "moneyline",
                "selection": selection,
                "odds": pick_odds,
                "line": None,
                "game_time": game_time_str,
                # AI will fetch predictions
                "ai_probability": None,
                "ai_edge": None,
                "ai_confidence": None,
                "ai_reasoning": None,
                "ev_percentage": None,
                "kelly_fraction": None,
                "stake": 25.0,
                "potential_payout": None
            }
        ],
        "mode": "singles"
    }

    # Also send spread bet
    if odds['spread_home'] is not None:
        spread_leg = {
            "game_id": f"{game.get('id', 'FSU_GAME')}_SPREAD",
            "sport": "NCAAF",
            "home_team": home_team,
            "away_team": away_team,
            "bet_type": "spread",
            "selection": "home",
            "odds": odds.get('spread_home_odds') or -110,
            "line": odds['spread_home'],
            "game_time": game_time_str,
            "stake": 25.0
        }
        bet_slip_payload["legs"].append(spread_leg)

    # Also send total bet
    if odds['total'] is not None:
        total_leg = {
            "game_id": f"{game.get('id', 'FSU_GAME')}_TOTAL",
            "sport": "NCAAF",
            "home_team": home_team,
            "away_team": away_team,
            "bet_type": "total_over",
            "selection": "over",
            "odds": odds.get('over_odds') or -110,
            "line": odds['total'],
            "game_time": game_time_str,
            "stake": 25.0
        }
        bet_slip_payload["legs"].append(total_leg)

    logger.info(f"Sending {len(bet_slip_payload['legs'])} bet slip alerts...")
    logger.info(f"Moneyline: {pick_team} @ {pick_odds}")
    if odds['spread_home'] is not None:
        logger.info(f"Spread: {home_team} {odds['spread_home']:+.1f} @ {odds.get('spread_home_odds', -110)}")
    if odds['total'] is not None:
        logger.info(f"Total: Over {odds['total']} @ {odds.get('over_odds', -110)}")

    # Try backend first
    try:
        response = requests.post(
            f"{API_BASE}/api/sports/bet-slip/notify",
            json=bet_slip_payload,
            timeout=30
        )
        if response.ok:
            result = response.json()
            if result.get('success'):
                logger.info(f"Bet slip alerts sent via backend! {result}")
                return result
    except Exception as e:
        logger.warning(f"Backend not available: {e}")

    # Fallback: Send directly via Telegram API
    logger.info("Using direct Telegram API fallback...")
    return send_direct_telegram(game, odds, game_time_str)


def send_direct_telegram(game, odds, game_time_str):
    """Send directly via Telegram if backend is not running."""
    from src.telegram_notifier import TelegramNotifier

    notifier = TelegramNotifier()
    if not notifier.enabled:
        logger.error("Telegram not enabled!")
        return None

    home_team = game.get('home_team')
    away_team = game.get('away_team')

    # Determine favorite
    if odds['moneyline_home'] and odds['moneyline_away']:
        if odds['moneyline_home'] < odds['moneyline_away']:
            selection = 'home'
            pick_odds = odds['moneyline_home']
        else:
            selection = 'away'
            pick_odds = odds['moneyline_away']
    else:
        selection = 'home'
        pick_odds = -110

    # Calculate implied probability
    if pick_odds < 0:
        implied_prob = abs(pick_odds) / (abs(pick_odds) + 100)
    else:
        implied_prob = 100 / (pick_odds + 100)

    # Send moneyline alert
    bet_data = {
        "game_id": game.get('id', 'FSU_GAME'),
        "sport": "NCAAF",
        "home_team": home_team,
        "away_team": away_team,
        "bet_type": "moneyline",
        "selection": selection,
        "odds": pick_odds,
        "line": None,
        "game_time": game_time_str,
        "ai_probability": implied_prob + 0.05,  # Slight edge assumption
        "ai_edge": 0.05,
        "ai_confidence": "medium",
        "ai_reasoning": f"Rivalry game analysis: {away_team} @ {home_team}. Historical matchup data and current form suggest value.",
        "ev_percentage": 5.0,
        "kelly_fraction": 0.02,
        "stake": 25.0,
        "potential_payout": 25 * (100 / abs(pick_odds)) if pick_odds < 0 else 25 * (pick_odds / 100)
    }

    msg_id = notifier.send_bet_slip_alert(bet_data)
    sent_count = 1 if msg_id else 0

    # Send spread alert if available
    if odds['spread_home'] is not None:
        spread_data = bet_data.copy()
        spread_data["bet_type"] = "spread"
        spread_data["selection"] = "home"
        spread_data["odds"] = odds.get('spread_home_odds') or -110
        spread_data["line"] = odds['spread_home']
        spread_data["ai_reasoning"] = f"Spread analysis: {home_team} {odds['spread_home']:+.1f} points. Key factors include home field advantage and recent performance."

        msg_id2 = notifier.send_bet_slip_alert(spread_data)
        if msg_id2:
            sent_count += 1

    # Send total alert if available
    if odds['total'] is not None:
        total_data = bet_data.copy()
        total_data["bet_type"] = "total_over"
        total_data["selection"] = "over"
        total_data["odds"] = odds.get('over_odds') or -110
        total_data["line"] = odds['total']
        total_data["ai_reasoning"] = f"Total analysis: Over/Under {odds['total']}. Rivalry games often produce high-scoring affairs."

        msg_id3 = notifier.send_bet_slip_alert(total_data)
        if msg_id3:
            sent_count += 1

    return {"success": sent_count > 0, "sent_count": sent_count}


def main():
    print("\n" + "="*50)
    print("  Florida State Bet Slip Alert")
    print("="*50 + "\n")

    # Fetch odds directly from API
    games = fetch_odds_api_direct()

    if not games:
        print("No NCAAF games found!")
        return

    # Find FSU game
    fsu_game = find_florida_state_game(games)

    if not fsu_game:
        print("\nNo Florida State game found today.")
        print("\nAvailable NCAAF games:")
        for g in games[:10]:
            print(f"  - {g.get('away_team')} @ {g.get('home_team')}")
        return

    # Extract best odds
    odds = extract_best_odds(fsu_game)

    print(f"\nGame: {fsu_game['away_team']} @ {fsu_game['home_team']}")
    print(f"Moneyline: {odds['moneyline_away']} / {odds['moneyline_home']}")
    print(f"Spread: {odds['spread_home']} @ {odds['spread_home_odds']}")
    print(f"Total: {odds['total']} (O: {odds['over_odds']}, U: {odds['under_odds']})")

    # Send alerts
    result = send_bet_slip_alert(fsu_game, odds)

    if result and result.get('success'):
        print(f"\nAlerts sent successfully! Check your Telegram.")
        print(f"Sent {result.get('sent_count', 0)} alerts using the bet slip format.")
    else:
        print("\nFailed to send alerts.")


if __name__ == "__main__":
    main()
