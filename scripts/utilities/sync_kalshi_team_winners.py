"""
Sync Kalshi Team Winner Markets for NFL and NCAA Football
Fetches simple team vs team winner markets (not parlays or player props)
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(override=True)

from src.kalshi_integration import KalshiIntegration
from src.kalshi_db_manager import KalshiDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeamWinnerMarketSync:
    """Sync only team vs team winner markets from Kalshi"""

    def __init__(self):
        self.client = KalshiIntegration()
        self.db = KalshiDBManager()
        self.synced_count = 0
        self.skipped_count = 0

    def is_team_winner_market(self, market: Dict) -> bool:
        """
        Check if market is a simple team vs team winner market

        Examples of VALID markets:
        - "Will Jacksonville beat Los Angeles?"
        - "NFL: Jaguars to beat Chargers"
        - "Jacksonville to win vs Los Angeles"

        Examples of INVALID markets (skip these):
        - "yes Baltimore,yes Carolina,yes Denver" (combo bet)
        - "Josh Allen 250+ yards" (player prop)
        - "Over 47.5 points" (totals)
        """
        title = market.get('title', '').lower()
        ticker = market.get('ticker', '').lower()

        # Skip combo/parlay markets (contain commas)
        if ',' in title or ',' in ticker:
            return False

        # Skip player props (contain player names or specific stats)
        player_prop_keywords = [
            'yards', 'touchdowns', 'receptions', 'completions',
            'passing', 'rushing', 'receiving', 'tackles',
            'interceptions', 'sacks', 'fantasy', 'anytime'
        ]
        if any(keyword in title for keyword in player_prop_keywords):
            return False

        # Skip totals/spreads
        totals_keywords = ['over', 'under', 'spread', 'points', '+', '-']
        if any(keyword in title for keyword in totals_keywords):
            # Exception: "to beat" is ok even if it has + or -
            if 'to beat' not in title and 'to win' not in title:
                return False

        # Valid team winner market keywords
        team_winner_keywords = [
            'to beat', 'to win', 'will win', 'winner',
            'beats', 'wins', 'defeat'
        ]

        return any(keyword in title for keyword in team_winner_keywords)

    def categorize_market(self, market: Dict) -> str:
        """Determine if market is NFL or CFB"""
        title = market.get('title', '').lower()
        ticker = market.get('ticker', '').lower()

        # NFL keywords
        nfl_keywords = ['nfl', 'patriots', 'bills', 'dolphins', 'jets',
                       'ravens', 'bengals', 'browns', 'steelers',
                       'texans', 'colts', 'jaguars', 'titans',
                       'broncos', 'chiefs', 'raiders', 'chargers',
                       'cowboys', 'giants', 'eagles', 'commanders',
                       'bears', 'lions', 'packers', 'vikings',
                       'falcons', 'panthers', 'saints', 'buccaneers',
                       'cardinals', '49ers', 'rams', 'seahawks']

        if any(keyword in title or keyword in ticker for keyword in nfl_keywords):
            return 'nfl'

        # CFB keywords (common college teams)
        cfb_keywords = ['cfb', 'college', 'ncaa', 'alabama', 'ohio state',
                       'georgia', 'michigan', 'clemson', 'usc', 'texas',
                       'oklahoma', 'notre dame', 'penn state', 'florida']

        if any(keyword in title or keyword in ticker for keyword in cfb_keywords):
            return 'cfb'

        return 'winner'  # Generic winner market

    def sync_markets(self, sport: str = 'all') -> Dict:
        """
        Sync team winner markets from Kalshi

        Args:
            sport: 'nfl', 'cfb', or 'all'

        Returns:
            Dict with sync statistics
        """
        logger.info(f"Starting Kalshi team winner market sync for {sport.upper()}")

        # Login to Kalshi
        if not self.client.login():
            logger.error("Failed to login to Kalshi. Check credentials in .env")
            logger.error("Required: KALSHI_EMAIL and KALSHI_PASSWORD")
            return {
                'success': False,
                'error': 'Authentication failed',
                'synced': 0,
                'skipped': 0
            }

        # Fetch all open markets
        logger.info("Fetching markets from Kalshi API...")
        all_markets = self.client.get_all_markets(status='open', limit=1000)

        if not all_markets:
            logger.warning("No markets returned from Kalshi API")
            return {
                'success': True,
                'synced': 0,
                'skipped': 0,
                'total_markets': 0
            }

        logger.info(f"Retrieved {len(all_markets)} total markets from Kalshi")

        # Filter for team winner markets
        team_winner_markets = []
        for market in all_markets:
            if self.is_team_winner_market(market):
                market_type = self.categorize_market(market)

                # Filter by requested sport
                if sport == 'all' or sport == market_type or \
                   (sport == 'football' and market_type in ['nfl', 'cfb']):
                    market['market_type'] = market_type
                    team_winner_markets.append(market)
                    self.synced_count += 1
                else:
                    self.skipped_count += 1
            else:
                self.skipped_count += 1

        logger.info(f"Found {len(team_winner_markets)} team winner markets")
        logger.info(f"Skipped {self.skipped_count} non-winner markets (combos/props/totals)")

        # Store in database
        if team_winner_markets:
            logger.info("Storing markets in database...")
            stored = self.db.store_markets(team_winner_markets)
            logger.info(f"Stored {stored} team winner markets in database")

        # Get latest prices for all markets
        logger.info("Fetching latest prices...")
        price_updates = 0
        for market in team_winner_markets[:50]:  # Limit to first 50 to avoid rate limits
            try:
                ticker = market.get('ticker')
                if ticker:
                    prices = self.client.get_market_orderbook(ticker)
                    if prices:
                        self.db.store_market_prices(ticker, prices)
                        price_updates += 1
            except Exception as e:
                logger.debug(f"Could not fetch prices for {ticker}: {e}")

        logger.info(f"Updated prices for {price_updates} markets")

        return {
            'success': True,
            'total_markets_fetched': len(all_markets),
            'team_winner_markets': len(team_winner_markets),
            'synced': self.synced_count,
            'skipped': self.skipped_count,
            'price_updates': price_updates
        }

    def list_recent_markets(self, limit: int = 10):
        """Display recently synced team winner markets"""
        conn = self.db.get_connection()
        cur = conn.cursor()

        try:
            query = """
            SELECT
                ticker,
                title,
                market_type,
                yes_price,
                no_price,
                volume,
                close_time,
                updated_at
            FROM kalshi_markets
            WHERE market_type IN ('nfl', 'cfb', 'winner')
              AND yes_price IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT %s
            """

            cur.execute(query, (limit,))
            markets = cur.fetchall()

            if markets:
                print("\n" + "=" * 80)
                print("RECENTLY SYNCED TEAM WINNER MARKETS")
                print("=" * 80)

                for market in markets:
                    ticker, title, market_type, yes_price, no_price, volume, close_time, updated_at = market

                    print(f"\n{title}")
                    print(f"  Ticker: {ticker}")
                    print(f"  Type: {market_type.upper()}")
                    print(f"  Odds: Yes {yes_price:.1%} / No {no_price:.1%}")
                    print(f"  Volume: ${volume:,.0f}")
                    print(f"  Closes: {close_time}")
                    print(f"  Updated: {updated_at}")

                print("\n" + "=" * 80)
            else:
                print("\nNo team winner markets found in database")
                print("Make sure KALSHI_EMAIL and KALSHI_PASSWORD are set in .env")

        finally:
            cur.close()
            conn.close()


def main():
    """Main sync function"""
    import argparse

    parser = argparse.ArgumentParser(description='Sync Kalshi Team Winner Markets')
    parser.add_argument('--sport', choices=['all', 'nfl', 'cfb', 'football'],
                       default='all', help='Which sport to sync')
    parser.add_argument('--list', action='store_true',
                       help='List recently synced markets')

    args = parser.parse_args()

    syncer = TeamWinnerMarketSync()

    if args.list:
        syncer.list_recent_markets(limit=20)
        return

    # Run sync
    print("=" * 80)
    print("KALSHI TEAM WINNER MARKET SYNC")
    print("=" * 80)
    print(f"\nSyncing {args.sport.upper()} team winner markets...")
    print("This will fetch simple 'Team A beats Team B' markets")
    print("Skipping combo bets, player props, and totals\n")

    result = syncer.sync_markets(sport=args.sport)

    print("\n" + "=" * 80)
    print("SYNC RESULTS")
    print("=" * 80)

    if result['success']:
        print(f"✅ Success!")
        print(f"\nTotal markets fetched: {result.get('total_markets_fetched', 0)}")
        print(f"Team winner markets: {result.get('team_winner_markets', 0)}")
        print(f"Synced to database: {result.get('synced', 0)}")
        print(f"Skipped (combos/props): {result.get('skipped', 0)}")
        print(f"Price updates: {result.get('price_updates', 0)}")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("1. Run: python sync_kalshi_team_winners.py --list")
        print("   to view synced markets")
        print("\n2. Open Sports Game Cards page in dashboard")
        print("   Kalshi odds should now appear on game cards!")
        print("\n3. Test Jacksonville vs LA example (should show 41% vs 59%)")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        print("\nTroubleshooting:")
        print("1. Check .env file has KALSHI_EMAIL and KALSHI_PASSWORD")
        print("2. Verify credentials are correct")
        print("3. Check Kalshi API status")

    print("=" * 80)


if __name__ == "__main__":
    main()
