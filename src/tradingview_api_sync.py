"""TradingView API Sync using session cookies - Based on GitHub project"""

import requests
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingViewAPISync:
    """Sync TradingView watchlists using API with session cookies"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

        # These will need to be obtained from browser
        self.session_id = os.getenv('TRADINGVIEW_SESSION_ID', '')
        self.base_url = 'https://www.tradingview.com'

        # Initialize database tables
        self.init_database()

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def init_database(self) -> None:
        """Initialize database tables for TradingView watchlists"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            # Main watchlists table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tv_watchlists_api (
                    id SERIAL PRIMARY KEY,
                    watchlist_id VARCHAR(50) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    color VARCHAR(50),
                    symbols TEXT[],
                    symbol_count INTEGER DEFAULT 0,
                    last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Individual symbols table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tv_symbols_api (
                    id SERIAL PRIMARY KEY,
                    watchlist_id VARCHAR(50) REFERENCES tv_watchlists_api(watchlist_id) ON DELETE CASCADE,
                    symbol VARCHAR(50) NOT NULL,
                    exchange VARCHAR(50),
                    full_symbol VARCHAR(100),
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(watchlist_id, symbol)
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tv_symbols_api_symbol
                ON tv_symbols_api(symbol)
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tv_symbols_api_watchlist
                ON tv_symbols_api(watchlist_id)
            """)

            conn.commit()
            logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def get_headers(self) -> None:
        """Get headers for API requests"""
        return {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'accept': 'application/json',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'en-US,en;q=0.9',
            'cookie': f'sessionid={self.session_id}',
            'origin': self.base_url,
            'referer': f'{self.base_url}/watchlists/',
            'x-requested-with': 'XMLHttpRequest',
        }

    def refresh_session_id(self) -> bool:
        """Attempt to refresh session ID using interactive script"""
        logger.warning("Session ID appears to be invalid or expired")
        logger.info("Attempting to get fresh session ID...")

        try:
            # Import and run the interactive session getter
            import subprocess
            import sys

            logger.info("\n" + "="*60)
            logger.info("Running interactive session ID retrieval...")
            logger.info("A browser window will open - please complete login/2FA")
            logger.info("="*60 + "\n")

            # Run the interactive script with --auto flag
            result = subprocess.run(
                [sys.executable, 'src/get_session_interactive.py', '--auto'],
                timeout=180,  # 3 minutes timeout
                capture_output=False
            )

            if result.returncode == 0:
                # Reload environment variables
                from dotenv import load_dotenv
                load_dotenv(override=True)
                new_session_id = os.getenv('TRADINGVIEW_SESSION_ID', '')

                if new_session_id and new_session_id != self.session_id:
                    self.session_id = new_session_id
                    logger.info(f"Session ID updated: {self.session_id[:8]}...")
                    return True

            logger.warning("Could not refresh session ID automatically")
            return False

        except Exception as e:
            logger.error(f"Error refreshing session ID: {e}")
            return False

    def get_all_watchlists(self) -> List[Dict]:
        """Get list of all watchlists (without symbols)"""
        if not self.session_id:
            logger.error("No session ID configured. Please set TRADINGVIEW_SESSION_ID")
            return []

        try:
            # Get custom watchlists
            custom_url = f'{self.base_url}/api/v1/symbols_list/custom/'
            response = requests.get(custom_url, headers=self.get_headers())

            watchlists = []

            if response.status_code == 200:
                custom_lists = response.json()
                if isinstance(custom_lists, list):
                    for wl in custom_lists:
                        watchlists.append({
                            'id': wl.get('id', ''),
                            'name': wl.get('name', ''),
                            'type': 'custom'
                        })
            elif response.status_code in [401, 403]:
                logger.error(f"Authentication failed (status {response.status_code})")
                # Try to refresh session ID
                if self.refresh_session_id():
                    logger.info("Retrying with new session ID...")
                    return self.get_all_watchlists()  # Recursive retry
                return []

            # Get colored watchlists (built-in)
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
            for color in colors:
                color_url = f'{self.base_url}/api/v1/symbols_list/colored/{color}'
                try:
                    response = requests.get(color_url, headers=self.get_headers())
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('symbols'):
                            watchlists.append({
                                'id': color,
                                'name': f'{color.capitalize()} List',
                                'type': 'colored',
                                'color': color
                            })
                except:
                    continue

            logger.info(f"Found {len(watchlists)} watchlists")
            return watchlists

        except Exception as e:
            logger.error(f"Error getting watchlists: {e}")
            return []

    def get_watchlist_symbols(self, watchlist_id: str, watchlist_type: str = 'custom') -> List[str]:
        """Get symbols from a specific watchlist"""
        if not self.session_id:
            return []

        try:
            if watchlist_type == 'colored':
                url = f'{self.base_url}/api/v1/symbols_list/colored/{watchlist_id}'
            else:
                url = f'{self.base_url}/api/v1/symbols_list/custom/{watchlist_id}'

            response = requests.get(url, headers=self.get_headers())

            if response.status_code == 200:
                data = response.json()
                symbols = data.get('symbols', [])
                logger.info(f"Fetched {len(symbols)} symbols from watchlist {watchlist_id}")
                return symbols
            else:
                logger.warning(f"Failed to get symbols for watchlist {watchlist_id}: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error getting symbols for watchlist {watchlist_id}: {e}")
            return []

    def update_watchlist_symbols(self, watchlist_id: str, symbols: List[str],
                                watchlist_type: str = 'custom', replace: bool = False) -> bool:
        """Update symbols in a watchlist"""
        if not self.session_id:
            return False

        try:
            if watchlist_type == 'colored':
                base_url = f'{self.base_url}/api/v1/symbols_list/colored/{watchlist_id}'
            else:
                base_url = f'{self.base_url}/api/v1/symbols_list/custom/{watchlist_id}'

            if replace:
                url = f'{base_url}/replace/?unsafe=true'
            else:
                url = f'{base_url}/append/'

            response = requests.post(url, json=symbols, headers=self.get_headers())

            if response.status_code == 200:
                logger.info(f"Successfully updated watchlist {watchlist_id}")
                return True
            else:
                logger.warning(f"Failed to update watchlist {watchlist_id}: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error updating watchlist {watchlist_id}: {e}")
            return False

    def sync_to_database(self) -> Dict[str, List[str]]:
        """Sync all watchlists to database"""
        if not self.session_id:
            logger.error("Cannot sync - no session ID configured")
            logger.info("\nAttempting to get session ID automatically...")

            # Try to get session ID interactively
            if self.refresh_session_id():
                logger.info("Session ID obtained, retrying sync...")
                # Recursive retry with new session
                return self.sync_to_database()
            else:
                logger.error("\nManual steps to get session ID:")
                logger.info("1. Run: python src/get_session_interactive.py")
                logger.info("2. OR manually get from browser DevTools")
                return {}

        conn = self.get_connection()
        cur = conn.cursor()

        all_watchlists = {}

        try:
            # Get all watchlists
            watchlists = self.get_all_watchlists()

            for wl in watchlists:
                watchlist_id = str(wl['id'])  # Convert to string to handle both int IDs and color names
                name = wl['name']
                wl_type = wl.get('type', 'custom')
                color = wl.get('color', '')

                # Get symbols for this watchlist
                symbols = self.get_watchlist_symbols(watchlist_id, wl_type)

                if symbols:
                    # Save to database
                    cur.execute("""
                        INSERT INTO tv_watchlists_api (watchlist_id, name, color, symbols, symbol_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (watchlist_id)
                        DO UPDATE SET
                            name = EXCLUDED.name,
                            symbols = EXCLUDED.symbols,
                            symbol_count = EXCLUDED.symbol_count,
                            last_synced = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                    """, (watchlist_id, name, color, symbols, len(symbols)))

                    # Clear and re-insert symbols
                    cur.execute("""
                        DELETE FROM tv_symbols_api WHERE watchlist_id = %s
                    """, (watchlist_id,))

                    for symbol in symbols:
                        # Parse exchange if present (e.g., "NASDAQ:NVDA")
                        if ':' in symbol:
                            exchange, sym = symbol.split(':', 1)
                            full_symbol = symbol
                        else:
                            exchange = ''
                            sym = symbol
                            full_symbol = symbol

                        cur.execute("""
                            INSERT INTO tv_symbols_api (watchlist_id, symbol, exchange, full_symbol)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (watchlist_id, symbol) DO NOTHING
                        """, (watchlist_id, sym, exchange, full_symbol))

                    all_watchlists[name] = symbols
                    logger.info(f"Synced '{name}' with {len(symbols)} symbols")

            conn.commit()
            logger.info(f"Successfully synced {len(all_watchlists)} watchlists to database")

        except Exception as e:
            logger.error(f"Database sync failed: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

        return all_watchlists

    def get_watchlists_from_db(self) -> Dict[str, List[str]]:
        """Get all watchlists from database"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT name, symbols FROM tv_watchlists_api
                ORDER BY name
            """)

            watchlists = {}
            for row in cur.fetchall():
                watchlists[row['name']] = row['symbols'] or []

            return watchlists

        except Exception as e:
            logger.error(f"Error fetching from database: {e}")
            return {}
        finally:
            cur.close()
            conn.close()

    def find_watchlist_with_symbol(self, symbol: str) -> List[str]:
        """Find which watchlists contain a specific symbol"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT DISTINCT w.name
                FROM tv_watchlists_api w
                JOIN tv_symbols_api s ON w.watchlist_id = s.watchlist_id
                WHERE s.symbol = %s OR s.full_symbol = %s
            """, (symbol.upper(), symbol.upper()))

            watchlists = [row[0] for row in cur.fetchall()]
            return watchlists

        except Exception as e:
            logger.error(f"Error finding watchlists: {e}")
            return []
        finally:
            cur.close()
            conn.close()


if __name__ == "__main__":
    # Test the sync
    syncer = TradingViewAPISync()

    # Check if session ID is configured
    if not syncer.session_id:
        print("\n" + "="*60)
        print("SETUP REQUIRED: TradingView Session ID")
        print("="*60)
        print("\nTo sync your TradingView watchlists:")
        print("1. Open TradingView.com in Chrome")
        print("2. Login to your account")
        print("3. Press F12 to open DevTools")
        print("4. Go to Application tab > Cookies > tradingview.com")
        print("5. Find 'sessionid' cookie (32 character string)")
        print("6. Add to .env file:")
        print("   TRADINGVIEW_SESSION_ID=your_session_id_here")
        print("\nExample:")
        print("   TRADINGVIEW_SESSION_ID=abcd1234efgh5678ijkl9012mnop3456")
        print("="*60)
    else:
        print(f"Session ID configured: {syncer.session_id[:8]}...")

        # Sync watchlists
        watchlists = syncer.sync_to_database()

        if watchlists:
            print("\n" + "="*50)
            print("Successfully synced watchlists:")
            print("="*50)
            for name, symbols in watchlists.items():
                print(f"\n[Watchlist] {name}: {len(symbols)} symbols")
                print(f"   {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

            # Test finding NVDA
            nvda_lists = syncer.find_watchlist_with_symbol('NVDA')
            if nvda_lists:
                print(f"\n[OK] NVDA found in: {', '.join(nvda_lists)}")
        else:
            print("\nNo watchlists found. Check your session ID.")