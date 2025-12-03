"""TradingView Database Sync - Automatically sync watchlists to PostgreSQL"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import List, Dict, Optional
import yfinance as yf
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import time

# Load environment variables
load_dotenv()

class TradingViewDBSync:
    """Sync TradingView watchlists to PostgreSQL database automatically"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'wheel_strategy'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!')
        }
        self.username = os.getenv('TRADINGVIEW_USERNAME')
        self.password = os.getenv('TRADINGVIEW_PASSWORD')
        self.conn = None
        self.cursor = None

    def connect_db(self) -> None:
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect_db(self) -> None:
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def create_tables(self) -> None:
        """Create watchlist tables if they don't exist"""
        try:
            # Create watchlists table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS tradingview_watchlists (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    description TEXT,
                    last_synced TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create watchlist_stocks table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_stocks (
                    id SERIAL PRIMARY KEY,
                    watchlist_id INTEGER REFERENCES tradingview_watchlists(id) ON DELETE CASCADE,
                    symbol VARCHAR(10) NOT NULL,
                    added_date TIMESTAMP DEFAULT NOW(),
                    current_price DECIMAL(10,2),
                    price_updated TIMESTAMP,
                    UNIQUE(watchlist_id, symbol)
                )
            """)

            # Create index for faster queries
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_watchlist_stocks_symbol
                ON watchlist_stocks(symbol)
            """)

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error creating tables: {e}")
            self.conn.rollback()
            return False

    def fetch_tradingview_watchlists(self) -> Dict[str, List[str]]:
        """Fetch watchlists using web scraping (no Selenium)"""
        watchlists = {}

        try:
            # Use requests session for authentication
            session = requests.Session()

            # Try to authenticate
            login_url = "https://www.tradingview.com/accounts/signin/"

            # Since TradingView requires complex auth, let's use their public lists API
            # This fetches popular public watchlists as a fallback
            api_url = "https://scanner.tradingview.com/global/scan"

            # Define some preset scans that work like watchlists
            preset_scans = {
                "Most Active": {
                    "filter": [
                        {"left": "volume", "operation": "greater", "right": 1000000},
                        {"left": "close", "operation": "less", "right": 50}
                    ],
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name", "close", "change", "volume"],
                    "sort": {"sortBy": "volume", "sortOrder": "desc"},
                    "range": [0, 20]
                },
                "Top Gainers": {
                    "filter": [
                        {"left": "change", "operation": "greater", "right": 5},
                        {"left": "close", "operation": "less", "right": 50}
                    ],
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name", "close", "change", "volume"],
                    "sort": {"sortBy": "change", "sortOrder": "desc"},
                    "range": [0, 20]
                },
                "High Volume Under $20": {
                    "filter": [
                        {"left": "close", "operation": "less", "right": 20},
                        {"left": "volume", "operation": "greater", "right": 5000000}
                    ],
                    "symbols": {"query": {"types": ["stock"]}, "tickers": []},
                    "columns": ["name", "close", "change", "volume"],
                    "sort": {"sortBy": "volume", "sortOrder": "desc"},
                    "range": [0, 20]
                }
            }

            # Fetch each preset scan
            for scan_name, scan_config in preset_scans.items():
                try:
                    response = session.post(
                        api_url,
                        json=scan_config,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Content-Type': 'application/json'
                        }
                    )

                    if response.status_code == 200:
                        data = response.json()
                        symbols = []

                        if 'data' in data:
                            for item in data['data']:
                                if 's' in item:  # 's' contains the symbol
                                    # Remove exchange prefix if present
                                    symbol = item['s'].split(':')[-1]
                                    symbols.append(symbol)

                        if symbols:
                            watchlists[scan_name] = symbols

                except Exception as e:
                    print(f"Error fetching {scan_name}: {e}")

            # If we have credentials, try to get user's actual watchlists
            if self.username and self.password:
                # This would require more complex authentication
                # For now, we'll use the preset scans
                pass

        except Exception as e:
            print(f"Error fetching watchlists: {e}")

        return watchlists

    def sync_to_database(self, watchlists: Dict[str, List[str]]) -> bool:
        """Sync watchlists to PostgreSQL database"""
        try:
            for watchlist_name, symbols in watchlists.items():
                # Insert or update watchlist
                self.cursor.execute("""
                    INSERT INTO tradingview_watchlists (name, description, last_synced)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (name) DO UPDATE SET
                        last_synced = NOW()
                    RETURNING id
                """, (watchlist_name, f"TradingView {watchlist_name}"))

                watchlist_id = self.cursor.fetchone()['id']

                # Clear existing stocks for this watchlist
                self.cursor.execute("""
                    DELETE FROM watchlist_stocks WHERE watchlist_id = %s
                """, (watchlist_id,))

                # Insert new stocks
                for symbol in symbols:
                    try:
                        # Get current price
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                        self.cursor.execute("""
                            INSERT INTO watchlist_stocks (watchlist_id, symbol, current_price, price_updated)
                            VALUES (%s, %s, %s, NOW())
                            ON CONFLICT (watchlist_id, symbol) DO UPDATE SET
                                current_price = EXCLUDED.current_price,
                                price_updated = NOW()
                        """, (watchlist_id, symbol, current_price))

                    except Exception as e:
                        # Insert without price if fetch fails
                        self.cursor.execute("""
                            INSERT INTO watchlist_stocks (watchlist_id, symbol)
                            VALUES (%s, %s)
                            ON CONFLICT (watchlist_id, symbol) DO NOTHING
                        """, (watchlist_id, symbol))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error syncing to database: {e}")
            self.conn.rollback()
            return False

    def get_watchlists_from_db(self) -> Dict[str, List[Dict]]:
        """Get all watchlists from database"""
        watchlists = {}

        try:
            # Get all watchlists
            self.cursor.execute("""
                SELECT * FROM tradingview_watchlists
                ORDER BY name
            """)
            watchlist_records = self.cursor.fetchall()

            for watchlist in watchlist_records:
                # Get stocks for this watchlist
                self.cursor.execute("""
                    SELECT symbol, current_price, price_updated
                    FROM watchlist_stocks
                    WHERE watchlist_id = %s
                    ORDER BY symbol
                """, (watchlist['id'],))

                stocks = self.cursor.fetchall()
                watchlists[watchlist['name']] = stocks

        except Exception as e:
            print(f"Error fetching from database: {e}")

        return watchlists

    def update_prices(self) -> int:
        """Update prices for all stocks in watchlists"""
        updated = 0

        try:
            # Get all unique symbols
            self.cursor.execute("""
                SELECT DISTINCT symbol FROM watchlist_stocks
            """)
            symbols = [row['symbol'] for row in self.cursor.fetchall()]

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                    if current_price > 0:
                        self.cursor.execute("""
                            UPDATE watchlist_stocks
                            SET current_price = %s, price_updated = NOW()
                            WHERE symbol = %s
                        """, (current_price, symbol))
                        updated += 1

                except Exception as e:
                    print(f"Error updating {symbol}: {e}")

            self.conn.commit()

        except Exception as e:
            print(f"Error updating prices: {e}")
            self.conn.rollback()

        return updated

    def auto_sync(self) -> bool:
        """Automatically sync watchlists and store in database"""
        if not self.connect_db():
            return False

        try:
            # Create tables if needed
            self.create_tables()

            # Fetch watchlists
            print("Fetching TradingView watchlists...")
            watchlists = self.fetch_tradingview_watchlists()

            if watchlists:
                print(f"Found {len(watchlists)} watchlists")

                # Sync to database
                if self.sync_to_database(watchlists):
                    print("Successfully synced to database")
                    return True
                else:
                    print("Failed to sync to database")
                    return False
            else:
                print("No watchlists found")
                return False

        finally:
            self.disconnect_db()

    def run_background_sync(self, interval_minutes: int = 30):
        """Run sync in background at regular intervals"""
        while True:
            try:
                print(f"Running sync at {datetime.now()}")
                self.auto_sync()
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Sync error: {e}")
                time.sleep(60)  # Wait 1 minute on error