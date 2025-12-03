"""TradingView Account Synchronization - Fetches real watchlists from your TradingView account"""

import os
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingViewAccountSync:
    """Sync actual TradingView watchlists to PostgreSQL database"""

    def __init__(self) -> None:
        self.username = os.getenv('TRADINGVIEW_USERNAME')
        self.password = os.getenv('TRADINGVIEW_PASSWORD')
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'database': os.getenv('DB_NAME', 'magnus')
        }
        self.driver = None
        self.logged_in = False

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def init_driver(self) -> None:
        """Initialize Chrome driver in headless mode"""
        if self.driver:
            return

        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome driver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Chrome driver: {e}")
            raise

    def login_to_tradingview(self) -> bool:
        """Login to TradingView account"""
        if self.logged_in:
            return True

        try:
            self.init_driver()

            # Navigate to TradingView login page
            logger.info("Navigating to TradingView...")
            self.driver.get("https://www.tradingview.com/accounts/signin/")
            time.sleep(3)

            # Click email signin button
            try:
                email_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, 'emailButton')]"))
                )
                email_button.click()
                time.sleep(2)
            except:
                logger.info("Email button not found, proceeding with form")

            # Enter username
            username_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "id_username"))
            )
            username_input.clear()
            username_input.send_keys(self.username)
            time.sleep(1)

            # Enter password
            password_input = self.driver.find_element(By.NAME, "id_password")
            password_input.clear()
            password_input.send_keys(self.password)
            time.sleep(1)

            # Submit form
            submit_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_button.click()

            # Wait for login to complete
            time.sleep(5)

            # Check if logged in by looking for user menu
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//button[@aria-label='User menu']"))
                )
                self.logged_in = True
                logger.info("Successfully logged in to TradingView")
                return True
            except:
                logger.error("Login failed - could not find user menu")
                return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def fetch_watchlists(self) -> Dict[str, List[str]]:
        """Fetch all watchlists from TradingView account"""
        if not self.login_to_tradingview():
            logger.error("Cannot fetch watchlists - not logged in")
            return {}

        try:
            # Navigate to watchlists
            self.driver.get("https://www.tradingview.com/watchlists/")
            time.sleep(3)

            watchlists = {}

            # Find all watchlist elements
            watchlist_elements = self.driver.find_elements(
                By.XPATH,
                "//div[contains(@class, 'listContainer')]//a[contains(@class, 'item')]"
            )

            logger.info(f"Found {len(watchlist_elements)} watchlists")

            for element in watchlist_elements:
                try:
                    # Get watchlist name
                    name = element.text.strip()
                    if not name:
                        continue

                    # Click on watchlist
                    element.click()
                    time.sleep(2)

                    # Get symbols from the watchlist
                    symbols = []
                    symbol_elements = self.driver.find_elements(
                        By.XPATH,
                        "//div[contains(@class, 'symbolName')]"
                    )

                    for sym_elem in symbol_elements:
                        symbol = sym_elem.text.strip()
                        if symbol and ':' in symbol:
                            # Extract just the symbol part (e.g., "NASDAQ:NVDA" -> "NVDA")
                            symbol = symbol.split(':')[-1]
                        if symbol:
                            symbols.append(symbol)

                    if symbols:
                        watchlists[name] = symbols
                        logger.info(f"Watchlist '{name}': {len(symbols)} symbols")

                except Exception as e:
                    logger.warning(f"Error processing watchlist: {e}")
                    continue

            return watchlists

        except Exception as e:
            logger.error(f"Failed to fetch watchlists: {e}")
            return {}

    def fetch_watchlist_api(self) -> Dict[str, List[str]]:
        """Alternative method using TradingView's internal API"""
        if not self.login_to_tradingview():
            return {}

        try:
            # Get cookies from selenium session
            cookies = self.driver.get_cookies()
            session = requests.Session()

            # Transfer cookies to requests session
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])

            # Try to fetch watchlists via API
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.tradingview.com/watchlists/',
                'Accept': 'application/json',
            }

            # Fetch user watchlists
            response = session.get(
                'https://www.tradingview.com/api/v1/symbols_list/custom/',
                headers=headers
            )

            if response.status_code == 200:
                data = response.json()
                watchlists = {}

                for wl in data:
                    name = wl.get('name', '')
                    symbols = wl.get('symbols', [])

                    # Clean symbols
                    clean_symbols = []
                    for symbol in symbols:
                        if isinstance(symbol, str):
                            # Remove exchange prefix if present
                            if ':' in symbol:
                                symbol = symbol.split(':')[-1]
                            clean_symbols.append(symbol)

                    if clean_symbols:
                        watchlists[name] = clean_symbols

                return watchlists
            else:
                logger.warning(f"API request failed with status {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            return {}

    def save_to_database(self, watchlists: Dict[str, List[str]]):
        """Save watchlists to PostgreSQL database"""
        conn = self.get_connection()
        cur = conn.cursor()

        try:
            # Create tables if they don't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tradingview_watchlists (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    symbols TEXT[],
                    symbol_count INTEGER,
                    last_synced TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create individual symbol table for easier querying
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tradingview_symbols (
                    id SERIAL PRIMARY KEY,
                    watchlist_name VARCHAR(255) REFERENCES tradingview_watchlists(name) ON DELETE CASCADE,
                    symbol VARCHAR(20) NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(watchlist_name, symbol)
                )
            """)

            # Save each watchlist
            for name, symbols in watchlists.items():
                # Insert or update watchlist
                cur.execute("""
                    INSERT INTO tradingview_watchlists (name, symbols, symbol_count, last_synced)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (name)
                    DO UPDATE SET
                        symbols = EXCLUDED.symbols,
                        symbol_count = EXCLUDED.symbol_count,
                        last_synced = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                """, (name, symbols, len(symbols)))

                # Clear old symbols for this watchlist
                cur.execute("""
                    DELETE FROM tradingview_symbols WHERE watchlist_name = %s
                """, (name,))

                # Insert new symbols
                for symbol in symbols:
                    cur.execute("""
                        INSERT INTO tradingview_symbols (watchlist_name, symbol)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                    """, (name, symbol))

            conn.commit()
            logger.info(f"Saved {len(watchlists)} watchlists to database")

        except Exception as e:
            logger.error(f"Database save failed: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def sync_watchlists(self) -> Dict[str, List[str]]:
        """Main method to sync TradingView watchlists to database"""
        logger.info("Starting TradingView sync...")

        # Try API method first (faster)
        watchlists = self.fetch_watchlist_api()

        # If API fails, try web scraping
        if not watchlists:
            logger.info("API method failed, trying web scraping...")
            watchlists = self.fetch_watchlists()

        if watchlists:
            self.save_to_database(watchlists)
            logger.info(f"Successfully synced {len(watchlists)} watchlists")
        else:
            logger.warning("No watchlists found to sync")

        return watchlists

    def get_watchlists_from_db(self) -> Dict[str, List[str]]:
        """Get all watchlists from database"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT name, symbols, symbol_count, last_synced
                FROM tradingview_watchlists
                ORDER BY name
            """)

            watchlists = {}
            for row in cur.fetchall():
                watchlists[row['name']] = row['symbols'] or []

            return watchlists

        except Exception as e:
            logger.error(f"Failed to fetch from database: {e}")
            return {}
        finally:
            cur.close()
            conn.close()

    def close(self) -> None:
        """Close the driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.logged_in = False


# Example usage
if __name__ == "__main__":
    syncer = TradingViewAccountSync()

    try:
        # Sync watchlists from TradingView
        watchlists = syncer.sync_watchlists()

        if watchlists:
            print("\n" + "="*50)
            print("Successfully synced TradingView watchlists:")
            print("="*50)

            for name, symbols in watchlists.items():
                print(f"\n📋 {name}: {len(symbols)} symbols")
                print(f"   Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        else:
            print("No watchlists found or sync failed")

    finally:
        syncer.close()