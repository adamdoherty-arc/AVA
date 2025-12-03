"""TradingView Watchlist Scraper - Automatically pull watchlists from TradingView"""

import os
import json
import time
from typing import List, Dict, Optional
from selenium import webdriver
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc
from pathlib import Path

class TradingViewScraper:
    """Automatically scrape watchlists from TradingView"""

    def __init__(self) -> None:
        self.username = os.getenv('TRADINGVIEW_USERNAME')
        self.password = os.getenv('TRADINGVIEW_PASSWORD')
        self.driver = None
        self.watchlists = {}
        self.cache_file = Path("tradingview_cache.json")

    def setup_driver(self) -> None:
        """Setup Chrome driver with undetected-chromedriver to avoid detection"""
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument("--disable-extensions")
        options.add_experimental_option('useAutomationExtension', False)
        # Remove excludeSwitches - causing compatibility issues
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])

        # Use existing Chrome profile if available
        user_data_dir = Path.home() / 'AppData' / 'Local' / 'Google' / 'Chrome' / 'User Data'
        if user_data_dir.exists():
            options.add_argument(f'--user-data-dir={user_data_dir}')
            options.add_argument('--profile-directory=Default')

        self.driver = uc.Chrome(options=options, version_main=None)
        self.driver.maximize_window()

    def login(self) -> bool:
        """Login to TradingView"""
        try:
            self.driver.get("https://www.tradingview.com/")
            time.sleep(3)

            # Check if already logged in
            if self.is_logged_in():
                print("Already logged in to TradingView")
                return True

            # Click on Sign in button
            sign_in_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Sign in')]"))
            )
            sign_in_btn.click()
            time.sleep(2)

            # Click Email button
            email_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Email')]"))
            )
            email_btn.click()
            time.sleep(1)

            # Enter email
            email_input = self.driver.find_element(By.NAME, "id_username")
            email_input.clear()
            email_input.send_keys(self.username)

            # Enter password
            password_input = self.driver.find_element(By.NAME, "id_password")
            password_input.clear()
            password_input.send_keys(self.password)

            # Submit
            submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_btn.click()

            # Wait for login to complete
            time.sleep(5)

            if self.is_logged_in():
                print("Successfully logged in to TradingView")
                return True
            else:
                print("Login failed - may need manual intervention")
                return False

        except Exception as e:
            print(f"Login error: {e}")
            return False

    def is_logged_in(self) -> bool:
        """Check if logged in"""
        try:
            # Look for user menu or avatar
            self.driver.find_element(By.CSS_SELECTOR, "[data-name='user-menu']")
            return True
        except:
            return False

    def get_watchlists(self) -> Dict[str, List[str]]:
        """Get all watchlists and their symbols"""
        try:
            # Navigate to watchlist page
            self.driver.get("https://www.tradingview.com/watchlists/")
            time.sleep(3)

            # Find all watchlist elements
            watchlist_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "[class*='watchlist']"
            )

            for element in watchlist_elements:
                try:
                    # Click on watchlist
                    element.click()
                    time.sleep(2)

                    # Get watchlist name
                    name = element.text.strip()

                    # Get symbols
                    symbols = self.extract_symbols()

                    if symbols:
                        self.watchlists[name] = symbols
                        print(f"Found watchlist '{name}' with {len(symbols)} symbols")

                except Exception as e:
                    print(f"Error processing watchlist: {e}")
                    continue

            # Save to cache
            self.save_cache()

            return self.watchlists

        except Exception as e:
            print(f"Error getting watchlists: {e}")
            # Try to load from cache
            return self.load_cache()

    def extract_symbols(self) -> List[str]:
        """Extract symbols from current watchlist view"""
        symbols = []
        try:
            # Wait for symbols to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='symbol']"))
            )

            # Get all symbol elements
            symbol_elements = self.driver.find_elements(
                By.CSS_SELECTOR, "[class*='symbol-link']"
            )

            for elem in symbol_elements:
                symbol = elem.text.strip()
                if symbol and len(symbol) <= 10:  # Basic validation
                    symbols.append(symbol.upper())

            # Remove duplicates
            symbols = list(set(symbols))

        except Exception as e:
            print(f"Error extracting symbols: {e}")

        return symbols

    def get_watchlist_api(self) -> Dict[str, List[str]]:
        """Alternative method using API-like approach"""
        try:
            # Get session cookies after login
            cookies = self.driver.get_cookies()

            # Create session with cookies
            import requests
            session = requests.Session()
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])

            # TradingView API endpoints
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.tradingview.com/',
                'Origin': 'https://www.tradingview.com'
            }

            # Get user data
            response = session.get(
                'https://www.tradingview.com/api/v1/user/me',
                headers=headers
            )

            if response.status_code == 200:
                user_data = response.json()
                user_id = user_data.get('id')

                # Get watchlists
                watchlist_response = session.get(
                    f'https://www.tradingview.com/api/v1/symbols_list/custom/{user_id}',
                    headers=headers
                )

                if watchlist_response.status_code == 200:
                    watchlist_data = watchlist_response.json()

                    for wl in watchlist_data.get('results', []):
                        name = wl.get('name', 'Unknown')
                        symbols = wl.get('symbols', [])
                        self.watchlists[name] = symbols

                    self.save_cache()
                    return self.watchlists

        except Exception as e:
            print(f"API method error: {e}")

        return self.load_cache()

    def save_cache(self) -> None:
        """Save watchlists to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'watchlists': self.watchlists,
                    'timestamp': time.time()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def load_cache(self) -> Dict[str, List[str]]:
        """Load watchlists from cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is less than 1 hour old
                    if time.time() - data.get('timestamp', 0) < 3600:
                        return data.get('watchlists', {})
        except Exception as e:
            print(f"Error loading cache: {e}")

        return {}

    def run(self) -> Dict[str, List[str]]:
        """Main method to get watchlists"""
        try:
            # Try cache first
            cached = self.load_cache()
            if cached:
                print(f"Using cached watchlists ({len(cached)} lists)")
                return cached

            # Setup driver
            self.setup_driver()

            # Login
            if not self.login():
                print("Failed to login to TradingView")
                return {}  # Return empty, no fallback data

            # Try API method first
            watchlists = self.get_watchlist_api()

            # If API fails, try scraping
            if not watchlists:
                watchlists = self.get_watchlists()

            return watchlists  # Return empty if no watchlists found

        except Exception as e:
            print(f"Error in run: {e}")
            return self.get_default_watchlists()
        finally:
            if self.driver:
                self.driver.quit()

    def get_default_watchlists(self) -> Dict[str, List[str]]:
        """Return empty watchlists - no fallback data"""
        return {}