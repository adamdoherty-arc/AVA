"""
Xtrades.net Scraper - Discord OAuth Authentication & Trade Alert Extraction
============================================================================
Logs into Xtrades.net via Discord OAuth and extracts trade alerts from profile pages.

Features:
- Discord OAuth login flow
- Profile scraping (https://app.xtrades.net/profile/{username})
- Alert parsing with multiple strategy types
- Session management with cookie persistence
- Anti-detection measures
- Retry logic with exponential backoff

Author: Magnus Wheel Strategy Dashboard
Created: 2025-11-02
"""

import os
import re
import time
import json
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class XtradesScraperException(Exception):
    """Base exception for Xtrades scraper errors"""
    pass


class LoginFailedException(XtradesScraperException):
    """Raised when login fails"""
    pass


class ProfileNotFoundException(XtradesScraperException):
    """Raised when profile is not found"""
    pass


class XtradesScraper:
    """
    Scraper for Xtrades.net that handles Discord OAuth login and trade alert extraction.

    Usage:
        scraper = XtradesScraper()
        try:
            scraper.login()
            alerts = scraper.get_profile_alerts("behappy")
            for alert in alerts:
                print(f"Trade: {alert['ticker']} - {alert['strategy']}")
        finally:
            scraper.close()
    """

    # Constants
    BASE_URL = "https://app.xtrades.net"
    LOGIN_URL = f"{BASE_URL}/login"
    DISCORD_OAUTH_URL = "https://discord.com/api/oauth2/authorize"

    # Timing constants (seconds)
    WAIT_SHORT = 1
    WAIT_MEDIUM = 2
    WAIT_LONG = 3
    WAIT_VERY_LONG = 5
    WAIT_PAGE_LOAD = 4
    WAIT_OAUTH_REDIRECT = 10

    # WebDriver timeouts
    DEFAULT_TIMEOUT = 20
    PAGE_VISIBLE_TIMEOUT = 15

    # Scrolling constants
    MAX_SCROLL_ATTEMPTS = 100
    SCROLL_PAUSE_SECONDS = 1.5
    NO_CHANGE_THRESHOLD = 3

    # Limits
    MAX_ALERTS_ABSOLUTE = 1000
    MAX_LOGIN_RETRIES = 3

    # Alert action patterns
    ACTION_PATTERNS = {
        'BTO': r'\bBTO\b',
        'STC': r'\bSTC\b',
        'BTC': r'\bBTC\b',
        'STO': r'\bSTO\b',
        'OPEN': r'\b(?:opened|opening)\b',
        'CLOSE': r'\b(?:closed|closing)\b'
    }

    # Strategy patterns
    STRATEGY_PATTERNS = {
        'CSP': r'\b(?:CSP|Cash[- ]?Secured[- ]?Put)\b',
        'CC': r'\b(?:CC|Covered[- ]?Call)\b',
        'Long Call': r'\b(?:Long[- ]?Call|LC)\b',
        'Long Put': r'\b(?:Long[- ]?Put|LP)\b',
        'Put Credit Spread': r'\b(?:Put[- ]?Credit[- ]?Spread|PCS)\b',
        'Call Credit Spread': r'\b(?:Call[- ]?Credit[- ]?Spread|CCS)\b',
        'Put Debit Spread': r'\b(?:Put[- ]?Debit[- ]?Spread|PDS)\b',
        'Call Debit Spread': r'\b(?:Call[- ]?Debit[- ]?Spread|CDS)\b',
        'Iron Condor': r'\b(?:Iron[- ]?Condor|IC)\b',
        'Butterfly': r'\b(?:Butterfly|BF)\b',
        'Straddle': r'\b(?:Straddle)\b',
        'Strangle': r'\b(?:Strangle)\b'
    }

    # Common words to exclude from ticker matching (false positives)
    TICKER_EXCLUSIONS = {
        'THE', 'AND', 'BUT', 'FOR', 'THIS', 'THAT', 'WITH', 'FROM', 'HAVE', 'BEEN',
        'CALL', 'PUT', 'BUY', 'SELL', 'OPEN', 'CLOSE', 'HOLD', 'LONG', 'SHORT',
        'MOVE', 'PLAY', 'RISK', 'STOP', 'TAKE', 'WAIT', 'WEEK', 'YEAR', 'TODAY',
        'NOW', 'NEW', 'OUT', 'ALL', 'GET', 'CAN', 'MAY', 'DAY', 'UP', 'GO',
        'ATM', 'OTM', 'ITM', 'DTE', 'EXP', 'LEAP', 'LEAPS', 'IV', 'HV',
        'AM', 'PM', 'EST', 'PST', 'UTC', 'US', 'USD', 'EUR', 'ATH', 'EOD', 'EOW',
        'HIGH', 'LOW', 'MID', 'NEXT', 'LAST', 'JUST', 'ALSO', 'ONLY', 'VERY',
        'FREE', 'PAID', 'GOOD', 'BEST', 'NICE', 'BACK', 'DOWN', 'THEM', 'WHEN'
    }

    def __init__(self, headless: bool = False, cache_dir: Optional[str] = None):
        """
        Initialize the Xtrades scraper.

        Args:
            headless: Run browser in headless mode (default: False for Discord login)
            cache_dir: Directory for caching cookies and session data
        """
        self.username = os.getenv('XTRADES_USERNAME')
        self.password = os.getenv('XTRADES_PASSWORD')

        if not self.username or not self.password:
            raise ValueError("XTRADES_USERNAME and XTRADES_PASSWORD must be set in .env")

        self.headless = headless
        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None

        # Cache directory for cookies
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.xtrades_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.cookies_file = self.cache_dir / 'cookies.pkl'

        self._setup_driver()

    def _setup_driver(self) -> None:
        """Setup Chrome driver with anti-detection measures"""
        options = uc.ChromeOptions()

        # Anti-detection options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        # Commented out - incompatible with some Chrome versions
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option('useAutomationExtension', False)

        # Use persistent Chrome profile to retain Discord session
        # This is more reliable than cookie-only approach for OAuth sessions
        profile_dir = self.cache_dir / 'chrome_profile'
        options.add_argument(f'--user-data-dir={profile_dir}')

        # More natural user agent (match current Chrome version)
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/142.0.0.0 Safari/537.36'
        )

        if self.headless:
            options.add_argument('--headless=new')
            options.add_argument('--disable-gpu')

        # Initialize driver
        try:
            self.driver = uc.Chrome(options=options, use_subprocess=True, version_main=None)

            # Try to maximize window, but don't fail if it doesn't work
            try:
                self.driver.maximize_window()
            except Exception as window_error:
                print(f"Warning: Could not maximize window: {window_error}")

            self.wait = WebDriverWait(self.driver, 20)

            # Remove webdriver flag
            try:
                self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                    'source': '''
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        })
                    '''
                })
            except Exception as cdp_error:
                print(f"Warning: Could not execute CDP command: {cdp_error}")

            print("Chrome driver initialized successfully")

        except Exception as e:
            raise XtradesScraperException(f"Failed to initialize Chrome driver: {e}")

    def _save_cookies(self) -> None:
        """Save cookies to file for session persistence"""
        try:
            cookies = self.driver.get_cookies()
            with open(self.cookies_file, 'wb') as f:
                pickle.dump(cookies, f)
            print(f"Cookies saved to {self.cookies_file}")
        except Exception as e:
            print(f"Warning: Failed to save cookies: {e}")

    def _load_cookies(self) -> bool:
        """Load cookies from file if available"""
        try:
            if not self.cookies_file.exists():
                return False

            with open(self.cookies_file, 'rb') as f:
                cookies = pickle.load(f)

            # Navigate to base URL first
            self.driver.get(self.BASE_URL)
            time.sleep(2)

            # Add cookies
            for cookie in cookies:
                # Remove expiry if present and invalid
                if 'expiry' in cookie:
                    del cookie['expiry']
                try:
                    self.driver.add_cookie(cookie)
                except Exception as e:
                    print(f"Warning: Could not add cookie {cookie.get('name')}: {e}")

            print("Cookies loaded successfully")
            return True

        except Exception as e:
            print(f"Warning: Failed to load cookies: {e}")
            return False

    def _is_logged_in(self) -> bool:
        """Check if currently logged in to Xtrades"""
        try:
            # Navigate to base URL
            self.driver.get(self.BASE_URL)
            time.sleep(3)

            # First check: Look for "Sign in" or "Sign up" buttons (NOT logged in)
            page_source = self.driver.page_source
            if 'Sign in' in page_source or 'Sign up' in page_source:
                # Double-check with button elements
                try:
                    sign_in_buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Sign in')] | //ion-button[contains(text(), 'Sign in')]")
                    if sign_in_buttons:
                        return False
                except:
                    pass

            # Check if we're on login page
            current_url = self.driver.current_url.lower()
            if 'login' in current_url or 'auth' in current_url:
                return False

            # Check for user profile elements or dashboard (logged in)
            indicators = [
                (By.CSS_SELECTOR, "[class*='profile']"),
                (By.CSS_SELECTOR, "[class*='user']"),
                (By.CSS_SELECTOR, "[class*='avatar']"),
                (By.XPATH, "//a[contains(@href, '/profile/')]"),
                (By.XPATH, "//*[contains(text(), 'Dashboard')]"),
            ]

            for by, selector in indicators:
                try:
                    self.driver.find_element(by, selector)
                    print("Already logged in to Xtrades")
                    return True
                except NoSuchElementException:
                    continue

            return False

        except Exception as e:
            print(f"Error checking login status: {e}")
            return False

    def login(self, retry_count: int = 3) -> bool:
        """
        Login to Xtrades.net via Discord OAuth.

        Args:
            retry_count: Number of retry attempts on failure

        Returns:
            True if login successful, False otherwise

        Raises:
            LoginFailedException: If login fails after all retries
        """
        for attempt in range(retry_count):
            try:
                print(f"\nLogin attempt {attempt + 1}/{retry_count}")

                # Try loading saved cookies first
                if attempt == 0 and self._load_cookies():
                    if self._is_logged_in():
                        print("Logged in using saved session")
                        return True
                    else:
                        print("Saved session expired, proceeding with fresh login")

                # Navigate to login page
                print(f"Navigating to {self.LOGIN_URL}")
                self.driver.get(self.LOGIN_URL)
                time.sleep(3)

                # Look for Discord login button
                discord_btn = self._find_discord_button()
                if not discord_btn:
                    raise LoginFailedException("Discord login button not found")

                print("Clicking Discord login button")
                discord_btn.click()
                time.sleep(3)

                # Handle Discord OAuth flow
                if not self._handle_discord_oauth():
                    raise LoginFailedException("Discord OAuth failed")

                # Verify login
                time.sleep(3)
                if self._is_logged_in():
                    print("Login successful!")
                    self._save_cookies()
                    return True
                else:
                    print("Login verification failed")

            except Exception as e:
                print(f"Login attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

        raise LoginFailedException(f"Failed to login after {retry_count} attempts")

    def _wait_for_page_visible(self, timeout: int = 15) -> bool:
        """
        Wait for Angular Ionic page to become visible (remove ion-page-invisible class).

        The Xtrades app uses Angular/Ionic which adds 'ion-page-invisible' class during
        page transitions with pointer-events: none, preventing clicks.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if page became visible, False if timeout
        """
        print("Waiting for page to become visible...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if any ion-page is visible (without ion-page-invisible class)
                visible_pages = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    ".ion-page:not(.ion-page-invisible)"
                )
                if visible_pages:
                    print("Page is now visible")
                    return True

                # Also check if discord button is interactable
                discord_btn = self.driver.find_elements(By.CSS_SELECTOR, "button.discord-btn")
                if discord_btn and discord_btn[0].is_displayed() and discord_btn[0].is_enabled():
                    # Check if parent has pointer-events: none
                    parent = self.driver.execute_script(
                        "return arguments[0].closest('.ion-page')", discord_btn[0]
                    )
                    if parent:
                        pointer_events = self.driver.execute_script(
                            "return window.getComputedStyle(arguments[0]).pointerEvents", parent
                        )
                        if pointer_events != 'none':
                            print("Discord button is now interactable")
                            return True

            except Exception as e:
                pass  # Continue waiting

            time.sleep(0.5)

        print(f"Timeout waiting for page to become visible after {timeout}s")
        return False

    def _find_discord_button(self) -> Optional[webdriver.remote.webelement.WebElement]:
        """Find and return Discord login button"""
        # First wait for page to be fully visible (Angular/Ionic transition complete)
        self._wait_for_page_visible()

        selectors = [
            # Most specific first - class-based selectors for buttons/links
            (By.CSS_SELECTOR, "button.discord-btn"),
            (By.CSS_SELECTOR, ".discord-btn"),
            (By.CSS_SELECTOR, "button[class*='discord']"),
            (By.CSS_SELECTOR, "a[class*='discord']"),
            # Text-based selectors (fallback if UI has text)
            (By.XPATH, "//button[contains(., 'Discord')]"),
            (By.XPATH, "//a[contains(., 'Discord')]"),
            (By.XPATH, "//*[contains(@class, 'discord')]"),
        ]

        for by, selector in selectors:
            try:
                element = self.wait.until(
                    EC.element_to_be_clickable((by, selector))
                )
                print(f"Found Discord button with selector: {selector}")
                return element
            except TimeoutException:
                continue

        return None

    def _handle_discord_oauth(self) -> bool:
        """
        Handle Discord OAuth login flow.

        Returns:
            True if OAuth completed successfully
        """
        try:
            # Wait for Discord login page
            time.sleep(2)
            current_url = self.driver.current_url

            # Check if we're on Discord OAuth page
            if 'discord.com' not in current_url:
                print("Not redirected to Discord OAuth")
                return False

            print("On Discord OAuth page")

            # Check if already authorized
            if 'authorize' not in current_url or 'xtrades' in current_url:
                print("Already authorized or redirected back")
                return True

            # Look for email input
            try:
                email_input = self.wait.until(
                    EC.presence_of_element_located((By.NAME, "email"))
                )
                email_input.clear()
                email_input.send_keys(self.username)
                print("Entered email credentials")
                time.sleep(1)
            except TimeoutException:
                print("Email input not found - might already be logged in")

            # Look for password input
            try:
                password_input = self.driver.find_element(By.NAME, "password")
                password_input.clear()
                password_input.send_keys(self.password)
                print("Entered password")
                time.sleep(1)
            except NoSuchElementException:
                print("Password input not found")

            # Click login button
            try:
                login_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                login_btn.click()
                print("Clicked Discord login button")
                time.sleep(3)
            except NoSuchElementException:
                print("Login button not found")

            # Look for "Authorize" button
            try:
                authorize_btn = self.wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Authorize') or contains(., 'authorize')]"))
                )
                authorize_btn.click()
                print("Clicked Authorize button")
                time.sleep(3)
            except TimeoutException:
                print("Authorize button not found - might be already authorized")

            # Wait for redirect back to Xtrades
            for _ in range(10):
                current_url = self.driver.current_url
                if 'xtrades.net' in current_url and 'login' not in current_url:
                    print("Redirected back to Xtrades")
                    return True
                time.sleep(1)

            print("Warning: Did not detect redirect back to Xtrades")
            return False

        except Exception as e:
            print(f"Error in Discord OAuth flow: {e}")
            return False

    def get_profile_alerts(
        self,
        username: str,
        max_alerts: Optional[int] = None
    ) -> List[Dict]:
        """
        Get trade alerts from a profile alerts tab.

        Args:
            username: Xtrades.net username to scrape
            max_alerts: Maximum number of alerts to retrieve (None = all)

        Returns:
            List of parsed trade alert dictionaries

        Raises:
            ProfileNotFoundException: If profile doesn't exist
            XtradesScraperException: For other errors
        """
        # Navigate to the profile page first
        profile_url = f"{self.BASE_URL}/profile/{username}"

        try:
            print(f"\nNavigating to profile page: {username}")
            self.driver.get(profile_url)
            time.sleep(4)

            # Check if profile exists
            if "404" in self.driver.title or "not found" in self.driver.page_source.lower():
                raise ProfileNotFoundException(f"Profile '{username}' not found")

            # Try multiple methods to find and click the Alerts tab
            alerts_tab_clicked = False

            # Method 1: Look for tab/button with "Alerts" text
            selectors_to_try = [
                "//button[contains(translate(., 'ALERTS', 'alerts'), 'alerts')]",
                "//a[contains(translate(., 'ALERTS', 'alerts'), 'alerts')]",
                "//div[@role='tab' and contains(translate(., 'ALERTS', 'alerts'), 'alerts')]",
                "//*[contains(@class, 'tab') and contains(translate(., 'ALERTS', 'alerts'), 'alerts')]",
                "//*[contains(@role, 'tab') and contains(text(), 'Alerts')]",
                "//nav//a[contains(text(), 'Alerts')]",
                "//nav//button[contains(text(), 'Alerts')]",
            ]

            for selector in selectors_to_try:
                try:
                    alerts_tab = self.driver.find_element(By.XPATH, selector)
                    alerts_tab.click()
                    time.sleep(3)
                    print(f"Clicked Alerts tab using selector: {selector[:50]}...")
                    alerts_tab_clicked = True
                    break
                except:
                    continue

            if not alerts_tab_clicked:
                print("Warning: Could not find Alerts tab, attempting direct URL")
                # Try direct /alerts URL
                alerts_url = f"{self.BASE_URL}/profile/{username}/alerts"
                self.driver.get(alerts_url)
                time.sleep(3)

                # If still not on alerts, scrape from current page
                if "404" in self.driver.title:
                    print("Alerts URL not valid, scraping from profile page")
                    self.driver.get(profile_url)
                    time.sleep(2)

            # Wait for alerts to load (Angular dynamic content)
            print("Waiting for alerts to load...")
            time.sleep(5)  # Give Angular time to fetch and render alerts

            # Scroll to load more content
            self._scroll_page()

            # Wait a bit more after scrolling
            time.sleep(2)

            # Get page source and parse
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Debug: Save page HTML for inspection (only in non-headless mode)
            if not self.headless:
                try:
                    debug_file = self.cache_dir / f"debug_{username}_page.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(page_source)
                    print(f"Debug: Saved page HTML to {debug_file}")
                except Exception as e:
                    print(f"Warning: Could not save debug HTML: {e}")

            # Find alerts
            alerts = self._find_alert_elements(soup)

            if not alerts:
                print(f"No alerts found for {username}")
                return []

            print(f"Found {len(alerts)} alert elements")

            # Parse each alert
            parsed_alerts = []
            for i, alert_elem in enumerate(alerts):
                if max_alerts and i >= max_alerts:
                    break

                try:
                    parsed = self.parse_alert(alert_elem)
                    if parsed:
                        parsed['profile_username'] = username
                        parsed_alerts.append(parsed)
                except Exception as e:
                    print(f"Error parsing alert {i+1}: {e}")
                    continue

            print(f"Successfully parsed {len(parsed_alerts)} alerts")
            return parsed_alerts

        except ProfileNotFoundException:
            raise
        except Exception as e:
            raise XtradesScraperException(f"Error getting profile alerts: {e}")

    def _scroll_page(self, scroll_pause: float = 1.5, max_scrolls: int = 100) -> int:
        """
        Scroll page to load ALL dynamic content using aggressive infinite scroll.

        Continues scrolling until no new content loads for 3 consecutive attempts.
        This ensures we get the full alert history, not just the first few posts.

        Args:
            scroll_pause: Seconds to wait after each scroll
            max_scrolls: Maximum scrolls to prevent infinite loops

        Returns:
            Number of scrolls performed
        """
        try:
            from selenium.webdriver.common.by import By

            last_height = self.driver.execute_script("return document.body.scrollHeight")
            last_post_count = 0
            no_change_iterations = 0

            print(f"Starting aggressive infinite scroll (max {max_scrolls} scrolls)...")

            for scroll_num in range(1, max_scrolls + 1):
                # Scroll to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)

                # Count current posts (app-post elements)
                try:
                    posts = self.driver.find_elements(By.TAG_NAME, "app-post")
                    current_count = len(posts)
                except Exception:
                    current_count = last_post_count

                # Get new page height
                new_height = self.driver.execute_script("return document.body.scrollHeight")

                # Check if anything changed (height OR post count)
                if new_height == last_height and current_count == last_post_count:
                    no_change_iterations += 1
                    print(f"  Scroll {scroll_num}: No new content (attempt {no_change_iterations}/3)")

                    # Stop after 3 consecutive scrolls with no change
                    if no_change_iterations >= 3:
                        print(f"✓ Reached end of content at scroll {scroll_num}")
                        print(f"✓ Total posts loaded: {current_count}")
                        return scroll_num
                else:
                    # Reset counter if we got new content
                    no_change_iterations = 0
                    new_posts = current_count - last_post_count
                    if new_posts > 0:
                        print(f"  Scroll {scroll_num}: +{new_posts} new posts (total: {current_count})")
                    else:
                        print(f"  Scroll {scroll_num}: Page height increased")

                last_height = new_height
                last_post_count = current_count

            # Reached max scrolls
            print(f"⚠ Reached max scrolls ({max_scrolls})")
            print(f"✓ Total posts loaded: {last_post_count}")
            return max_scrolls

        except Exception as e:
            print(f"Error during infinite scroll: {e}")
            return 0

    def _find_alert_elements(self, soup: BeautifulSoup) -> List:
        """
        Find alert elements in page HTML.

        Args:
            soup: BeautifulSoup object of page HTML

        Returns:
            List of alert elements
        """
        # Try various selectors for Xtrades.net alerts
        selectors = [
            # REAL ALERTS - Add this first!
            {'name': 'app-post'},
            # Angular/Xtrades-specific components
            {'name': 'app-alert-row'},
            {'name': 'app-alert-item'},
            {'name': 'app-trade-alert'},
            {'name': re.compile(r'app-.*alert.*', re.I)},
            # Table row selectors (alerts might be in table rows)
            {'name': 'div', 'class_': re.compile(r'row.*alert', re.I)},
            {'name': 'div', 'class_': re.compile(r'alert.*row', re.I)},
            {'name': 'div', 'class_': re.compile(r'row-profile', re.I)},  # Specific to Xtrades
            # Alert-specific selectors
            {'name': 'div', 'class_': re.compile(r'alert.*item', re.I)},
            {'name': 'div', 'class_': re.compile(r'alert.*card', re.I)},
            {'name': 'div', 'class_': re.compile(r'alert.*container', re.I)},
            {'name': 'div', 'attrs': {'data-alert': True}},
            # Trade/post selectors
            {'name': 'div', 'class_': re.compile(r'trade.*card', re.I)},
            {'name': 'div', 'class_': re.compile(r'post.*item', re.I)},
            {'name': 'article', 'class_': re.compile(r'alert', re.I)},
            {'name': 'article', 'class_': re.compile(r'trade', re.I)},
            # Generic containers that might hold alerts
            {'name': 'div', 'class_': re.compile(r'feed.*item', re.I)},
            {'name': 'div', 'class_': re.compile(r'activity.*item', re.I)},
            # Fallback broader selectors
            {'name': 'div', 'class_': re.compile(r'alert', re.I)},
            {'name': 'div', 'class_': re.compile(r'trade', re.I)},
            {'name': 'div', 'class_': re.compile(r'card', re.I)},
            {'name': 'article'},
        ]

        for selector in selectors:
            elements = soup.find_all(**selector)
            if elements:
                print(f"Found {len(elements)} elements with selector {selector}")
                return elements

        # Fallback: get all divs with substantial text
        all_divs = soup.find_all('div')
        alerts = [div for div in all_divs if div.get_text(strip=True) and len(div.get_text(strip=True)) > 20]

        return alerts

    def parse_alert(self, alert_element) -> Optional[Dict]:
        """
        Parse individual alert element into structured data.

        Args:
            alert_element: BeautifulSoup element or HTML string

        Returns:
            Dictionary with parsed trade data or None if parsing fails
        """
        try:
            # Get text content
            if isinstance(alert_element, str):
                alert_text = alert_element
            else:
                alert_text = alert_element.get_text(strip=True)

            if not alert_text or len(alert_text) < 5:
                return None

            # Initialize result
            result: Dict = {
                'alert_text': alert_text,
                'ticker': None,
                'strategy': None,
                'action': None,
                'entry_price': None,
                'exit_price': None,
                'quantity': None,
                'strike_price': None,
                'expiration_date': None,
                'pnl': None,
                'pnl_percent': None,
                'alert_timestamp': None,
                'entry_date': None,
                'exit_date': None,
                'status': 'open'
            }

            # Extract ticker (2-5 uppercase letters, isolated)
            # Must exclude common words that aren't tickers
            ticker_matches = re.findall(r'\b([A-Z]{2,5})\b', alert_text)
            for potential_ticker in ticker_matches:
                if potential_ticker not in self.TICKER_EXCLUSIONS:
                    result['ticker'] = potential_ticker
                    break

            # Extract strategy
            for strategy, pattern in self.STRATEGY_PATTERNS.items():
                if re.search(pattern, alert_text, re.IGNORECASE):
                    result['strategy'] = strategy
                    break

            # Extract action
            for action, pattern in self.ACTION_PATTERNS.items():
                if re.search(pattern, alert_text, re.IGNORECASE):
                    result['action'] = action
                    break

            # Extract prices (looking for $XX.XX or @$XX.XX)
            prices = re.findall(r'[@$]\$?(\d+\.?\d*)', alert_text)
            if prices:
                result['entry_price'] = float(prices[0])
                if len(prices) > 1:
                    # Could be exit price or strike
                    result['exit_price'] = float(prices[1])

            # Extract strike price (often like "$170 strike" or "$170 Put")
            strike_match = re.search(r'\$(\d+\.?\d*)\s*(?:strike|put|call)', alert_text, re.IGNORECASE)
            if strike_match:
                result['strike_price'] = float(strike_match.group(1))

            # Extract quantity
            qty_match = re.search(r'(\d+)\s*x\s*\$', alert_text)
            if not qty_match:
                qty_match = re.search(r'\b(\d+)\s*contracts?\b', alert_text, re.IGNORECASE)
            if qty_match:
                result['quantity'] = int(qty_match.group(1))
            else:
                result['quantity'] = 1  # Default

            # Extract P&L
            pnl_match = re.search(r'(\+|-)\$?(\d+\.?\d*)', alert_text)
            if pnl_match:
                sign = 1 if pnl_match.group(1) == '+' else -1
                result['pnl'] = sign * float(pnl_match.group(2))

            # Extract P&L percentage
            pnl_pct_match = re.search(r'(\+|-)(\d+\.?\d*)%', alert_text)
            if pnl_pct_match:
                sign = 1 if pnl_pct_match.group(1) == '+' else -1
                result['pnl_percent'] = sign * float(pnl_pct_match.group(2))

            # Extract dates (MM/DD, MM/DD/YY, MM/DD/YYYY)
            date_matches = re.findall(r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)', alert_text)
            if date_matches:
                result['expiration_date'] = self._parse_date(date_matches[0])

            # Determine status
            if result['action'] in ['STC', 'BTC', 'CLOSE'] or result['exit_price']:
                result['status'] = 'closed'
            elif 'expired' in alert_text.lower():
                result['status'] = 'expired'

            # Set timestamps
            result['alert_timestamp'] = datetime.now().isoformat()
            result['scraped_at'] = datetime.now().isoformat()

            # Only return if we have minimum required data for a TRADE (not just any post)
            # Requires: ticker AND at least one of (strategy, action, entry_price)
            # This filters out general discussion posts that just mention stock symbols
            if result['ticker'] and (result['strategy'] or result['action'] or result['entry_price']):
                # Validate the parsed data
                if self._validate_alert_data(result):
                    return result

            return None

        except Exception as e:
            print(f"Error parsing alert: {e}")
            return None

    def _validate_alert_data(self, result: Dict) -> bool:
        """
        Validate parsed alert data for sanity checks.

        Args:
            result: Parsed alert dictionary

        Returns:
            True if data passes validation, False otherwise
        """
        # Ticker must be valid length and uppercase
        ticker = result.get('ticker', '')
        if not ticker or len(ticker) < 2 or len(ticker) > 5:
            return False

        # Ticker should be uppercase (already filtered in extraction)
        if ticker != ticker.upper():
            return False

        # Entry price should be positive if present
        entry_price = result.get('entry_price')
        if entry_price is not None and entry_price <= 0:
            print(f"Warning: Invalid entry price {entry_price} for {ticker}")
            return False

        # Strike price should be positive if present
        strike_price = result.get('strike_price')
        if strike_price is not None and strike_price <= 0:
            print(f"Warning: Invalid strike price {strike_price} for {ticker}")
            return False

        # Exit price should be positive if present
        exit_price = result.get('exit_price')
        if exit_price is not None and exit_price <= 0:
            print(f"Warning: Invalid exit price {exit_price} for {ticker}")
            return False

        # Quantity should be positive if present
        quantity = result.get('quantity')
        if quantity is not None and quantity <= 0:
            print(f"Warning: Invalid quantity {quantity} for {ticker}")
            return False

        return True

    def _parse_date(self, date_str: str) -> Optional[str]:
        """
        Parse date string into ISO format.

        Args:
            date_str: Date string in various formats

        Returns:
            ISO format date string or None
        """
        try:
            # Try various formats
            formats = [
                '%m/%d/%Y',
                '%m/%d/%y',
                '%m/%d',
                '%m-%d-%Y',
                '%m-%d-%y',
                '%m-%d'
            ]

            for fmt in formats:
                try:
                    if '/' in date_str or '-' in date_str:
                        dt = datetime.strptime(date_str, fmt)
                        # If year not provided, assume current year
                        if dt.year == 1900:
                            dt = dt.replace(year=datetime.now().year)
                        return dt.date().isoformat()
                except ValueError:
                    continue

            return None

        except Exception as e:
            print(f"Error parsing date '{date_str}': {e}")
            return None

    def close(self) -> None:
        """Close browser and cleanup"""
        if self.driver:
            try:
                self.driver.quit()
                print("Browser closed")
            except Exception as e:
                print(f"Error closing browser: {e}")


# Convenience functions

def scrape_profile(username: str, max_alerts: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to scrape a profile.

    Args:
        username: Xtrades.net username
        max_alerts: Maximum alerts to retrieve

    Returns:
        List of parsed alerts
    """
    scraper = XtradesScraper()
    try:
        scraper.login()
        return scraper.get_profile_alerts(username, max_alerts)
    finally:
        scraper.close()


def main():
    """Example usage"""
    scraper = XtradesScraper()

    try:
        # Login
        print("Logging in to Xtrades.net...")
        scraper.login()

        # Scrape profile
        username = "behappy"  # Example profile
        print(f"\nScraping profile: {username}")
        alerts = scraper.get_profile_alerts(username, max_alerts=10)

        # Display results
        print(f"\n{'='*80}")
        print(f"Found {len(alerts)} alerts from {username}")
        print(f"{'='*80}\n")

        for i, alert in enumerate(alerts, 1):
            print(f"Alert {i}:")
            print(f"  Ticker: {alert['ticker']}")
            print(f"  Strategy: {alert['strategy']}")
            print(f"  Action: {alert['action']}")
            print(f"  Entry Price: ${alert['entry_price']}" if alert['entry_price'] else "  Entry Price: N/A")
            print(f"  Strike: ${alert['strike_price']}" if alert['strike_price'] else "  Strike: N/A")
            print(f"  Expiration: {alert['expiration_date']}" if alert['expiration_date'] else "  Expiration: N/A")
            print(f"  P&L: ${alert['pnl']}" if alert['pnl'] else "  P&L: N/A")
            print(f"  Status: {alert['status']}")
            print(f"  Alert Text: {alert['alert_text'][:100]}...")
            print()

    except LoginFailedException as e:
        print(f"\nLogin failed: {e}")
        print("Please check your credentials in .env file")
    except ProfileNotFoundException as e:
        print(f"\nProfile error: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
