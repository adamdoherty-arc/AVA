"""
Modern Async Scraper for Xtrades.net
=====================================

Using Playwright for modern, async-first web scraping with:
- Automatic browser management
- Cookie persistence
- Anti-detection measures
- Concurrent alert processing
- Structured logging
"""

import asyncio
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
import structlog

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import XtradeAlert, XtradeProfile, AlertType

logger = structlog.get_logger(__name__)


# Constants
XTRADES_BASE_URL = "https://app.xtrades.net"
XTRADES_LOGIN_URL = f"{XTRADES_BASE_URL}/login"
XTRADES_FEED_URL = f"{XTRADES_BASE_URL}/feed"

# Timing constants
WAIT_SHORT = 1000  # 1 second in ms
WAIT_MEDIUM = 2000
WAIT_LONG = 3000
WAIT_PAGE_LOAD = 5000
PAGE_VISIBLE_TIMEOUT = 15000
DEFAULT_TIMEOUT = 30000

# Parsing patterns
TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
PRICE_PATTERN = re.compile(r'\$?(\d+(?:\.\d{1,2})?)')
STRIKE_PATTERN = re.compile(r'(\d+(?:\.\d{1,2})?)\s*[CP]|(\d+)\s*(?:call|put)', re.IGNORECASE)
DATE_PATTERN = re.compile(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?')

# Exclusions for ticker detection
TICKER_EXCLUSIONS = {
    'THE', 'AND', 'BUT', 'FOR', 'THIS', 'THAT', 'WITH', 'FROM', 'HAVE', 'BEEN',
    'WILL', 'WOULD', 'COULD', 'SHOULD', 'JUST', 'LIKE', 'MORE', 'SOME', 'THAN',
    'THEM', 'THEN', 'THERE', 'THESE', 'THEY', 'WHAT', 'WHEN', 'WHERE', 'WHICH',
    'WHILE', 'WHO', 'WHY', 'YOUR', 'ALSO', 'BACK', 'BEEN', 'CALL', 'PUT', 'BUY',
    'SELL', 'OPEN', 'CLOSE', 'HOLD', 'LONG', 'SHORT', 'BTO', 'STO', 'BTC', 'STC',
    'NOW', 'OUT', 'OTM', 'ITM', 'ATM', 'DTE', 'EXP', 'PT', 'SL', 'TP',
    'ALL', 'ANY', 'ARE', 'CAN', 'DAY', 'DID', 'GET', 'GOT', 'HAD', 'HAS', 'HER',
    'HIM', 'HIS', 'HOW', 'ITS', 'LET', 'MAY', 'NEW', 'NOT', 'OLD', 'ONE', 'OUR',
    'OWN', 'SAY', 'SHE', 'TOO', 'TRY', 'TWO', 'USE', 'WAS', 'WAY', 'YET', 'YOU',
}


class ModernXtradesScraper:
    """
    Modern async scraper for Xtrades.net using Playwright.

    Features:
    - Async/await first design
    - Automatic cookie persistence
    - Anti-detection with stealth mode
    - Structured logging
    - Retry with exponential backoff
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        headless: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
        max_alerts: int = 100
    ):
        """
        Initialize the scraper.

        Args:
            cache_dir: Directory for cookies and cache
            headless: Run browser in headless mode
            timeout: Default timeout for operations (ms)
            max_alerts: Maximum alerts to fetch per profile
        """
        self.cache_dir = cache_dir or Path.home() / '.xtrades_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cookies_file = self.cache_dir / 'cookies.pkl'

        self.headless = headless
        self.timeout = timeout
        self.max_alerts = max_alerts

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

        self.logger = logger.bind(component="ModernXtradesScraper")

    async def __aenter__(self) -> 'ModernXtradesScraper':
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the browser."""
        self.logger.info("Starting Playwright browser", headless=self.headless)

        self._playwright = await async_playwright().start()

        # Use Chromium with stealth-like settings
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
            ]
        )

        # Create context with realistic viewport
        self._context = await self._browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
        )

        # Set default timeout
        self._context.set_default_timeout(self.timeout)

        # Load saved cookies
        await self._load_cookies()

        self.logger.info("Browser started successfully")

    async def stop(self) -> None:
        """Stop the browser and save state."""
        self.logger.info("Stopping browser")

        if self._context:
            await self._save_cookies()
            await self._context.close()

        if self._browser:
            await self._browser.close()

        if self._playwright:
            await self._playwright.stop()

        self._context = None
        self._browser = None
        self._playwright = None

    async def _load_cookies(self) -> bool:
        """Load cookies from file."""
        if not self.cookies_file.exists():
            self.logger.warning("No cookies file found", path=str(self.cookies_file))
            return False

        try:
            with open(self.cookies_file, 'rb') as f:
                cookies = pickle.load(f)

            if cookies and self._context:
                # Convert Selenium cookie format to Playwright format
                playwright_cookies = []
                for cookie in cookies:
                    pc = {
                        'name': cookie['name'],
                        'value': cookie['value'],
                        'domain': cookie.get('domain', '.xtrades.net'),
                        'path': cookie.get('path', '/'),
                    }
                    # Add optional fields
                    if cookie.get('secure'):
                        pc['secure'] = True
                    if cookie.get('httpOnly'):
                        pc['httpOnly'] = True
                    if cookie.get('expiry'):
                        pc['expires'] = cookie['expiry']

                    playwright_cookies.append(pc)

                await self._context.add_cookies(playwright_cookies)
                self.logger.info("Loaded cookies", count=len(playwright_cookies))
                return True

        except Exception as e:
            self.logger.error("Failed to load cookies", error=str(e))

        return False

    async def _save_cookies(self) -> bool:
        """Save cookies to file."""
        if not self._context:
            return False

        try:
            cookies = await self._context.cookies()

            # Convert to Selenium-compatible format for backwards compatibility
            selenium_cookies = []
            for cookie in cookies:
                sc = {
                    'name': cookie['name'],
                    'value': cookie['value'],
                    'domain': cookie.get('domain', ''),
                    'path': cookie.get('path', '/'),
                    'secure': cookie.get('secure', False),
                    'httpOnly': cookie.get('httpOnly', False),
                }
                if cookie.get('expires'):
                    sc['expiry'] = int(cookie['expires'])
                selenium_cookies.append(sc)

            with open(self.cookies_file, 'wb') as f:
                pickle.dump(selenium_cookies, f)

            self.logger.info("Saved cookies", count=len(selenium_cookies))
            return True

        except Exception as e:
            self.logger.error("Failed to save cookies", error=str(e))
            return False

    async def _get_page(self) -> Page:
        """Get a new page."""
        if not self._context:
            raise RuntimeError("Browser not started. Call start() first.")
        return await self._context.new_page()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError,))
    )
    async def check_login_status(self) -> bool:
        """Check if we're logged in to Xtrades."""
        page = await self._get_page()

        try:
            await page.goto(XTRADES_FEED_URL, wait_until='networkidle')

            # Check if redirected to login
            current_url = page.url
            if 'login' in current_url.lower():
                self.logger.warning("Not logged in - redirected to login page")
                return False

            # Check for user elements
            try:
                await page.wait_for_selector('[data-testid="user-menu"], .user-avatar, .profile-link', timeout=5000)
                self.logger.info("Login verified - user elements found")
                return True
            except Exception:
                pass

            # Check for feed content
            try:
                await page.wait_for_selector('.feed-item, .alert-card, ion-card', timeout=5000)
                self.logger.info("Login verified - feed content found")
                return True
            except Exception:
                pass

            return False

        finally:
            await page.close()

    async def _wait_for_page_ready(self, page: Page) -> bool:
        """Wait for Angular/Ionic page to be fully loaded and visible."""
        try:
            # Wait for Ionic app to be ready
            await page.wait_for_function(
                """() => {
                    const app = document.querySelector('ion-app');
                    if (!app) return false;

                    // Check for invisible page class
                    const pages = document.querySelectorAll('ion-page');
                    for (const p of pages) {
                        if (p.classList.contains('ion-page-invisible')) {
                            return false;
                        }
                    }
                    return true;
                }""",
                timeout=PAGE_VISIBLE_TIMEOUT
            )
            return True
        except Exception as e:
            self.logger.warning("Page ready check failed", error=str(e))
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((TimeoutError,))
    )
    async def fetch_profile_alerts(
        self,
        profile: XtradeProfile,
        since: Optional[datetime] = None,
        max_alerts: Optional[int] = None
    ) -> List[XtradeAlert]:
        """
        Fetch alerts from a user profile.

        Args:
            profile: Profile to fetch alerts from
            since: Only fetch alerts after this time
            max_alerts: Maximum number of alerts to fetch

        Returns:
            List of parsed alerts
        """
        max_alerts = max_alerts or self.max_alerts
        profile_url = f"{XTRADES_BASE_URL}/profile/{profile.username}"

        self.logger.info("Fetching profile alerts", username=profile.username, url=profile_url)

        page = await self._get_page()

        try:
            # Navigate to profile
            await page.goto(profile_url, wait_until='networkidle')
            await self._wait_for_page_ready(page)
            await asyncio.sleep(WAIT_MEDIUM / 1000)  # Convert to seconds

            # Check for error states
            if 'not found' in (await page.content()).lower():
                self.logger.warning("Profile not found", username=profile.username)
                return []

            # Wait for alerts to load
            try:
                await page.wait_for_selector(
                    '.alert-card, .feed-item, ion-card, .post-card',
                    timeout=10000
                )
            except Exception:
                self.logger.warning("No alerts found on page", username=profile.username)
                return []

            # Scroll to load more alerts
            alerts_raw = []
            last_count = 0
            scroll_attempts = 0
            max_scroll_attempts = 20

            while len(alerts_raw) < max_alerts and scroll_attempts < max_scroll_attempts:
                # Extract current alerts
                alerts_raw = await self._extract_alerts_from_page(page)

                if len(alerts_raw) >= max_alerts:
                    break

                if len(alerts_raw) == last_count:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                    last_count = len(alerts_raw)

                # Scroll down
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await asyncio.sleep(WAIT_SHORT / 1000)

            self.logger.info(
                "Extracted raw alerts",
                username=profile.username,
                count=len(alerts_raw)
            )

            # Parse alerts
            alerts = []
            for raw in alerts_raw[:max_alerts]:
                alert = self._parse_alert(raw, profile)
                if alert:
                    # Filter by date if specified
                    if since and alert.posted_at < since:
                        continue
                    alerts.append(alert)

            self.logger.info(
                "Parsed alerts",
                username=profile.username,
                count=len(alerts)
            )

            return alerts

        finally:
            await page.close()

    async def _extract_alerts_from_page(self, page: Page) -> List[Dict[str, Any]]:
        """Extract raw alert data from page."""
        return await page.evaluate("""
            () => {
                const alerts = [];
                const cards = document.querySelectorAll('.alert-card, .feed-item, ion-card, .post-card, [class*="alert"], [class*="post"]');

                for (const card of cards) {
                    const text = card.innerText || card.textContent || '';
                    if (!text.trim()) continue;

                    // Try to find ID
                    let alertId = card.getAttribute('data-id') ||
                                  card.getAttribute('id') ||
                                  card.querySelector('[data-id]')?.getAttribute('data-id') ||
                                  '';

                    // Try to find timestamp
                    const timeEl = card.querySelector('time, [datetime], .timestamp, .time, .date');
                    let timestamp = timeEl?.getAttribute('datetime') ||
                                   timeEl?.innerText ||
                                   '';

                    // Get HTML for detailed parsing
                    const html = card.innerHTML || '';

                    alerts.push({
                        id: alertId,
                        text: text.trim(),
                        html: html,
                        timestamp: timestamp
                    });
                }

                return alerts;
            }
        """)

    def _parse_alert(self, raw: Dict[str, Any], profile: XtradeProfile) -> Optional[XtradeAlert]:
        """Parse raw alert data into XtradeAlert model."""
        try:
            text = raw.get('text', '')
            if not text or len(text) < 10:
                return None

            # Generate ID if missing
            alert_id = raw.get('id') or f"{profile.username}_{hash(text)}"

            # Parse timestamp
            timestamp_str = raw.get('timestamp', '')
            posted_at = self._parse_timestamp(timestamp_str)

            # Extract ticker
            ticker = self._extract_ticker(text)

            # Determine alert type
            alert_type = self._determine_alert_type(text)

            # Extract trade details
            strategy = self._extract_strategy(text)
            action = self._extract_action(text)
            prices = self._extract_prices(text)
            strike = self._extract_strike(text)
            expiration = self._extract_expiration(text)

            return XtradeAlert(
                profile_id=profile.id or 0,
                alert_id=alert_id,
                alert_text=text,
                alert_type=alert_type,
                posted_at=posted_at,
                ticker=ticker,
                strategy=strategy,
                action=action,
                strike_price=strike,
                expiration_date=expiration,
                entry_price=prices.get('entry'),
                target_price=prices.get('target'),
                stop_loss=prices.get('stop'),
                raw_html=raw.get('html')
            )

        except Exception as e:
            self.logger.error("Failed to parse alert", error=str(e))
            return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string into datetime."""
        if not timestamp_str:
            return datetime.utcnow()

        timestamp_str = timestamp_str.strip().lower()

        # Handle relative times
        if 'just now' in timestamp_str or 'now' in timestamp_str:
            return datetime.utcnow()
        if 'min' in timestamp_str:
            try:
                mins = int(re.search(r'(\d+)', timestamp_str).group(1))
                return datetime.utcnow() - timedelta(minutes=mins)
            except Exception:
                pass
        if 'hour' in timestamp_str or 'hr' in timestamp_str:
            try:
                hours = int(re.search(r'(\d+)', timestamp_str).group(1))
                return datetime.utcnow() - timedelta(hours=hours)
            except Exception:
                pass
        if 'day' in timestamp_str:
            try:
                days = int(re.search(r'(\d+)', timestamp_str).group(1))
                return datetime.utcnow() - timedelta(days=days)
            except Exception:
                pass

        # Try ISO format
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            pass

        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%m/%d/%y %H:%M',
            '%b %d, %Y',
            '%B %d, %Y',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except Exception:
                continue

        return datetime.utcnow()

    def _extract_ticker(self, text: str) -> Optional[str]:
        """Extract stock ticker from text."""
        # Look for $TICKER pattern first
        dollar_match = re.search(r'\$([A-Z]{1,5})\b', text)
        if dollar_match:
            return dollar_match.group(1)

        # Look for standalone tickers
        matches = TICKER_PATTERN.findall(text)
        for match in matches:
            if match not in TICKER_EXCLUSIONS and len(match) >= 2:
                return match

        return None

    def _determine_alert_type(self, text: str) -> AlertType:
        """Determine the type of alert."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ['bto', 'buy to open', 'opening', 'entry', 'entering']):
            return AlertType.ENTRY
        if any(kw in text_lower for kw in ['stc', 'sell to close', 'closing', 'exit', 'exiting', 'sold']):
            return AlertType.EXIT
        if any(kw in text_lower for kw in ['update', 'adjustment', 'rolling']):
            return AlertType.UPDATE
        if any(kw in text_lower for kw in ['watching', 'watchlist', 'eyeing', 'monitor']):
            return AlertType.WATCHLIST
        if any(kw in text_lower for kw in ['analysis', 'outlook', 'thesis', 'setup']):
            return AlertType.ANALYSIS

        return AlertType.OTHER

    def _extract_strategy(self, text: str) -> Optional[str]:
        """Extract trading strategy from text."""
        text_lower = text.lower()

        # Check for option notation (e.g., 150C, 200P)
        if re.search(r'\d+[cC]\b', text):
            return 'call'
        if re.search(r'\d+[pP]\b', text):
            return 'put'

        if 'call' in text_lower and 'put' not in text_lower:
            return 'call'
        if 'put' in text_lower and 'call' not in text_lower:
            return 'put'
        if 'spread' in text_lower:
            return 'spread'
        if 'iron condor' in text_lower:
            return 'iron_condor'
        if 'butterfly' in text_lower:
            return 'butterfly'
        if 'straddle' in text_lower:
            return 'straddle'
        if 'strangle' in text_lower:
            return 'strangle'
        if 'covered call' in text_lower:
            return 'covered_call'
        if 'csp' in text_lower or 'cash secured put' in text_lower:
            return 'cash_secured_put'
        if any(kw in text_lower for kw in ['stock', 'shares', 'equity']):
            return 'stock'

        return None

    def _extract_action(self, text: str) -> Optional[str]:
        """Extract trade action from text."""
        text_lower = text.lower()

        if 'bto' in text_lower or 'buy to open' in text_lower:
            return 'bto'
        if 'sto' in text_lower or 'sell to open' in text_lower:
            return 'sto'
        if 'btc' in text_lower or 'buy to close' in text_lower:
            return 'btc'
        if 'stc' in text_lower or 'sell to close' in text_lower:
            return 'stc'
        if 'buy' in text_lower or 'buying' in text_lower or 'long' in text_lower:
            return 'buy'
        if 'sell' in text_lower or 'selling' in text_lower or 'short' in text_lower:
            return 'sell'

        return None

    def _extract_prices(self, text: str) -> Dict[str, Optional[float]]:
        """Extract entry, target, and stop prices."""
        prices = {'entry': None, 'target': None, 'stop': None}
        text_lower = text.lower()

        # Entry price patterns
        entry_patterns = [
            r'(?:entry|enter|entered|bought|buy|@)\s*\$?(\d+(?:\.\d{1,2})?)',
            r'\$(\d+(?:\.\d{1,2})?)\s*(?:entry|avg)',
        ]
        for pattern in entry_patterns:
            match = re.search(pattern, text_lower)
            if match:
                prices['entry'] = float(match.group(1))
                break

        # Target price patterns
        target_patterns = [
            r'(?:target|pt|tp|goal)\s*\$?(\d+(?:\.\d{1,2})?)',
            r'\$?(\d+(?:\.\d{1,2})?)\s*(?:target|pt|tp)',
        ]
        for pattern in target_patterns:
            match = re.search(pattern, text_lower)
            if match:
                prices['target'] = float(match.group(1))
                break

        # Stop loss patterns
        stop_patterns = [
            r'(?:stop|sl|stop loss)\s*\$?(\d+(?:\.\d{1,2})?)',
            r'\$?(\d+(?:\.\d{1,2})?)\s*(?:stop|sl)',
        ]
        for pattern in stop_patterns:
            match = re.search(pattern, text_lower)
            if match:
                prices['stop'] = float(match.group(1))
                break

        return prices

    def _extract_strike(self, text: str) -> Optional[float]:
        """Extract option strike price."""
        # Pattern: 150C, 150P, 150 call, 150 put
        match = STRIKE_PATTERN.search(text)
        if match:
            strike = match.group(1) or match.group(2)
            if strike:
                return float(strike)
        return None

    def _extract_expiration(self, text: str) -> Optional[datetime]:
        """Extract option expiration date."""
        match = DATE_PATTERN.search(text)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)

            if year:
                year = int(year)
                if year < 100:
                    year += 2000
            else:
                # Assume current year or next year if date has passed
                year = datetime.now().year
                test_date = datetime(year, month, day)
                if test_date < datetime.now():
                    year += 1

            try:
                return datetime(year, month, day)
            except ValueError:
                pass

        return None

    async def fetch_all_profiles(
        self,
        profiles: List[XtradeProfile],
        max_concurrent: int = 3
    ) -> AsyncGenerator[tuple[XtradeProfile, List[XtradeAlert]], None]:
        """
        Fetch alerts from multiple profiles concurrently.

        Args:
            profiles: List of profiles to fetch
            max_concurrent: Maximum concurrent fetches

        Yields:
            Tuples of (profile, alerts)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(profile: XtradeProfile):
            async with semaphore:
                alerts = await self.fetch_profile_alerts(profile)
                return profile, alerts

        tasks = [fetch_with_limit(p) for p in profiles]

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                self.logger.error("Profile fetch failed", error=str(e))
