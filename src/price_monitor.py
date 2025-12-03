"""
Real-Time Price Monitor
Tracks stock prices and detects zone events for alerts
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import logging
import time

from zone_database_manager import ZoneDatabaseManager
from zone_analyzer import ZoneAnalyzer

logger = logging.getLogger(__name__)


class PriceMonitor:
    """
    Monitors stock prices in real-time and detects zone events

    Zone Events:
    - PRICE_ENTERING_DEMAND: Price approaching demand zone from above
    - PRICE_AT_DEMAND: Price inside demand zone
    - PRICE_ENTERING_SUPPLY: Price approaching supply zone from below
    - PRICE_AT_SUPPLY: Price inside supply zone
    - ZONE_BOUNCE: Price bounced from zone
    - ZONE_BREAK: Price broke through zone
    """

    def __init__(
        self,
        db_manager: Optional[ZoneDatabaseManager] = None,
        zone_analyzer: Optional[ZoneAnalyzer] = None,
        alert_distance_pct: float = 2.0,
        check_interval_seconds: int = 60
    ):
        """
        Initialize price monitor

        Args:
            db_manager: Database manager instance
            zone_analyzer: Zone analyzer instance
            alert_distance_pct: Alert when price within X% of zone (default: 2%)
            check_interval_seconds: Seconds between price checks (default: 60)
        """
        self.db = db_manager or ZoneDatabaseManager()
        self.analyzer = zone_analyzer or ZoneAnalyzer()
        self.alert_distance_pct = alert_distance_pct
        self.check_interval_seconds = check_interval_seconds

        # Track alerted zones to avoid spam
        self.alerted_zones: Set[Tuple[int, str]] = set()  # (zone_id, alert_type)

        # Cache for price data
        self.price_cache: Dict[str, Dict] = {}
        self.cache_expiry_seconds = 30

    def monitor_symbol(self, symbol: str) -> List[Dict]:
        """
        Monitor a single symbol for zone events

        Args:
            symbol: Stock ticker

        Returns:
            List of zone events (if any)
        """

        events = []

        try:
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"{symbol}: Could not fetch current price")
                return events

            # Get active zones near current price
            zones = self.db.get_zones_near_price(
                symbol=symbol,
                current_price=current_price,
                distance_pct=10.0  # Look within 10%
            )

            if not zones:
                return events

            logger.info(f"{symbol}: Monitoring {len(zones)} zones at price ${current_price:.2f}")

            # Check each zone for events
            for zone in zones:
                zone_events = self._check_zone_events(zone, current_price)
                events.extend(zone_events)

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")

        return events

    def monitor_symbols_batch(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """
        Monitor multiple symbols in batch

        Args:
            symbols: List of stock tickers

        Returns:
            Dictionary mapping symbol -> list of events
        """

        all_events = {}

        for symbol in symbols:
            events = self.monitor_symbol(symbol)
            if events:
                all_events[symbol] = events

        return all_events

    def monitor_watchlist(self, watchlist_name: str = "default") -> Dict[str, List[Dict]]:
        """
        Monitor all symbols in a TradingView watchlist

        Args:
            watchlist_name: Watchlist name (default: 'default')

        Returns:
            Dictionary mapping symbol -> list of events
        """

        # Get symbols from database (assuming TradingView watchlist integration)
        # This would query the tradingview_watchlist_stocks table
        query = """
            SELECT DISTINCT ticker
            FROM tradingview_watchlist_stocks
            WHERE watchlist_name = %s
        """

        try:
            with self.db.get_connection() as conn:
                import psycopg2.extras
                with conn.cursor() as cursor:
                    cursor.execute(query, (watchlist_name,))
                    symbols = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching watchlist symbols: {e}")
            symbols = []

        if not symbols:
            logger.warning(f"No symbols found in watchlist '{watchlist_name}'")
            return {}

        logger.info(f"Monitoring {len(symbols)} symbols from watchlist '{watchlist_name}'")
        return self.monitor_symbols_batch(symbols)

    def monitor_all_active_zones(self) -> Dict[str, List[Dict]]:
        """
        Monitor all symbols with active zones

        Returns:
            Dictionary mapping symbol -> list of events
        """

        # Get all unique symbols with active zones
        query = """
            SELECT DISTINCT ticker
            FROM sd_zones
            WHERE is_active = TRUE
              AND status != 'BROKEN'
        """

        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    symbols = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching symbols with zones: {e}")
            symbols = []

        if not symbols:
            logger.info("No symbols with active zones")
            return {}

        logger.info(f"Monitoring {len(symbols)} symbols with active zones")
        return self.monitor_symbols_batch(symbols)

    def _check_zone_events(self, zone: Dict, current_price: float) -> List[Dict]:
        """
        Check for zone events (entries, touches, bounces, breaks)

        Args:
            zone: Zone dictionary
            current_price: Current stock price

        Returns:
            List of event dictionaries
        """

        events = []

        zone_id = zone['id']
        zone_type = zone['zone_type']
        zone_top = zone['zone_top']
        zone_bottom = zone['zone_bottom']
        zone_mid = zone['zone_midpoint']

        # Calculate distance
        if current_price > zone_top:
            distance_pct = ((current_price - zone_top) / zone_top) * 100
            position = 'ABOVE'
        elif current_price < zone_bottom:
            distance_pct = ((current_price - zone_bottom) / zone_bottom) * 100
            position = 'BELOW'
        else:
            distance_pct = 0.0
            position = 'INSIDE'

        # Check for events based on zone type and price position
        if zone_type == 'DEMAND':
            # Demand zone events

            if position == 'INSIDE':
                # Price is inside demand zone - BUYING OPPORTUNITY
                event = self._create_event(
                    zone=zone,
                    event_type='PRICE_AT_DEMAND',
                    current_price=current_price,
                    distance_pct=0.0,
                    priority='HIGH'
                )
                if self._is_new_alert(zone_id, 'PRICE_AT_DEMAND'):
                    events.append(event)
                    self._mark_alerted(zone_id, 'PRICE_AT_DEMAND')

            elif position == 'ABOVE' and abs(distance_pct) <= self.alert_distance_pct:
                # Price approaching demand zone from above
                event = self._create_event(
                    zone=zone,
                    event_type='PRICE_ENTERING_DEMAND',
                    current_price=current_price,
                    distance_pct=distance_pct,
                    priority='MEDIUM'
                )
                if self._is_new_alert(zone_id, 'PRICE_ENTERING_DEMAND'):
                    events.append(event)
                    self._mark_alerted(zone_id, 'PRICE_ENTERING_DEMAND')

            elif position == 'BELOW':
                # Price below demand zone - check for break
                if abs(distance_pct) > 5.0:
                    # Zone broken
                    event = self._create_event(
                        zone=zone,
                        event_type='ZONE_BREAK',
                        current_price=current_price,
                        distance_pct=distance_pct,
                        priority='LOW'
                    )
                    if self._is_new_alert(zone_id, 'ZONE_BREAK'):
                        events.append(event)
                        self._mark_alerted(zone_id, 'ZONE_BREAK')
                        # Mark zone as broken in database
                        self.db.mark_zone_broken(zone_id)

        else:  # SUPPLY zone
            # Supply zone events

            if position == 'INSIDE':
                # Price is inside supply zone - SELLING OPPORTUNITY
                event = self._create_event(
                    zone=zone,
                    event_type='PRICE_AT_SUPPLY',
                    current_price=current_price,
                    distance_pct=0.0,
                    priority='HIGH'
                )
                if self._is_new_alert(zone_id, 'PRICE_AT_SUPPLY'):
                    events.append(event)
                    self._mark_alerted(zone_id, 'PRICE_AT_SUPPLY')

            elif position == 'BELOW' and abs(distance_pct) <= self.alert_distance_pct:
                # Price approaching supply zone from below
                event = self._create_event(
                    zone=zone,
                    event_type='PRICE_ENTERING_SUPPLY',
                    current_price=current_price,
                    distance_pct=distance_pct,
                    priority='MEDIUM'
                )
                if self._is_new_alert(zone_id, 'PRICE_ENTERING_SUPPLY'):
                    events.append(event)
                    self._mark_alerted(zone_id, 'PRICE_ENTERING_SUPPLY')

            elif position == 'ABOVE':
                # Price above supply zone - check for break
                if abs(distance_pct) > 5.0:
                    # Zone broken
                    event = self._create_event(
                        zone=zone,
                        event_type='ZONE_BREAK',
                        current_price=current_price,
                        distance_pct=distance_pct,
                        priority='LOW'
                    )
                    if self._is_new_alert(zone_id, 'ZONE_BREAK'):
                        events.append(event)
                        self._mark_alerted(zone_id, 'ZONE_BREAK')
                        # Mark zone as broken in database
                        self.db.mark_zone_broken(zone_id)

        return events

    def _create_event(
        self,
        zone: Dict,
        event_type: str,
        current_price: float,
        distance_pct: float,
        priority: str
    ) -> Dict:
        """Create zone event dictionary"""

        return {
            'zone_id': zone['id'],
            'symbol': zone['ticker'],
            'zone_type': zone['zone_type'],
            'event_type': event_type,
            'current_price': current_price,
            'zone_top': zone['zone_top'],
            'zone_bottom': zone['zone_bottom'],
            'zone_midpoint': zone['zone_midpoint'],
            'distance_pct': distance_pct,
            'strength_score': zone.get('strength_score', 50),
            'status': zone.get('status', 'FRESH'),
            'priority': priority,
            'timestamp': datetime.now()
        }

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current stock price with caching

        Args:
            symbol: Stock ticker

        Returns:
            Current price or None
        """

        # Check cache
        now = time.time()
        if symbol in self.price_cache:
            cached_data = self.price_cache[symbol]
            if now - cached_data['timestamp'] < self.cache_expiry_seconds:
                return cached_data['price']

        # Fetch fresh price
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')

            if data.empty:
                logger.warning(f"{symbol}: No price data available")
                return None

            current_price = float(data['Close'].iloc[-1])

            # Update cache
            self.price_cache[symbol] = {
                'price': current_price,
                'timestamp': now
            }

            return current_price

        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def _is_new_alert(self, zone_id: int, alert_type: str) -> bool:
        """Check if this is a new alert (not recently sent)"""
        return (zone_id, alert_type) not in self.alerted_zones

    def _mark_alerted(self, zone_id: int, alert_type: str):
        """Mark zone+type as alerted"""
        self.alerted_zones.add((zone_id, alert_type))

    def reset_alert_tracking(self) -> None:
        """Reset alert tracking (call periodically to allow re-alerts)"""
        self.alerted_zones.clear()
        logger.info("Reset alert tracking")

    def get_price_summary(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get price summary for multiple symbols

        Args:
            symbols: List of tickers

        Returns:
            DataFrame with current prices and zone info
        """

        summary_data = []

        for symbol in symbols:
            try:
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue

                # Get zones near price
                zones = self.db.get_zones_near_price(
                    symbol=symbol,
                    current_price=current_price,
                    distance_pct=10.0
                )

                # Count zones by type
                demand_zones = sum(1 for z in zones if z['zone_type'] == 'DEMAND')
                supply_zones = sum(1 for z in zones if z['zone_type'] == 'SUPPLY')

                # Find nearest zone
                nearest_zone = None
                min_distance = float('inf')

                for zone in zones:
                    if current_price > zone['zone_top']:
                        distance = current_price - zone['zone_top']
                    elif current_price < zone['zone_bottom']:
                        distance = zone['zone_bottom'] - current_price
                    else:
                        distance = 0

                    if distance < min_distance:
                        min_distance = distance
                        nearest_zone = zone

                summary_data.append({
                    'Symbol': symbol,
                    'Current Price': current_price,
                    'Demand Zones': demand_zones,
                    'Supply Zones': supply_zones,
                    'Total Zones': len(zones),
                    'Nearest Zone Type': nearest_zone['zone_type'] if nearest_zone else None,
                    'Nearest Zone Distance': min_distance if nearest_zone else None
                })

            except Exception as e:
                logger.error(f"Error getting summary for {symbol}: {e}")

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Test price monitor
    logging.basicConfig(level=logging.INFO)

    monitor = PriceMonitor()

    print("Testing PriceMonitor with AAPL...")

    # Monitor single symbol
    events = monitor.monitor_symbol('AAPL')

    print(f"\nFound {len(events)} events:\n")

    for event in events:
        print(f"Event: {event['event_type']}")
        print(f"  Symbol: {event['symbol']}")
        print(f"  Price: ${event['current_price']:.2f}")
        print(f"  Zone: ${event['zone_bottom']:.2f} - ${event['zone_top']:.2f}")
        print(f"  Distance: {event['distance_pct']:.2f}%")
        print(f"  Priority: {event['priority']}")
        print()
