"""
Position Data Aggregator
Fetches and enriches position data with market context, Greeks, and technical indicators
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import yfinance as yf
import robin_stocks.robinhood as rh
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnrichedPosition:
    """Complete position data with market context"""

    # Basic position data
    symbol: str
    position_type: str  # 'CSP', 'CC', 'Long Call', 'Long Put'
    strike: float
    expiration: date
    dte: int
    quantity: int

    # Financial metrics
    premium_collected: float
    current_value: float
    pnl_dollar: float
    pnl_percent: float

    # Market data
    stock_price: float
    stock_price_ah: Optional[float]
    stock_change_percent: float

    # Greeks (estimated or calculated)
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float

    # Moneyness metrics
    moneyness: str  # 'ITM', 'ATM', 'OTM'
    distance_to_strike: float  # percentage
    probability_itm: float  # 0-100

    # Volatility metrics
    iv_rank: Optional[float]
    iv_percentile: Optional[float]

    # Technical indicators
    stock_rsi: Optional[float]
    stock_trend: Optional[str]  # 'bullish', 'bearish', 'neutral'
    support_level: Optional[float]
    resistance_level: Optional[float]

    # News sentiment
    news_sentiment: Optional[float]  # -1 to 1
    news_count_24h: int

    # Metadata
    analyzed_at: datetime
    position_id: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert date objects to strings
        data['expiration'] = self.expiration.isoformat() if self.expiration else None
        data['analyzed_at'] = self.analyzed_at.isoformat()
        return data


class PositionDataAggregator:
    """
    Aggregates position data from multiple sources

    Data Flow:
    1. Fetch positions from Robinhood
    2. Enrich with market data (yfinance)
    3. Calculate Greeks (mibian or estimated)
    4. Add technical indicators
    5. Include news sentiment
    """

    def __init__(self) -> None:
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes

    def fetch_all_positions(self) -> List[EnrichedPosition]:
        """
        Fetch and enrich all option positions from Robinhood

        Returns:
            List of EnrichedPosition objects
        """
        try:
            # Ensure logged in
            if not self._ensure_login():
                logger.error("Failed to login to Robinhood")
                return []

            # Get option positions
            positions = rh.get_open_option_positions()

            enriched_positions = []
            for pos in positions:
                try:
                    enriched = self._enrich_position(pos)
                    if enriched:
                        enriched_positions.append(enriched)
                except Exception as e:
                    logger.error(f"Error enriching position: {e}")
                    continue

            logger.info(f"Fetched and enriched {len(enriched_positions)} positions")
            return enriched_positions

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def _ensure_login(self) -> bool:
        """Ensure Robinhood is logged in"""
        username = os.getenv('ROBINHOOD_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD')

        if not username or not password:
            return False

        try:
            rh.load_account_profile()
            return True
        except:
            try:
                rh.login(username=username, password=password)
                return True
            except Exception as e:
                logger.error(f"Login failed: {e}")
                return False

    def _enrich_position(self, rh_position: dict) -> Optional[EnrichedPosition]:
        """
        Enrich a single Robinhood position with market data

        Args:
            rh_position: Raw position dict from Robinhood

        Returns:
            EnrichedPosition object or None
        """
        try:
            # Parse option instrument
            instrument_url = rh_position.get('option')
            instrument = rh.get_option_instrument_data_by_id(
                instrument_url.split('/')[-2]
            )

            symbol = instrument['chain_symbol']
            strike = float(instrument['strike_price'])
            expiration_str = instrument['expiration_date']
            expiration = datetime.strptime(expiration_str, '%Y-%m-%d').date()
            option_type = instrument['type']  # 'call' or 'put'

            # Calculate DTE
            dte = (expiration - date.today()).days

            # Determine position type
            position_type = self._determine_position_type(
                option_type,
                float(rh_position.get('average_price', 0))
            )

            # Get quantity
            quantity = float(rh_position.get('quantity', 0))

            # Calculate P/L
            premium = float(rh_position.get('average_price', 0)) * quantity * 100
            current_value = float(rh_position.get('market_value', 0))
            pnl_dollar = current_value - premium
            pnl_percent = (pnl_dollar / premium * 100) if premium != 0 else 0

            # Get market data
            market_data = self._get_market_data(symbol)

            if not market_data:
                logger.warning(f"Failed to get market data for {symbol}")
                return None

            stock_price = market_data['price']
            stock_price_ah = market_data.get('price_ah')
            stock_change_pct = market_data['change_percent']

            # Calculate moneyness
            moneyness_data = self._calculate_moneyness(
                stock_price, strike, option_type
            )

            # Estimate Greeks
            greeks = self._estimate_greeks(
                stock_price, strike, dte, option_type, position_type
            )

            # Get technical indicators
            technicals = self._get_technical_indicators(symbol)

            # Get news sentiment
            news_data = self._get_news_sentiment(symbol)

            # Build position ID
            position_id = f"{symbol}_{strike}_{expiration_str}_{option_type}"

            # Create enriched position
            return EnrichedPosition(
                # Basic data
                symbol=symbol,
                position_type=position_type,
                strike=strike,
                expiration=expiration,
                dte=dte,
                quantity=int(quantity),
                # Financial
                premium_collected=premium,
                current_value=current_value,
                pnl_dollar=pnl_dollar,
                pnl_percent=pnl_percent,
                # Market
                stock_price=stock_price,
                stock_price_ah=stock_price_ah,
                stock_change_percent=stock_change_pct,
                # Greeks
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                implied_volatility=greeks['iv'],
                # Moneyness
                moneyness=moneyness_data['status'],
                distance_to_strike=moneyness_data['distance'],
                probability_itm=moneyness_data['prob_itm'],
                # Volatility
                iv_rank=technicals.get('iv_rank'),
                iv_percentile=technicals.get('iv_percentile'),
                # Technicals
                stock_rsi=technicals.get('rsi'),
                stock_trend=technicals.get('trend'),
                support_level=technicals.get('support'),
                resistance_level=technicals.get('resistance'),
                # News
                news_sentiment=news_data.get('sentiment'),
                news_count_24h=news_data.get('count', 0),
                # Metadata
                analyzed_at=datetime.now(),
                position_id=position_id
            )

        except Exception as e:
            logger.error(f"Error enriching position: {e}")
            return None

    def _determine_position_type(self, option_type: str, avg_price: float) -> str:
        """
        Determine position type based on option type and whether sold or bought

        Args:
            option_type: 'call' or 'put'
            avg_price: Average price paid/received

        Returns:
            Position type string
        """
        # If avg_price is negative, it was sold (credit received)
        # If positive, it was bought (debit paid)

        # Note: Robinhood returns average_price as positive for both
        # We need to check the 'type' field in the position
        # For now, we'll use heuristics

        # This is simplified - in production, check Robinhood's position type field
        if option_type == 'put':
            return 'CSP'  # Assume sold puts are CSPs
        else:
            return 'CC'  # Assume sold calls are CCs

    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current market data for a symbol

        Args:
            symbol: Stock ticker

        Returns:
            Dict with price data
        """
        cache_key = f"market_{symbol}"

        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get current price
            price = info.get('currentPrice') or info.get('regularMarketPrice')

            if not price:
                # Fallback to last close
                hist = ticker.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]

            # Get after-hours if available
            price_ah = info.get('postMarketPrice')

            # Get change percent
            change_pct = info.get('regularMarketChangePercent', 0)

            data = {
                'price': price,
                'price_ah': price_ah,
                'change_percent': change_pct,
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume')
            }

            # Cache it
            self.cache[cache_key] = (data, datetime.now())

            return data

        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    def _calculate_moneyness(
        self,
        stock_price: float,
        strike: float,
        option_type: str
    ) -> Dict:
        """
        Calculate moneyness metrics

        Args:
            stock_price: Current stock price
            strike: Option strike price
            option_type: 'call' or 'put'

        Returns:
            Dict with moneyness data
        """
        distance = ((stock_price - strike) / strike * 100)

        # Determine ITM/ATM/OTM
        threshold = 2.0  # 2% for ATM range

        if option_type == 'call':
            if stock_price > strike + (strike * threshold / 100):
                status = 'ITM'
                prob_itm = min(95, 50 + distance * 2)
            elif stock_price < strike - (strike * threshold / 100):
                status = 'OTM'
                prob_itm = max(5, 50 + distance * 2)
            else:
                status = 'ATM'
                prob_itm = 50
        else:  # put
            if stock_price < strike - (strike * threshold / 100):
                status = 'ITM'
                prob_itm = min(95, 50 - distance * 2)
            elif stock_price > strike + (strike * threshold / 100):
                status = 'OTM'
                prob_itm = max(5, 50 - distance * 2)
            else:
                status = 'ATM'
                prob_itm = 50

        return {
            'status': status,
            'distance': distance,
            'prob_itm': prob_itm
        }

    def _estimate_greeks(
        self,
        stock_price: float,
        strike: float,
        dte: int,
        option_type: str,
        position_type: str
    ) -> Dict:
        """
        Estimate Greeks using simplified Black-Scholes approximations

        For production, use mibian or Polygon API for real Greeks

        Args:
            stock_price: Current stock price
            strike: Strike price
            dte: Days to expiration
            option_type: 'call' or 'put'
            position_type: Position type

        Returns:
            Dict with Greeks
        """
        try:
            import mibian

            # Estimate IV (simplified - use historical volatility)
            # In production, fetch real IV from Polygon or calculate from option prices
            iv = 30.0  # Placeholder - 30% IV

            # Calculate Greeks using mibian
            if dte <= 0:
                dte = 1  # Avoid division by zero

            if option_type == 'call':
                bs = mibian.BS([stock_price, strike, 0.5, dte], volatility=iv)
                delta = bs.callDelta / 100
                theta = bs.callTheta
            else:
                bs = mibian.BS([stock_price, strike, 0.5, dte], volatility=iv)
                delta = bs.putDelta / 100
                theta = bs.putTheta

            gamma = bs.gamma
            vega = bs.vega / 100

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'iv': iv
            }

        except Exception as e:
            logger.warning(f"Error calculating Greeks with mibian: {e}")

            # Fallback: Simple approximations
            moneyness_ratio = stock_price / strike

            if option_type == 'call':
                delta = 0.5 if abs(moneyness_ratio - 1) < 0.05 else (0.8 if moneyness_ratio > 1 else 0.2)
            else:
                delta = -0.5 if abs(moneyness_ratio - 1) < 0.05 else (-0.8 if moneyness_ratio < 1 else -0.2)

            # Simple theta estimate: -stock_price * 0.001 per day
            theta = -stock_price * 0.001

            return {
                'delta': delta,
                'gamma': 0.05,
                'theta': theta,
                'vega': 0.10,
                'iv': 30.0
            }

    def _get_technical_indicators(self, symbol: str) -> Dict:
        """
        Calculate technical indicators

        Args:
            symbol: Stock ticker

        Returns:
            Dict with technical data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo', interval='1d')

            if hist.empty:
                return {}

            # Calculate RSI
            rsi = self._calculate_rsi(hist['Close'])

            # Determine trend
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            current_price = hist['Close'].iloc[-1]

            if current_price > sma_20 > sma_50:
                trend = 'bullish'
            elif current_price < sma_20 < sma_50:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Simple support/resistance (last 20 days)
            recent = hist.tail(20)
            support = recent['Low'].min()
            resistance = recent['High'].max()

            return {
                'rsi': rsi,
                'trend': trend,
                'support': support,
                'resistance': resistance,
                'iv_rank': None,  # Requires historical IV data
                'iv_percentile': None
            }

        except Exception as e:
            logger.error(f"Error calculating technicals for {symbol}: {e}")
            return {}

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1]

        except:
            return None

    def _get_news_sentiment(self, symbol: str) -> Dict:
        """
        Get news sentiment for symbol

        For production, integrate Finnhub or Polygon news APIs

        Args:
            symbol: Stock ticker

        Returns:
            Dict with sentiment data
        """
        # Placeholder - integrate real news API in production
        return {
            'sentiment': 0.0,
            'count': 0
        }


# ============================================================================
# Testing
# ============================================================================

def test_aggregator():
    """Test position data aggregation"""
    aggregator = PositionDataAggregator()

    print("\n" + "="*80)
    print("POSITION DATA AGGREGATOR TEST")
    print("="*80)

    # Fetch positions
    positions = aggregator.fetch_all_positions()

    print(f"\nFetched {len(positions)} positions")

    for pos in positions:
        print(f"\n{pos.symbol} ${pos.strike} {pos.position_type}")
        print(f"  P/L: ${pos.pnl_dollar:.2f} ({pos.pnl_percent:+.1f}%)")
        print(f"  Stock: ${pos.stock_price:.2f} ({pos.stock_change_percent:+.2f}%)")
        print(f"  DTE: {pos.dte} | Moneyness: {pos.moneyness}")
        print(f"  Greeks: Δ={pos.delta:.2f} Θ={pos.theta:.2f}")


if __name__ == "__main__":
    test_aggregator()
