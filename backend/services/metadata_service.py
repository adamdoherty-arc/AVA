"""
Metadata Service - Stock/Company Metadata for Position Enrichment
===================================================================

OPTIMIZATIONS APPLIED:
1. Redis/In-Memory distributed caching (replaces global dict)
2. Async-compatible cache operations
3. Batch metadata fetching with parallel processing
4. Proper cache TTL management

Updated: 2025-11-29 - Performance optimizations
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

logger = logging.getLogger(__name__)

# Thread pool for concurrent yfinance calls
_yf_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="yf_meta")


class MetadataService:
    """
    Service for fetching stock/company metadata.

    Performance Features:
    - Distributed caching (Redis with in-memory fallback)
    - Parallel batch fetching for multiple symbols
    - 1-hour TTL for metadata (rarely changes)
    """

    # Cache TTLs
    CACHE_TTL_METADATA = 3600   # 1 hour for company metadata
    CACHE_TTL_CALENDAR = 1800   # 30 minutes for earnings calendar

    def __init__(self) -> None:
        # Try to use distributed cache
        try:
            from backend.infrastructure.cache import get_cache
            self._cache = get_cache()
            self._use_distributed = True
        except ImportError:
            self._use_distributed = False
            self._local_cache: Dict[str, tuple] = {}

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value (sync-compatible)."""
        if self._use_distributed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return None
                return loop.run_until_complete(self._cache.get(key))
            except Exception:
                return None
        else:
            if key in self._local_cache:
                value, expiry = self._local_cache[key]
                if datetime.now() < expiry:
                    return value
                del self._local_cache[key]
            return None

    def _set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cached value (sync-compatible)."""
        if self._use_distributed:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self._cache.set(key, value, ttl))
            except Exception:
                pass
        else:
            expiry = datetime.now() + timedelta(seconds=ttl)
            self._local_cache[key] = (value, expiry)

    def get_symbol_metadata(
        self, symbol: str, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get quick metadata for a symbol: sector, market cap, earnings, etc.

        Uses distributed caching with 1-hour TTL.
        """
        cache_key = f"metadata:symbol:{symbol}"

        # Check cache unless forced refresh
        if not force_refresh:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get earnings date
            calendar = ticker.calendar
            next_earnings = None
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.columns:
                    next_earnings = str(calendar['Earnings Date'].iloc[0])[:10]

            metadata = {
                "symbol": symbol,
                "name": info.get("longName") or info.get("shortName") or symbol,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "market_cap_formatted": self._format_market_cap(
                    info.get("marketCap", 0)
                ),
                "pe_ratio": round(info.get("trailingPE", 0), 2)
                    if info.get("trailingPE") else None,
                "forward_pe": round(info.get("forwardPE", 0), 2)
                    if info.get("forwardPE") else None,
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2)
                    if info.get("dividendYield") else None,
                "beta": round(info.get("beta", 1), 2)
                    if info.get("beta") else None,
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "next_earnings": next_earnings,
                "analyst_rating": info.get("recommendationKey", "")
                    .replace("_", " ").title(),
                "analyst_target": info.get("targetMeanPrice"),
                "tradingview_url": f"https://www.tradingview.com/chart/?symbol={symbol}",
                "updated_at": datetime.now().isoformat()
            }

            # Cache the result
            self._set_cache(cache_key, metadata, self.CACHE_TTL_METADATA)

            return metadata

        except Exception as e:
            logger.warning(f"Error fetching metadata for {symbol}: {e}")
            return {
                "symbol": symbol,
                "name": symbol,
                "sector": "Unknown",
                "industry": "Unknown",
                "market_cap": 0,
                "market_cap_formatted": "N/A",
                "tradingview_url": f"https://www.tradingview.com/chart/?symbol={symbol}",
                "error": str(e)
            }

    def get_batch_metadata(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for multiple symbols with parallel fetching.

        Performance: Fetches all symbols concurrently using thread pool.
        """
        result = {}

        # First, check cache for all symbols
        uncached = []
        for symbol in symbols:
            cached = self._get_cached(f"metadata:symbol:{symbol}")
            if cached:
                result[symbol] = cached
            else:
                uncached.append(symbol)

        if not uncached:
            return result

        # Fetch uncached symbols in parallel using ThreadPoolExecutor
        def fetch_single(sym: str) -> tuple:
            return (sym, self.get_symbol_metadata(sym))

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_single, sym) for sym in uncached]
            for future in futures:
                try:
                    sym, data = future.result(timeout=30)
                    result[sym] = data
                except Exception as e:
                    logger.warning(f"Batch fetch failed for symbol: {e}")

        return result

    async def get_batch_metadata_async(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Async version of batch metadata fetch.

        Uses asyncio.gather for parallel fetching.
        """
        result = {}
        uncached = []

        # Check cache first
        for symbol in symbols:
            if self._use_distributed:
                cached = await self._cache.get(f"metadata:symbol:{symbol}")
            else:
                cached = self._get_cached(f"metadata:symbol:{symbol}")

            if cached:
                result[symbol] = cached
            else:
                uncached.append(symbol)

        if not uncached:
            return result

        # Fetch uncached in parallel
        loop = asyncio.get_event_loop()

        async def fetch_one(sym: str) -> tuple:
            data = await loop.run_in_executor(
                _yf_executor, self.get_symbol_metadata, sym
            )
            return (sym, data)

        tasks = [fetch_one(sym) for sym in uncached]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for item in fetched:
            if isinstance(item, tuple):
                sym, data = item
                result[sym] = data

        return result

    def _format_market_cap(self, value: int) -> str:
        """Format market cap as readable string"""
        if not value or value == 0:
            return "N/A"
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        if value >= 1e6:
            return f"${value/1e6:.2f}M"
        return f"${value:,.0f}"

    def get_ai_position_recommendation(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI recommendation for a position based on various factors.
        """
        symbol = position.get("symbol", "")
        position_type = position.get("type", "stock")
        pl_pct = position.get("pl_pct", 0)
        dte = position.get("dte", 0)
        strategy = position.get("strategy", "")

        try:
            # Get current market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            info = ticker.info

            if hist.empty:
                return {"recommendation": "Hold", "reasoning": "Insufficient data", "confidence": 50}

            closes = hist['Close'].tolist()
            current_price = closes[-1]

            # Technical indicators
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else current_price
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else current_price
            trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 else "Neutral"

            # RSI calculation
            if len(closes) >= 15:
                deltas = [closes[i] - closes[i-1] for i in range(1, min(15, len(closes)))]
                gains = [d for d in deltas if d > 0]
                losses = [-d for d in deltas if d < 0]
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 0.001
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50

            # Generate recommendation based on position type
            if position_type == "stock":
                return self._stock_recommendation(symbol, trend, rsi, pl_pct, info)
            else:
                return self._option_recommendation(symbol, strategy, dte, pl_pct, trend, rsi, position)

        except Exception as e:
            logger.warning(f"Error generating recommendation for {symbol}: {e}")
            return {
                "recommendation": "Hold",
                "reasoning": "Unable to analyze",
                "confidence": 50,
                "error": str(e)
            }

    def _stock_recommendation(self, symbol: str, trend: str, rsi: float, pl_pct: float, info: Dict) -> Dict[str, Any]:
        """Generate recommendation for stock position"""
        reasoning = []
        actions = []
        confidence = 60

        # Trend analysis
        if trend == "Bullish":
            reasoning.append(f"Stock is in an uptrend")
            confidence += 10
        elif trend == "Bearish":
            reasoning.append(f"Stock is in a downtrend")
            confidence -= 5

        # RSI analysis
        if rsi > 70:
            reasoning.append(f"RSI at {rsi:.0f} indicates overbought conditions")
            actions.append("Consider taking profits")
        elif rsi < 30:
            reasoning.append(f"RSI at {rsi:.0f} indicates oversold conditions")
            actions.append("Consider adding to position")

        # P/L analysis
        if pl_pct > 25:
            actions.append("Consider taking partial profits (25%+ gain)")
        elif pl_pct < -15:
            actions.append("Review thesis - down 15%+")

        # Analyst rating
        rating = info.get("recommendationKey", "").lower()
        if rating in ["strong_buy", "buy"]:
            reasoning.append("Analysts rate as Buy")
            confidence += 10
        elif rating in ["sell", "strong_sell"]:
            reasoning.append("Analysts rate as Sell")
            confidence -= 10

        # Determine overall recommendation
        if rsi > 75 and pl_pct > 20:
            rec = "Take Profits"
        elif trend == "Bullish" and rsi < 60:
            rec = "Hold/Add"
        elif trend == "Bearish" and pl_pct < -10:
            rec = "Review/Trim"
        else:
            rec = "Hold"

        return {
            "recommendation": rec,
            "reasoning": "; ".join(reasoning) if reasoning else "Normal conditions",
            "actions": actions,
            "confidence": min(95, max(40, confidence)),
            "trend": trend,
            "rsi": round(rsi, 1)
        }

    def _option_recommendation(self, symbol: str, strategy: str, dte: int, pl_pct: float,
                               trend: str, rsi: float, position: Dict) -> Dict[str, Any]:
        """Generate recommendation for option position"""
        reasoning = []
        actions = []
        confidence = 60

        # Extract additional data from position
        greeks = position.get("greeks", {})
        delta = abs(greeks.get("delta", 0))
        theta = greeks.get("theta", 0)
        iv = greeks.get("iv", 0)
        current_price = position.get("current_price", 0)
        entry_price = position.get("avg_price", 0)

        # CSP (Cash Secured Put) recommendations
        if strategy == "CSP":
            # Profit capture analysis
            if pl_pct > 50:
                reasoning.append(f"Captured {pl_pct:.0f}% of premium - excellent")
                actions.append("BTC (buy to close) and roll to next cycle for more income")
                confidence += 20
            elif pl_pct > 25:
                reasoning.append(f"Captured {pl_pct:.0f}% of premium - good progress")
                confidence += 10

            # DTE-based recommendations
            if dte <= 7:
                reasoning.append(f"Entering expiration week ({dte} DTE)")
                if pl_pct > 50:
                    actions.append("Let expire worthless or BTC for minimal cost")
                else:
                    actions.append("Monitor closely - gamma risk increasing")
                confidence += 10
            elif dte <= 14:
                reasoning.append(f"Theta acceleration zone ({dte} DTE)")

            # Risk analysis
            if delta > 40:
                reasoning.append(f"Delta at {delta:.0f}% - elevated assignment risk")
                actions.append("Consider rolling down/out if concerned about assignment")
                confidence -= 5

            if trend == "Bearish":
                reasoning.append("Stock in downtrend - assignment probability elevated")
                actions.append("Ensure comfortable owning shares at strike")
                confidence -= 10
            elif trend == "Bullish":
                reasoning.append("Stock in uptrend - favorable for CSP")
                confidence += 10

            # IV analysis
            if iv > 80:
                reasoning.append(f"IV at {iv:.0f}% - elevated, good for premium sellers")
            elif iv < 30:
                reasoning.append(f"IV at {iv:.0f}% - low, consider waiting for vol spike")

        # CC (Covered Call) recommendations
        elif strategy == "CC":
            if pl_pct > 50:
                reasoning.append(f"Captured {pl_pct:.0f}% of premium")
                actions.append("BTC and roll up/out for more premium")
                confidence += 15

            if dte <= 7:
                if pl_pct > 75:
                    reasoning.append("Near max profit zone")
                    actions.append("Let expire or roll to capture more premium")
                else:
                    reasoning.append("Expiration week - monitor for assignment")

            if trend == "Bullish" and rsi > 65:
                reasoning.append("Strong momentum - potential early assignment risk")
                actions.append("Be prepared to roll up if called away is undesirable")
                confidence -= 5

            if delta > 60:
                reasoning.append(f"High delta ({delta:.0f}%) - likely to be called")

        # Long Call recommendations
        elif strategy == "Long Call":
            if dte <= 21:
                reasoning.append(f"Only {dte} DTE - theta decay is accelerating")
                if pl_pct > 0:
                    actions.append("Consider taking profits before decay erodes gains")
                else:
                    actions.append("Roll to later expiration to preserve value")
                confidence += 10 if dte <= 14 else 5

            if pl_pct > 100:
                reasoning.append(f"Excellent {pl_pct:.0f}% gain")
                actions.append("Lock in profits - sell half, let rest ride")
                confidence += 20
            elif pl_pct > 50:
                reasoning.append(f"Good {pl_pct:.0f}% profit")
                actions.append("Consider trailing stop or partial profit")
                confidence += 15
            elif pl_pct < -50:
                reasoning.append(f"Down {abs(pl_pct):.0f}% - significant loss")
                if dte > 30:
                    actions.append("Roll to later date to give more time")
                else:
                    actions.append("Consider cutting losses")

            if trend == "Bullish":
                reasoning.append("Stock trending favorably")
                confidence += 10
            elif trend == "Bearish":
                reasoning.append("Stock trending against position")
                confidence -= 10

        # Long Put recommendations
        elif strategy == "Long Put":
            if dte <= 21:
                reasoning.append(f"Only {dte} DTE - theta accelerating")
                actions.append("Monitor closely or roll out")

            if pl_pct > 50:
                reasoning.append(f"Good {pl_pct:.0f}% profit on protective put")
                actions.append("Consider taking profits")
            elif pl_pct < -50 and dte <= 14:
                actions.append("Position likely expiring worthless - close for salvage")

        # Determine recommendation based on all factors
        if pl_pct > 75 and dte <= 14:
            rec = "Close/Roll"
        elif dte <= 3 and delta > 30:
            rec = "Monitor Closely"
        elif pl_pct > 50 and strategy in ["CSP", "CC"]:
            rec = "Take Profits"
        elif pl_pct < -75 and "Long" in strategy:
            rec = "Cut Losses"
        elif trend == "Bullish" and strategy in ["CSP", "Long Call"]:
            rec = "Hold/Add"
        elif trend == "Bearish" and strategy == "CSP" and delta > 50:
            rec = "Review Position"
        else:
            rec = "Hold"

        return {
            "recommendation": rec,
            "reasoning": "; ".join(reasoning) if reasoning else f"{strategy} in normal range",
            "actions": actions,
            "confidence": min(95, max(40, confidence)),
            "trend": trend,
            "dte_warning": dte <= 7,
            "delta_risk": delta > 50,
            "theta_income": theta if strategy in ["CSP", "CC"] else None
        }


# Singleton
_metadata_service: Optional[MetadataService] = None


def get_metadata_service() -> MetadataService:
    global _metadata_service
    if _metadata_service is None:
        _metadata_service = MetadataService()
    return _metadata_service
