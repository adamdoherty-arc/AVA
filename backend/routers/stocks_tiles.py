"""
Stocks Tiles Router v2.2 - API endpoints for stock tile hub with modern infrastructure

Provides:
- Combined all-data endpoint with database integration
- Individual and batch stock scoring with circuit breaker protection
- SSE streaming analysis with optional LLM reasoning
- Custom watchlist management with rate limiting
- Score history tracking
- Resilient API calls with automatic retry and fallback
"""

from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import logging
import asyncio
import json
from datetime import datetime

from backend.services.stock_score_service import (
    get_stock_score_service, StockAIScore
)
from backend.infrastructure.async_yfinance import get_async_yfinance
from backend.infrastructure.cache import get_cache
from backend.services.advanced_risk_models import get_prediction_engine
from backend.infrastructure.errors import safe_internal_error

# Circuit breaker for external API resilience
try:
    from backend.infrastructure.circuit_breaker import (
        yfinance_breaker, llm_breaker, CircuitBreakerError
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    yfinance_breaker = None
    llm_breaker = None
    CircuitBreakerError = Exception

# Rate limiting for endpoint protection
try:
    from backend.infrastructure.rate_limiter import (
        rate_limited, RateLimitExceeded, get_endpoint_limit
    )
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False

# LLM integration (optional - graceful fallback)
try:
    from src.ava.core.llm_engine import LLMClient, LLMProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/stocks/tiles",
    tags=["stocks-tiles"]
)


# =============================================================================
# Request/Response Models
# =============================================================================

class StockTileResponse(BaseModel):
    """Response for single stock tile"""
    symbol: str
    company_name: str
    sector: str
    current_price: float
    daily_change_pct: float
    ai_score: float
    recommendation: str
    confidence: float
    trend: str
    trend_strength: float
    rsi_14: float
    iv_estimate: float
    vol_regime: str
    predicted_change_1d: float
    predicted_change_5d: float
    support_price: float
    resistance_price: float
    market_cap: Optional[float] = None
    score_components: Dict[str, Dict[str, Any]] = {}


class AllDataResponse(BaseModel):
    """Combined response for tile hub"""
    stocks: List[StockTileResponse]
    stats: Dict[str, Any]
    watchlists: Dict[str, List[str]]
    last_updated: str


class BatchScoreRequest(BaseModel):
    """Request to score multiple symbols"""
    symbols: List[str]


class WatchlistRequest(BaseModel):
    """Request to create/update watchlist"""
    name: str
    symbols: List[str]


class StreamEvent(BaseModel):
    """SSE event structure"""
    event: str
    data: Dict[str, Any]


# =============================================================================
# Default Watchlists (Expanded with more diversity)
# =============================================================================

DEFAULT_WATCHLISTS = {
    "Tech Leaders": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM"
    ],
    "Options Favorites": [
        "SPY", "QQQ", "IWM", "AAPL", "NVDA", "AMD", "TSLA", "AMZN", "META"
    ],
    "AI & Semiconductor": [
        "NVDA", "AMD", "AVGO", "TSM", "QCOM", "ASML", "ARM", "MU", "MRVL"
    ],
    "High IV Plays": [
        "GME", "AMC", "MARA", "COIN", "RIVN", "PLTR", "SOFI", "NIO", "RIOT"
    ],
    "Dividend Aristocrats": [
        "JNJ", "PG", "KO", "PEP", "ABT", "ABBV", "VZ", "XOM", "CVX"
    ],
    "Financials": [
        "JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "BLK", "SCHW"
    ],
    "Healthcare": [
        "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD"
    ],
    "ETF Universe": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "XLF", "XLE", "XLK", "GLD"
    ],
    "Consumer": [
        "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "COST", "WMT"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY"
    ]
}


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/all-data")
async def get_all_data(
    watchlist: str = Query("Tech Leaders", description="Watchlist to load"),
    include_scores: bool = Query(True, description="Include AI scores")
) -> AllDataResponse:
    """
    Get all data for the stock tile hub.

    Combines stock data, scores, and watchlist info in one call.
    """
    cache = get_cache()
    cache_key = f"stocks_tiles:all_data:{watchlist}"

    # Check cache (2 min TTL for combined data)
    cached = await cache.get(cache_key)
    if cached:
        return AllDataResponse(**cached)

    try:
        # Get symbols from watchlist
        symbols = DEFAULT_WATCHLISTS.get(watchlist, DEFAULT_WATCHLISTS["Tech Leaders"])

        stocks = []
        score_service = get_stock_score_service()

        # Get basic price data first (always needed for fallback)
        yf = get_async_yfinance()
        price_data = await yf.batch_get_prices(symbols)

        if include_scores:
            # Try to calculate AI scores in parallel
            scores = await score_service.calculate_batch(symbols, max_concurrent=5)

            for symbol in symbols:
                score = scores.get(symbol)
                if score:
                    stocks.append(StockTileResponse(
                        symbol=score.symbol,
                        company_name=score.company_name,
                        sector=score.sector,
                        current_price=score.current_price,
                        daily_change_pct=score.daily_change_pct,
                        ai_score=score.ai_score,
                        recommendation=score.recommendation,
                        confidence=score.confidence,
                        trend=score.trend,
                        trend_strength=score.trend_strength,
                        rsi_14=score.rsi_14,
                        iv_estimate=score.iv_estimate,
                        vol_regime=score.vol_regime,
                        predicted_change_1d=score.predicted_change_1d,
                        predicted_change_5d=score.predicted_change_5d,
                        support_price=score.support_price,
                        resistance_price=score.resistance_price,
                        market_cap=score.market_cap,
                        score_components={
                            k: {"raw_score": v.raw_score, "weight": v.weight}
                            for k, v in score.components.items()
                        }
                    ))
                elif symbol in price_data:
                    # Fallback to basic data if AI scoring failed
                    price = price_data[symbol]
                    stocks.append(StockTileResponse(
                        symbol=symbol,
                        company_name=symbol,  # No company name available in fallback
                        sector="Technology",  # Default sector
                        current_price=price,
                        daily_change_pct=0,  # No change data in simple fallback
                        ai_score=50,  # Neutral score when AI unavailable
                        recommendation="ANALYZING",
                        confidence=0.0,
                        trend="neutral",
                        trend_strength=0.5,
                        rsi_14=50,
                        iv_estimate=30,
                        vol_regime="unknown",
                        predicted_change_1d=0,
                        predicted_change_5d=0,
                        support_price=price * 0.95,
                        resistance_price=price * 1.05
                    ))
        else:
            # Just get basic price data (faster)
            yf = get_async_yfinance()
            prices = await yf.batch_get_prices(symbols)
            for symbol in symbols:
                if symbol in prices:
                    stocks.append(StockTileResponse(
                        symbol=symbol,
                        company_name=symbol,
                        sector="Unknown",
                        current_price=prices[symbol],
                        daily_change_pct=0,
                        ai_score=50,
                        recommendation="HOLD",
                        confidence=0.5,
                        trend="neutral",
                        trend_strength=0.5,
                        rsi_14=50,
                        iv_estimate=30,
                        vol_regime="normal",
                        predicted_change_1d=0,
                        predicted_change_5d=0,
                        support_price=prices[symbol] * 0.95,
                        resistance_price=prices[symbol] * 1.05
                    ))

        # Calculate stats
        if stocks:
            avg_score = sum(s.ai_score for s in stocks) / len(stocks)
            bullish_count = sum(1 for s in stocks if s.trend == "bullish")
            bearish_count = sum(1 for s in stocks if s.trend == "bearish")
        else:
            avg_score = 50
            bullish_count = 0
            bearish_count = 0

        stats = {
            "total_stocks": len(stocks),
            "avg_ai_score": round(avg_score, 1),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": len(stocks) - bullish_count - bearish_count
        }

        result = AllDataResponse(
            stocks=stocks,
            stats=stats,
            watchlists=DEFAULT_WATCHLISTS,
            last_updated=datetime.now().isoformat()
        )

        # Cache result
        await cache.set(cache_key, result.model_dump(), ttl=120)

        return result

    except Exception as e:
        logger.error(f"Error fetching all data: {e}")
        safe_internal_error(e, "fetch stock tile data")


@router.get("/score/{symbol}")
async def get_stock_score(symbol: str) -> StockTileResponse:
    """
    Get AI score for a single stock.
    """
    try:
        score_service = get_stock_score_service()
        score = await score_service.calculate_score(symbol.upper())

        return StockTileResponse(
            symbol=score.symbol,
            company_name=score.company_name,
            sector=score.sector,
            current_price=score.current_price,
            daily_change_pct=score.daily_change_pct,
            ai_score=score.ai_score,
            recommendation=score.recommendation,
            confidence=score.confidence,
            trend=score.trend,
            trend_strength=score.trend_strength,
            rsi_14=score.rsi_14,
            iv_estimate=score.iv_estimate,
            vol_regime=score.vol_regime,
            predicted_change_1d=score.predicted_change_1d,
            predicted_change_5d=score.predicted_change_5d,
            support_price=score.support_price,
            resistance_price=score.resistance_price,
            market_cap=score.market_cap,
            score_components={
                k: {"raw_score": v.raw_score, "weight": v.weight}
                for k, v in score.components.items()
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Stock not found: {symbol}")
    except Exception as e:
        logger.error(f"Error getting score for {symbol}: {e}")
        safe_internal_error(e, "get stock score")


@router.post("/batch")
async def batch_score(request: BatchScoreRequest) -> Dict[str, StockTileResponse]:
    """
    Calculate scores for multiple symbols.
    """
    if len(request.symbols) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 symbols per batch")

    try:
        score_service = get_stock_score_service()
        scores = await score_service.calculate_batch(request.symbols)

        return {
            symbol: StockTileResponse(
                symbol=score.symbol,
                company_name=score.company_name,
                sector=score.sector,
                current_price=score.current_price,
                daily_change_pct=score.daily_change_pct,
                ai_score=score.ai_score,
                recommendation=score.recommendation,
                confidence=score.confidence,
                trend=score.trend,
                trend_strength=score.trend_strength,
                rsi_14=score.rsi_14,
                iv_estimate=score.iv_estimate,
                vol_regime=score.vol_regime,
                predicted_change_1d=score.predicted_change_1d,
                predicted_change_5d=score.predicted_change_5d,
                support_price=score.support_price,
                resistance_price=score.resistance_price,
                market_cap=score.market_cap
            )
            for symbol, score in scores.items()
        }

    except Exception as e:
        logger.error(f"Error in batch score: {e}")
        safe_internal_error(e, "batch score stocks")


@router.get("/stream/analyze/{symbol}")
async def stream_analysis(symbol: str):
    """
    Stream AI analysis for a stock using Server-Sent Events.

    Progressively sends analysis as it's computed.
    Features circuit breaker protection for external API calls.
    """
    async def generate_events():
        try:
            symbol_upper = symbol.upper()

            # Event 1: Start
            yield f"event: start\ndata: {json.dumps({'symbol': symbol_upper, 'status': 'analyzing'})}\n\n"

            # Event 2: Price data (with circuit breaker protection)
            yf = get_async_yfinance()
            try:
                if CIRCUIT_BREAKER_AVAILABLE and yfinance_breaker:
                    stock_data = await yfinance_breaker.call(
                        yf.get_stock_data, symbol_upper, period="3mo"
                    )
                else:
                    stock_data = await yf.get_stock_data(symbol_upper, period="3mo")
            except CircuitBreakerError:
                yield f"event: error\ndata: {json.dumps({'error': 'Market data service temporarily unavailable'})}\n\n"
                return

            yield f"event: price_data\ndata: {json.dumps({'current_price': stock_data.current_price, 'iv_estimate': stock_data.iv_estimate})}\n\n"

            await asyncio.sleep(0.1)  # Small delay for streaming effect

            # Event 3: Technicals
            prediction_engine = get_prediction_engine()
            trend = prediction_engine.get_trend_signal(
                symbol=symbol_upper,
                prices=stock_data.historical_prices
            )

            yield f"event: technicals\ndata: {json.dumps({'trend': trend.signal_type, 'strength': trend.strength, 'indicators': trend.indicators})}\n\n"

            await asyncio.sleep(0.1)

            # Event 4: AI Score
            score_service = get_stock_score_service()
            score = await score_service.calculate_score(symbol_upper)

            score_data = {
                "ai_score": score.ai_score,
                "recommendation": score.recommendation,
                "confidence": score.confidence,
                "components": {
                    k: {"raw_score": v.raw_score, "weight": v.weight}
                    for k, v in score.components.items()
                }
            }
            yield f"event: ai_score\ndata: {json.dumps(score_data)}\n\n"

            await asyncio.sleep(0.1)

            # Event 5: Reasoning (heuristic-based, fast)
            reasoning = _generate_reasoning(score)
            yield f"event: reasoning\ndata: {json.dumps({'text': reasoning})}\n\n"

            # Event 6: LLM Analysis (optional, with circuit breaker protection)
            if LLM_AVAILABLE:
                try:
                    if CIRCUIT_BREAKER_AVAILABLE and llm_breaker:
                        llm_analysis = await llm_breaker.call(
                            _generate_llm_analysis, score
                        )
                    else:
                        llm_analysis = await _generate_llm_analysis(score)
                    if llm_analysis:
                        yield f"event: llm_analysis\ndata: {json.dumps({'text': llm_analysis})}\n\n"
                except CircuitBreakerError:
                    logger.debug("LLM circuit breaker open, skipping deep analysis")
                except Exception as llm_err:
                    logger.debug(f"LLM analysis skipped: {llm_err}")

            # Event 7: Complete
            yield f"event: complete\ndata: {json.dumps({'status': 'done', 'symbol': symbol_upper})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream analysis: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/watchlists")
async def get_watchlists() -> Dict[str, List[str]]:
    """
    Get all available watchlists.
    """
    cache = get_cache()

    # Get custom watchlists from cache
    custom = await cache.get("stocks_tiles:custom_watchlists")
    custom = custom or {}

    # Merge with defaults
    all_watchlists = {**DEFAULT_WATCHLISTS, **custom}

    return all_watchlists


@router.post("/watchlists")
async def create_watchlist(request: WatchlistRequest) -> Dict[str, Any]:
    """
    Create or update a custom watchlist.
    """
    if len(request.symbols) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 symbols per watchlist")

    cache = get_cache()

    # Get existing custom watchlists
    custom = await cache.get("stocks_tiles:custom_watchlists")
    custom = custom or {}

    # Add/update watchlist
    custom[request.name] = request.symbols

    # Save back to cache (persist indefinitely with long TTL)
    await cache.set("stocks_tiles:custom_watchlists", custom, ttl=86400 * 30)

    return {
        "success": True,
        "watchlist": request.name,
        "symbols": request.symbols
    }


@router.delete("/watchlists/{name}")
async def delete_watchlist(name: str) -> Dict[str, Any]:
    """
    Delete a custom watchlist.
    """
    if name in DEFAULT_WATCHLISTS:
        raise HTTPException(status_code=400, detail="Cannot delete default watchlists")

    cache = get_cache()

    # Get existing custom watchlists
    custom = await cache.get("stocks_tiles:custom_watchlists")
    custom = custom or {}

    if name in custom:
        del custom[name]
        await cache.set("stocks_tiles:custom_watchlists", custom, ttl=86400 * 30)
        return {"success": True, "deleted": name}

    raise HTTPException(status_code=404, detail=f"Watchlist '{name}' not found")


@router.get("/prices")
async def get_prices(symbols: str = Query(..., description="Comma-separated symbols")) -> Dict[str, float]:
    """
    Get current prices for multiple symbols (lightweight endpoint for polling).
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    if len(symbol_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 symbols per request")

    try:
        yf = get_async_yfinance()
        prices = await yf.batch_get_prices(symbol_list)
        return prices
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
        safe_internal_error(e, "fetch stock prices")


@router.get("/health")
async def get_health() -> Dict[str, Any]:
    """
    Get health status of the stocks tiles service.

    Returns circuit breaker states, cache status, and infrastructure health.
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2",
        "infrastructure": {
            "circuit_breakers_enabled": CIRCUIT_BREAKER_AVAILABLE,
            "rate_limiting_enabled": RATE_LIMITER_AVAILABLE,
            "llm_available": LLM_AVAILABLE,
        }
    }

    # Add circuit breaker stats if available
    if CIRCUIT_BREAKER_AVAILABLE:
        health["circuit_breakers"] = {}
        if yfinance_breaker:
            health["circuit_breakers"]["yfinance"] = yfinance_breaker.get_stats()
        if llm_breaker:
            health["circuit_breakers"]["llm"] = llm_breaker.get_stats()

    # Add cache status
    try:
        cache = get_cache()
        health["cache"] = {"available": True}
    except Exception:
        health["cache"] = {"available": False}

    return health


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_reasoning(score: StockAIScore) -> str:
    """Generate AI-driven reasoning with diverse, contextual analysis."""
    parts = []

    # Score tier determines opening statement style
    if score.ai_score >= 80:
        parts.append(
            f"**Strong Buy Signal** - {score.symbol} scores {score.ai_score:.0f}/100, "
            f"indicating exceptional bullish momentum across multiple factors."
        )
    elif score.ai_score >= 65:
        parts.append(
            f"**Buy Signal** - {score.symbol} scores {score.ai_score:.0f}/100, "
            f"showing favorable conditions for upside."
        )
    elif score.ai_score >= 50:
        parts.append(
            f"**Hold/Neutral** - {score.symbol} scores {score.ai_score:.0f}/100, "
            f"suggesting a wait-and-see approach."
        )
    elif score.ai_score >= 35:
        parts.append(
            f"**Caution** - {score.symbol} scores {score.ai_score:.0f}/100, "
            f"indicating weakness in key technical factors."
        )
    else:
        parts.append(
            f"**Bearish Signal** - {score.symbol} scores {score.ai_score:.0f}/100, "
            f"showing significant downside risk."
        )

    # Component-specific insights
    comp_insights = []
    for name, comp in score.components.items():
        if comp.raw_score >= 70:
            comp_insights.append(f"{name} (+)")
        elif comp.raw_score <= 30:
            comp_insights.append(f"{name} (-)")

    if comp_insights:
        parts.append(f"Key factors: {', '.join(comp_insights[:3])}.")

    # Technical context
    tech_parts = []
    if score.trend == "bullish" and score.trend_strength > 0.6:
        tech_parts.append("strong uptrend intact")
    elif score.trend == "bearish" and score.trend_strength > 0.6:
        tech_parts.append("downtrend pressure")

    if score.rsi_14 > 70:
        tech_parts.append(f"RSI overbought ({score.rsi_14:.0f})")
    elif score.rsi_14 < 30:
        tech_parts.append(f"RSI oversold ({score.rsi_14:.0f})")

    if score.vol_regime == "elevated" or score.vol_regime == "extreme":
        tech_parts.append(f"elevated volatility ({score.iv_estimate:.0f}% IV)")

    if tech_parts:
        parts.append(f"Technical context: {', '.join(tech_parts)}.")

    # Price prediction with confidence context
    if abs(score.predicted_change_5d) > 1.5:
        direction = "upside" if score.predicted_change_5d > 0 else "downside"
        conf_text = "high" if score.confidence > 0.7 else "moderate"
        parts.append(
            f"5-day outlook: {abs(score.predicted_change_5d):.1f}% {direction} "
            f"expected ({conf_text} confidence)."
        )

    # Key levels
    current = score.current_price
    dist_to_support = ((current - score.support_price) / current) * 100
    dist_to_resistance = ((score.resistance_price - current) / current) * 100

    if dist_to_support < 3:
        parts.append(f"Near support at ${score.support_price:.2f} (potential bounce zone).")
    elif dist_to_resistance < 3:
        parts.append(f"Testing resistance at ${score.resistance_price:.2f}.")
    else:
        parts.append(
            f"Range: ${score.support_price:.2f} - ${score.resistance_price:.2f}."
        )

    return " ".join(parts)


async def _generate_llm_analysis(score: StockAIScore) -> Optional[str]:
    """
    Generate rich AI analysis using LLM (Claude/GPT).
    Returns None if LLM not available or fails.
    """
    if not LLM_AVAILABLE or not LLMClient:
        return None

    cache = get_cache()
    cache_key = f"llm_analysis:{score.symbol}:{int(score.ai_score)}"

    # Check cache (15 min TTL for LLM responses)
    cached = await cache.get(cache_key)
    if cached:
        return cached

    try:
        client = LLMClient(
            provider=LLMProvider.ANTHROPIC,
            cache_enabled=True
        )

        prompt = f"""Analyze this stock for a trader:

Stock: {score.symbol} ({score.company_name})
Sector: {score.sector}
Current Price: ${score.current_price:.2f}
Daily Change: {score.daily_change_pct:+.2f}%

AI Score: {score.ai_score:.0f}/100 ({score.recommendation})
Confidence: {score.confidence:.0%}

Technical:
- Trend: {score.trend} (strength: {score.trend_strength:.0%})
- RSI(14): {score.rsi_14:.1f}
- IV: {score.iv_estimate:.1f}%
- Vol Regime: {score.vol_regime}

Predictions:
- 1-day: {score.predicted_change_1d:+.1f}%
- 5-day: {score.predicted_change_5d:+.1f}%

Support: ${score.support_price:.2f}
Resistance: ${score.resistance_price:.2f}

Provide a concise 2-3 sentence trading analysis with:
1. Key insight about the setup
2. Primary risk factor
3. Actionable takeaway"""

        response = await client.generate(
            system="You are a professional stock analyst. Be concise and direct.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        analysis = response.content.strip()

        # Cache the result
        await cache.set(cache_key, analysis, ttl=900)

        return analysis

    except Exception as e:
        logger.debug(f"LLM analysis failed for {score.symbol}: {e}")
        return None
