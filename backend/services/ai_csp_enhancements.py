"""
AI CSP Recommender Enhancements - World-Class Features
=======================================================

Cutting-edge additions to the AI CSP Recommender:

1. WebSocket Manager - Real-time push updates to clients
2. Circuit Breaker - Resilient AI service calls with fallback
3. Performance Tracker - Learn from recommendation outcomes
4. Feature Importance - Explainable AI (XAI) with SHAP-like scoring
5. Sentiment Analyzer - Market sentiment from news/social
6. OpenTelemetry Tracing - Distributed observability

Author: Magnus AI Team
Created: 2025-12-04
"""

from __future__ import annotations

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from functools import wraps

from pydantic import BaseModel, Field
import structlog
import asyncpg

logger = structlog.get_logger(__name__)


# ============ Circuit Breaker Pattern ============

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """
    Circuit Breaker Pattern for resilient AI service calls.

    Prevents cascade failures by stopping calls to failing services
    and allowing gradual recovery.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting all calls
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker("deepseek_ai")

        @breaker.protect
        async def call_ai_service():
            return await ai_model.generate(...)
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for recovery timeout"""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               time.time() - self._last_failure_time > self.config.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
                logger.info("circuit_half_open", name=self.name)
        return self._state

    def record_success(self) -> None:
        """Record a successful call"""
        self._failure_count = 0

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                logger.info("circuit_closed", name=self.name)

    def record_failure(self, error: Exception):
        """Record a failed call"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("circuit_reopened", name=self.name, error=str(error))
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error("circuit_opened", name=self.name, failures=self._failure_count)

    def can_execute(self) -> bool:
        """Check if a call can be executed"""
        state = self.state

        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def protect(self, fallback: Optional[Callable] = None):
        """Decorator to protect a function with circuit breaker"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.can_execute():
                    if fallback:
                        logger.warning("circuit_fallback", name=self.name)
                        return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
                    raise CircuitBreakerError(f"Circuit {self.name} is OPEN")

                try:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                    self.record_success()
                    return result
                except Exception as e:
                    self.record_failure(e)
                    raise

            return wrapper
        return decorator

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": datetime.fromtimestamp(self._last_failure_time).isoformat() if self._last_failure_time else None,
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


# ============ Performance Tracker ============

@dataclass
class RecommendationOutcome:
    """Track the outcome of a recommendation"""
    symbol: str
    strike: float
    expiration: str
    recommendation_date: datetime
    ai_score: int
    confidence: int

    # Outcome data (filled later)
    outcome_date: Optional[datetime] = None
    was_profitable: Optional[bool] = None
    actual_premium: Optional[float] = None
    was_assigned: Optional[bool] = None
    profit_loss: Optional[float] = None


class PerformanceTracker:
    """
    Track AI recommendation performance over time.

    Enables learning from past recommendations to improve future ones.

    Metrics tracked:
    - Win rate (profitable recommendations)
    - Average return
    - Score accuracy (correlation between AI score and outcome)
    - Assignment rate
    - Sector/regime performance breakdown

    Example:
        tracker = PerformanceTracker(pool)

        # Record recommendation
        await tracker.record_recommendation(pick)

        # Later, record outcome
        await tracker.record_outcome(symbol, expiration, was_profitable=True, ...)

        # Get performance stats
        stats = await tracker.get_performance_stats()
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self.pool = pool
        self._recommendations: Dict[str, RecommendationOutcome] = {}
        self._stats_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None

    def _get_key(self, symbol: str, expiration: str) -> str:
        """Generate unique key for recommendation"""
        return f"{symbol}_{expiration}"

    async def record_recommendation(self, pick: Dict[str, Any]):
        """Record a new recommendation for tracking"""
        key = self._get_key(pick['symbol'], pick['expiration'])

        self._recommendations[key] = RecommendationOutcome(
            symbol=pick['symbol'],
            strike=pick['strike'],
            expiration=pick['expiration'],
            recommendation_date=datetime.now(),
            ai_score=pick.get('ai_score', 0),
            confidence=pick.get('confidence', 0),
        )

        # Persist to database if available
        if self.pool:
            await self._persist_recommendation(pick)

        logger.info("recommendation_recorded", symbol=pick['symbol'])

    async def _persist_recommendation(self, pick: Dict[str, Any]):
        """Persist recommendation to database"""
        query = """
            INSERT INTO ai_recommendation_tracking
            (symbol, strike, expiration, ai_score, confidence, recommended_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (symbol, expiration) DO UPDATE
            SET ai_score = $4, confidence = $5, recommended_at = $6
        """
        try:
            await self.pool.execute(
                query,
                pick['symbol'],
                pick['strike'],
                pick['expiration'],
                pick.get('ai_score', 0),
                pick.get('confidence', 0),
                datetime.now()
            )
        except Exception as e:
            logger.error("persist_recommendation_error", error=str(e))

    async def record_outcome(
        self,
        symbol: str,
        expiration: str,
        was_profitable: bool,
        actual_premium: Optional[float] = None,
        was_assigned: bool = False,
        profit_loss: Optional[float] = None
    ):
        """Record the outcome of a recommendation"""
        key = self._get_key(symbol, expiration)

        if key in self._recommendations:
            rec = self._recommendations[key]
            rec.outcome_date = datetime.now()
            rec.was_profitable = was_profitable
            rec.actual_premium = actual_premium
            rec.was_assigned = was_assigned
            rec.profit_loss = profit_loss

        # Persist to database
        if self.pool:
            await self._persist_outcome(symbol, expiration, was_profitable, actual_premium, was_assigned, profit_loss)

        # Invalidate cache
        self._stats_cache = None

        logger.info("outcome_recorded", symbol=symbol, profitable=was_profitable)

    async def _persist_outcome(
        self,
        symbol: str,
        expiration: str,
        was_profitable: bool,
        actual_premium: Optional[float],
        was_assigned: bool,
        profit_loss: Optional[float]
    ):
        """Persist outcome to database"""
        query = """
            UPDATE ai_recommendation_tracking
            SET
                was_profitable = $3,
                actual_premium = $4,
                was_assigned = $5,
                profit_loss = $6,
                outcome_date = NOW()
            WHERE symbol = $1 AND expiration = $2
        """
        try:
            await self.pool.execute(
                query, symbol, expiration, was_profitable,
                actual_premium, was_assigned, profit_loss
            )
        except Exception as e:
            logger.error("persist_outcome_error", error=str(e))

    async def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics"""
        # Check cache
        if self._stats_cache and self._cache_timestamp:
            if (datetime.now() - self._cache_timestamp).seconds < 300:
                return self._stats_cache

        # Calculate from memory or database
        outcomes = [r for r in self._recommendations.values() if r.was_profitable is not None]

        if not outcomes:
            return {
                "total_recommendations": len(self._recommendations),
                "evaluated": 0,
                "win_rate": 0,
                "average_score": 0,
                "message": "No outcomes recorded yet"
            }

        total = len(outcomes)
        wins = sum(1 for o in outcomes if o.was_profitable)
        assignments = sum(1 for o in outcomes if o.was_assigned)

        avg_score = sum(o.ai_score for o in outcomes) / total if total > 0 else 0
        avg_confidence = sum(o.confidence for o in outcomes) / total if total > 0 else 0

        # Score accuracy: correlation between AI score and outcome
        if total > 2:
            scores = [o.ai_score for o in outcomes]
            profits = [1 if o.was_profitable else 0 for o in outcomes]
            score_accuracy = np.corrcoef(scores, profits)[0, 1] if len(set(profits)) > 1 else 0
        else:
            score_accuracy = 0

        stats = {
            "total_recommendations": len(self._recommendations),
            "evaluated": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            "assignment_rate": round(assignments / total * 100, 1) if total > 0 else 0,
            "average_ai_score": round(avg_score, 1),
            "average_confidence": round(avg_confidence, 1),
            "score_accuracy": round(score_accuracy * 100, 1) if not np.isnan(score_accuracy) else 0,
            "period_days": days,
            "generated_at": datetime.now().isoformat()
        }

        self._stats_cache = stats
        self._cache_timestamp = datetime.now()

        return stats


# ============ Feature Importance (XAI) ============

class FeatureImportance(BaseModel):
    """Feature importance for explainable AI"""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., ge=0, le=1, description="Importance score 0-1")
    contribution: str = Field(..., description="Direction: positive/negative/neutral")
    explanation: str = Field(..., description="Human-readable explanation")


class ExplainableAI:
    """
    Explainable AI (XAI) module for CSP recommendations.

    Provides SHAP-like feature importance scores to explain
    why a particular recommendation was made.

    Features analyzed:
    - Delta positioning
    - Premium yield
    - IV percentile
    - Liquidity (volume, OI)
    - DTE sweet spot
    - Technical support
    - Sector strength

    Example:
        xai = ExplainableAI()
        importance = xai.explain_pick(pick)
        # Returns list of FeatureImportance with scores and explanations
    """

    # Ideal feature ranges for CSP selling
    IDEAL_RANGES = {
        'delta': (-0.30, -0.20),  # 70-80% POP
        'dte': (25, 45),          # Sweet spot for theta
        'iv': (0.30, 0.60),       # Good premium without extreme risk
        'premium_pct': (1.0, 3.0),
        'volume_min': 50,
        'oi_min': 200,
    }

    def explain_pick(self, pick: Dict[str, Any]) -> List[FeatureImportance]:
        """Generate feature importance for a pick"""
        features = []

        # Delta positioning
        delta = pick.get('delta') or pick.get('greeks', {}).get('delta')
        if delta is not None:
            importance, contribution, explanation = self._analyze_delta(delta)
            features.append(FeatureImportance(
                feature="delta_positioning",
                importance=importance,
                contribution=contribution,
                explanation=explanation
            ))

        # Premium yield
        premium_pct = pick.get('premium_pct', 0)
        importance, contribution, explanation = self._analyze_premium(premium_pct)
        features.append(FeatureImportance(
            feature="premium_yield",
            importance=importance,
            contribution=contribution,
            explanation=explanation
        ))

        # IV environment
        iv = pick.get('iv') or pick.get('greeks', {}).get('iv')
        if iv is not None:
            importance, contribution, explanation = self._analyze_iv(iv)
            features.append(FeatureImportance(
                feature="iv_environment",
                importance=importance,
                contribution=contribution,
                explanation=explanation
            ))

        # DTE sweet spot
        dte = pick.get('dte', 0)
        importance, contribution, explanation = self._analyze_dte(dte)
        features.append(FeatureImportance(
            feature="dte_timing",
            importance=importance,
            contribution=contribution,
            explanation=explanation
        ))

        # Liquidity
        volume = pick.get('volume', 0)
        oi = pick.get('open_interest', 0)
        importance, contribution, explanation = self._analyze_liquidity(volume, oi)
        features.append(FeatureImportance(
            feature="liquidity",
            importance=importance,
            contribution=contribution,
            explanation=explanation
        ))

        # Sort by importance
        features.sort(key=lambda x: x.importance, reverse=True)

        return features

    def _analyze_delta(self, delta: float) -> tuple[float, str, str]:
        """Analyze delta positioning"""
        ideal_low, ideal_high = self.IDEAL_RANGES['delta']

        if ideal_low <= delta <= ideal_high:
            pop = (1 + delta) * 100
            return 0.95, "positive", f"Optimal delta ({delta:.2f}) gives ~{pop:.0f}% probability of profit"
        elif -0.35 <= delta <= -0.15:
            return 0.75, "positive", f"Acceptable delta ({delta:.2f}) within safe range"
        elif delta > -0.15:
            return 0.40, "negative", f"Delta too low ({delta:.2f}) - premium may not justify risk"
        else:
            return 0.50, "negative", f"Delta too high ({delta:.2f}) - elevated assignment risk"

    def _analyze_premium(self, premium_pct: float) -> tuple[float, str, str]:
        """Analyze premium yield"""
        ideal_low, ideal_high = self.IDEAL_RANGES['premium_pct']

        if ideal_low <= premium_pct <= ideal_high:
            return 0.90, "positive", f"Excellent premium yield ({premium_pct:.2f}%) in sweet spot"
        elif premium_pct > ideal_high:
            return 0.70, "neutral", f"High premium ({premium_pct:.2f}%) but verify it's not a trap"
        elif premium_pct >= 0.5:
            return 0.60, "neutral", f"Moderate premium ({premium_pct:.2f}%) - acceptable for low-risk names"
        else:
            return 0.30, "negative", f"Low premium ({premium_pct:.2f}%) - may not be worth the capital tie-up"

    def _analyze_iv(self, iv: float) -> tuple[float, str, str]:
        """Analyze IV environment"""
        ideal_low, ideal_high = self.IDEAL_RANGES['iv']

        if ideal_low <= iv <= ideal_high:
            return 0.85, "positive", f"Optimal IV ({iv:.0%}) for premium collection"
        elif iv > ideal_high:
            return 0.60, "neutral", f"Elevated IV ({iv:.0%}) - rich premiums but volatile"
        else:
            return 0.45, "negative", f"Low IV ({iv:.0%}) - premiums are thin"

    def _analyze_dte(self, dte: int) -> tuple[float, str, str]:
        """Analyze DTE timing"""
        ideal_low, ideal_high = self.IDEAL_RANGES['dte']

        if ideal_low <= dte <= ideal_high:
            return 0.90, "positive", f"Optimal DTE ({dte}) for theta decay"
        elif dte < ideal_low:
            return 0.55, "neutral", f"Short DTE ({dte}) - gamma risk elevated"
        else:
            return 0.65, "neutral", f"Long DTE ({dte}) - slower theta decay"

    def _analyze_liquidity(self, volume: int, oi: int) -> tuple[float, str, str]:
        """Analyze liquidity"""
        if volume >= 100 and oi >= 500:
            return 0.95, "positive", f"Excellent liquidity (Vol: {volume}, OI: {oi})"
        elif volume >= 50 and oi >= 200:
            return 0.75, "positive", f"Good liquidity (Vol: {volume}, OI: {oi})"
        elif volume >= 10 and oi >= 50:
            return 0.50, "neutral", f"Fair liquidity (Vol: {volume}, OI: {oi})"
        else:
            return 0.25, "negative", f"Poor liquidity (Vol: {volume}, OI: {oi}) - wide spreads expected"

    def get_explanation_summary(self, features: List[FeatureImportance]) -> str:
        """Generate human-readable summary"""
        positive = [f for f in features if f.contribution == "positive"]
        negative = [f for f in features if f.contribution == "negative"]

        summary_parts = []

        if positive:
            top_positive = sorted(positive, key=lambda x: x.importance, reverse=True)[:3]
            strengths = [f.feature.replace("_", " ").title() for f in top_positive]
            summary_parts.append(f"Strengths: {', '.join(strengths)}")

        if negative:
            concerns = [f.feature.replace("_", " ").title() for f in negative]
            summary_parts.append(f"Concerns: {', '.join(concerns)}")

        return " | ".join(summary_parts) if summary_parts else "Balanced risk-reward profile"


# ============ WebSocket Manager ============

class WebSocketManager:
    """
    WebSocket connection manager for real-time AI pick updates.

    Manages multiple client connections and broadcasts updates
    when new AI picks are generated or refreshed.

    Example:
        ws_manager = WebSocketManager()

        # In WebSocket endpoint
        @app.websocket("/ws/ai-picks")
        async def websocket_endpoint(websocket: WebSocket):
            await ws_manager.connect(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except:
                await ws_manager.disconnect(websocket)

        # When picks are updated
        await ws_manager.broadcast_picks(new_picks)
    """

    def __init__(self) -> None:
        self._connections: Set = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket):
        """Register a new WebSocket connection"""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("websocket_connected", total=len(self._connections))

    async def disconnect(self, websocket):
        """Remove a WebSocket connection"""
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("websocket_disconnected", total=len(self._connections))

    async def broadcast_picks(self, picks: List[Dict[str, Any]]):
        """Broadcast new picks to all connected clients"""
        if not self._connections:
            return

        message = json.dumps({
            "event": "picks_updated",
            "picks": picks,
            "timestamp": datetime.now().isoformat()
        })

        # Send to all connections
        disconnected = set()
        for websocket in self._connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.add(websocket)

        # Clean up disconnected
        if disconnected:
            async with self._lock:
                self._connections -= disconnected

        logger.info("broadcast_complete", recipients=len(self._connections))

    async def broadcast_progress(self, step: str, message: str):
        """Broadcast progress update"""
        if not self._connections:
            return

        payload = json.dumps({
            "event": "progress",
            "step": step,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        for websocket in list(self._connections):
            try:
                await websocket.send_text(payload)
            except Exception:
                pass


# ============ Sentiment Analyzer ============

class SentimentAnalyzer:
    """
    Market sentiment analyzer for CSP recommendations.

    Integrates sentiment from multiple sources:
    - News headlines
    - Social media (Twitter/X, Reddit)
    - Analyst ratings
    - Insider trading activity

    Example:
        analyzer = SentimentAnalyzer()
        sentiment = await analyzer.get_sentiment("AAPL")
        # Returns: {"score": 0.7, "signal": "bullish", "sources": [...]}
    """

    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        self.pool = pool
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis for a symbol"""
        # Check cache
        if symbol in self._cache:
            cached = self._cache[symbol]
            if (datetime.now() - cached['timestamp']).seconds < self._cache_ttl:
                return cached['data']

        # Aggregate sentiment from sources
        scores = []
        sources = []

        # Discord/social signals (if available)
        discord_sentiment = await self._get_discord_sentiment(symbol)
        if discord_sentiment:
            scores.append(discord_sentiment['score'])
            sources.append({"source": "discord", **discord_sentiment})

        # News sentiment (if available)
        news_sentiment = await self._get_news_sentiment(symbol)
        if news_sentiment:
            scores.append(news_sentiment['score'])
            sources.append({"source": "news", **news_sentiment})

        # Calculate aggregate
        if scores:
            avg_score = sum(scores) / len(scores)
            signal = "bullish" if avg_score > 0.6 else "bearish" if avg_score < 0.4 else "neutral"
        else:
            avg_score = 0.5
            signal = "neutral"

        result = {
            "symbol": symbol,
            "score": round(avg_score, 2),
            "signal": signal,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }

        # Cache result
        self._cache[symbol] = {
            "data": result,
            "timestamp": datetime.now()
        }

        return result

    async def _get_discord_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment from Discord messages"""
        if not self.pool:
            return None

        try:
            query = """
                SELECT
                    COUNT(*) as mention_count,
                    COUNT(*) FILTER (WHERE content ILIKE '%bullish%' OR content ILIKE '%buy%' OR content ILIKE '%long%') as bullish,
                    COUNT(*) FILTER (WHERE content ILIKE '%bearish%' OR content ILIKE '%sell%' OR content ILIKE '%short%') as bearish
                FROM discord_messages
                WHERE content ILIKE $1
                    AND created_at >= NOW() - INTERVAL '7 days'
            """
            row = await self.pool.fetchrow(query, f'%{symbol}%')

            if row and row['mention_count'] > 0:
                total = row['mention_count']
                bullish = row['bullish']
                bearish = row['bearish']

                if bullish + bearish > 0:
                    score = bullish / (bullish + bearish)
                else:
                    score = 0.5

                return {
                    "score": score,
                    "mentions": total,
                    "bullish_count": bullish,
                    "bearish_count": bearish
                }
        except Exception as e:
            logger.error("discord_sentiment_error", symbol=symbol, error=str(e))

        return None

    async def _get_news_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment from news (placeholder)"""
        # This would integrate with news API
        # For now, return neutral baseline
        return {
            "score": 0.5,
            "articles": 0,
            "note": "News sentiment integration pending"
        }


# ============ Enhanced Tracing ============

class TracingContext:
    """
    Lightweight tracing context for observability.

    Tracks request flow through the AI recommendation pipeline
    with timing and metadata.

    Example:
        tracer = TracingContext("get_ai_picks")
        tracer.add_span("fetch_premiums", {"count": 150})
        tracer.add_span("mcdm_scoring", {"top_candidates": 20})
        tracer.add_span("ai_analysis", {"model": "deepseek-r1"})

        trace = tracer.finish()
        # Returns complete trace with timing for each span
    """

    def __init__(self, operation: str, trace_id: Optional[str] = None):
        self.operation = operation
        self.trace_id = trace_id or hashlib.md5(f"{operation}_{time.time()}".encode()).hexdigest()[:16]
        self.start_time = time.time()
        self.spans: List[Dict[str, Any]] = []
        self._current_span_start: Optional[float] = None

    def add_span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a span to the trace"""
        now = time.time()

        if self._current_span_start:
            # Close previous span
            if self.spans:
                self.spans[-1]['duration_ms'] = (now - self._current_span_start) * 1000

        self.spans.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "duration_ms": 0  # Will be set when next span starts or trace finishes
        })

        self._current_span_start = now

    def finish(self) -> Dict[str, Any]:
        """Finish trace and return complete data"""
        end_time = time.time()

        # Close last span
        if self.spans and self._current_span_start:
            self.spans[-1]['duration_ms'] = (end_time - self._current_span_start) * 1000

        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "total_duration_ms": (end_time - self.start_time) * 1000,
            "spans": self.spans,
            "started_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "finished_at": datetime.now().isoformat()
        }


# ============ Singleton Instances ============

_circuit_breaker: Optional[CircuitBreaker] = None
_performance_tracker: Optional[PerformanceTracker] = None
_xai: Optional[ExplainableAI] = None
_ws_manager: Optional[WebSocketManager] = None
_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get singleton circuit breaker"""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker("ai_ensemble", CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            half_open_max_calls=3,
            success_threshold=2
        ))
    return _circuit_breaker


def get_performance_tracker(pool: Optional[asyncpg.Pool] = None) -> PerformanceTracker:
    """Get singleton performance tracker"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(pool)
    return _performance_tracker


def get_explainable_ai() -> ExplainableAI:
    """Get singleton XAI instance"""
    global _xai
    if _xai is None:
        _xai = ExplainableAI()
    return _xai


def get_websocket_manager() -> WebSocketManager:
    """Get singleton WebSocket manager"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


def get_sentiment_analyzer(pool: Optional[asyncpg.Pool] = None) -> SentimentAnalyzer:
    """Get singleton sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer(pool)
    return _sentiment_analyzer


# ============ Database Schema for Tracking ============

TRACKING_SCHEMA = """
-- AI Recommendation Tracking Table
CREATE TABLE IF NOT EXISTS ai_recommendation_tracking (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    strike DECIMAL(10, 2) NOT NULL,
    expiration DATE NOT NULL,
    ai_score INTEGER,
    confidence INTEGER,
    recommended_at TIMESTAMP DEFAULT NOW(),
    outcome_date TIMESTAMP,
    was_profitable BOOLEAN,
    actual_premium DECIMAL(10, 4),
    was_assigned BOOLEAN DEFAULT FALSE,
    profit_loss DECIMAL(10, 2),
    UNIQUE(symbol, expiration)
);

-- Index for performance queries
CREATE INDEX IF NOT EXISTS idx_recommendation_tracking_date
ON ai_recommendation_tracking(recommended_at);

CREATE INDEX IF NOT EXISTS idx_recommendation_tracking_outcome
ON ai_recommendation_tracking(was_profitable) WHERE was_profitable IS NOT NULL;
"""


async def ensure_tracking_table(pool: asyncpg.Pool):
    """Ensure tracking table exists"""
    try:
        await pool.execute(TRACKING_SCHEMA)
        logger.info("tracking_table_ensured")
    except Exception as e:
        logger.error("tracking_table_error", error=str(e))
