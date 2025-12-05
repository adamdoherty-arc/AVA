"""
Advanced Portfolio Analytics Service
Modern AI-powered risk metrics, multi-agent consensus, and real-time insights
"""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class SignalStrength(Enum):
    STRONG_SELL = -2
    SELL = -1
    NEUTRAL = 0
    BUY = 1
    STRONG_BUY = 2


@dataclass
class GreeksExposure:
    """Aggregated Greeks exposure across portfolio"""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0  # Daily theta income/decay
    net_vega: float = 0.0
    weighted_iv: float = 0.0
    delta_dollars: float = 0.0  # Dollar exposure per 1% move


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # Greeks Exposure
    greeks: GreeksExposure = field(default_factory=GreeksExposure)

    # Value at Risk
    var_1d_95: float = 0.0  # 1-day 95% VaR
    var_1d_99: float = 0.0  # 1-day 99% VaR
    max_loss_scenario: float = 0.0  # Worst case loss

    # Concentration Risk
    largest_position_pct: float = 0.0
    top_3_concentration: float = 0.0
    sector_concentration: Dict[str, float] = field(default_factory=dict)

    # Options-Specific Risk
    gamma_risk_score: float = 0.0  # 0-100, higher = more gamma exposure
    assignment_risk_count: int = 0  # Positions at high assignment risk
    expiring_this_week: int = 0
    expiring_next_week: int = 0

    # Overall Risk Score
    portfolio_risk_score: float = 0.0  # 0-100
    risk_level: RiskLevel = RiskLevel.MODERATE


@dataclass
class ProbabilityMetrics:
    """Probability-based analytics"""
    portfolio_pop: float = 0.0  # Probability of profit
    expected_value: float = 0.0  # Expected P/L
    sharpe_estimate: float = 0.0  # Risk-adjusted return estimate
    theta_efficiency: float = 0.0  # Theta capture efficiency


@dataclass
class AIConsensus:
    """Multi-agent consensus recommendation"""
    action: str
    confidence: float
    urgency: str
    agents_agree: int
    agents_total: int
    dissenting_opinions: List[str] = field(default_factory=list)
    key_factors: List[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class PositionAlert:
    """Proactive position alert"""
    symbol: str
    alert_type: str  # "expiring", "assignment_risk", "stop_loss", "take_profit", "iv_crush"
    severity: str  # "info", "warning", "critical"
    message: str
    action_required: str
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedPortfolioAnalytics:
    """
    Advanced AI-powered portfolio analytics engine

    Features:
    - Real-time risk metrics (VaR, Greeks exposure, concentration)
    - Multi-agent consensus system
    - Probability analysis (PoP, expected value)
    - Proactive position alerts
    - IV rank/percentile analysis
    - Streaming insights for real-time updates
    """

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._alert_history: List[PositionAlert] = []

    async def analyze_portfolio_risk(
        self,
        positions: Dict[str, Any],
        metadata_cache: Optional[Dict[str, Any]] = None
    ) -> RiskMetrics:
        """
        Comprehensive risk analysis of portfolio

        Args:
            positions: Portfolio positions data
            metadata_cache: Optional cached metadata for symbols

        Returns:
            Complete risk metrics
        """
        stocks = positions.get("stocks", [])
        options = positions.get("options", [])
        summary = positions.get("summary", {})

        total_equity = summary.get("total_equity", 0)
        if total_equity == 0:
            return RiskMetrics()

        metrics = RiskMetrics()

        # Calculate Greeks exposure
        metrics.greeks = await self._calculate_greeks_exposure(options, total_equity)

        # Calculate VaR (simplified historical approach)
        metrics.var_1d_95, metrics.var_1d_99 = self._calculate_var(
            stocks, options, total_equity
        )

        # Concentration risk
        all_positions = stocks + options
        position_values = [abs(p.get("current_value", 0)) for p in all_positions]
        total_value = sum(position_values)

        if total_value > 0:
            sorted_values = sorted(position_values, reverse=True)
            metrics.largest_position_pct = (sorted_values[0] / total_value) * 100 if sorted_values else 0
            metrics.top_3_concentration = (sum(sorted_values[:3]) / total_value) * 100

        # Sector concentration (if metadata available)
        if metadata_cache:
            sector_values: Dict[str, float] = {}
            for pos in stocks:
                symbol = pos.get("symbol", "")
                meta = metadata_cache.get(symbol, {})
                sector = meta.get("sector", "Unknown")
                sector_values[sector] = sector_values.get(sector, 0) + pos.get("current_value", 0)
            if total_value > 0:
                metrics.sector_concentration = {
                    k: (v / total_value) * 100 for k, v in sector_values.items()
                }

        # Options-specific risk
        metrics.gamma_risk_score = self._calculate_gamma_risk(options)
        metrics.assignment_risk_count = sum(
            1 for o in options
            if o.get("dte", 999) <= 7 and abs(o.get("greeks", {}).get("delta", 0)) > 40
        )
        metrics.expiring_this_week = sum(1 for o in options if o.get("dte", 999) <= 7)
        metrics.expiring_next_week = sum(1 for o in options if 7 < o.get("dte", 999) <= 14)

        # Calculate max loss scenario
        metrics.max_loss_scenario = self._calculate_max_loss(options)

        # Overall risk score (0-100)
        metrics.portfolio_risk_score = self._calculate_risk_score(metrics)
        metrics.risk_level = self._determine_risk_level(metrics.portfolio_risk_score)

        return metrics

    async def _calculate_greeks_exposure(
        self,
        options: List[Dict],
        total_equity: float
    ) -> GreeksExposure:
        """Calculate aggregated Greeks exposure"""
        exposure = GreeksExposure()

        for opt in options:
            greeks = opt.get("greeks", {})
            qty = opt.get("quantity", 0)
            is_short = opt.get("type") == "short"

            # Delta (already adjusted for short positions in portfolio_service)
            delta = greeks.get("delta", 0) / 100  # Convert from percentage
            exposure.net_delta += delta * qty

            # Gamma
            gamma = greeks.get("gamma", 0)
            exposure.net_gamma += gamma * qty * (-1 if is_short else 1)

            # Theta (already positive for shorts in portfolio_service)
            theta = greeks.get("theta", 0)
            exposure.net_theta += theta * qty

            # Vega
            vega = greeks.get("vega", 0)
            exposure.net_vega += vega * qty * (-1 if is_short else 1)

            # IV weighting
            iv = greeks.get("iv", 0)
            value = abs(opt.get("current_value", 0))
            exposure.weighted_iv += iv * value

        # Normalize weighted IV
        total_option_value = sum(abs(o.get("current_value", 0)) for o in options)
        if total_option_value > 0:
            exposure.weighted_iv /= total_option_value

        # Delta dollars (exposure per 1% underlying move)
        exposure.delta_dollars = exposure.net_delta * total_equity * 0.01

        return exposure

    def _calculate_var(
        self,
        stocks: List[Dict],
        options: List[Dict],
        total_equity: float
    ) -> tuple:
        """
        Calculate Value at Risk using parametric method

        Returns:
            (VaR_95%, VaR_99%) as dollar amounts
        """
        if total_equity == 0:
            return 0.0, 0.0

        # Estimate portfolio volatility (simplified)
        # Use weighted average of implied volatilities for options
        # and assume 20% vol for stocks as baseline
        stock_value = sum(s.get("current_value", 0) for s in stocks)
        option_value = sum(abs(o.get("current_value", 0)) for o in options)

        # Stock volatility assumption (annualized)
        stock_vol = 0.20  # 20% baseline

        # Option volatility (use weighted IV)
        if options:
            weighted_iv = sum(
                o.get("greeks", {}).get("iv", 30) * abs(o.get("current_value", 0))
                for o in options
            )
            if option_value > 0:
                option_vol = (weighted_iv / option_value) / 100
            else:
                option_vol = 0.30
        else:
            option_vol = 0

        # Portfolio volatility (weighted)
        total_value = stock_value + option_value
        if total_value > 0:
            portfolio_vol = (
                (stock_value / total_value) * stock_vol +
                (option_value / total_value) * option_vol
            )
        else:
            portfolio_vol = 0.20

        # Daily volatility
        daily_vol = portfolio_vol / math.sqrt(252)

        # VaR calculation (parametric)
        var_95 = total_equity * daily_vol * 1.645  # 95% confidence
        var_99 = total_equity * daily_vol * 2.326  # 99% confidence

        return round(var_95, 2), round(var_99, 2)

    def _calculate_gamma_risk(self, options: List[Dict]) -> float:
        """
        Calculate gamma risk score (0-100)

        High gamma = high risk of delta changing rapidly
        """
        if not options:
            return 0.0

        gamma_scores = []
        for opt in options:
            dte = opt.get("dte", 999)
            gamma = abs(opt.get("greeks", {}).get("gamma", 0))
            delta = abs(opt.get("greeks", {}).get("delta", 0))

            # Gamma risk increases as:
            # 1. DTE decreases (especially < 7 days)
            # 2. Delta near 50 (ATM)
            # 3. High gamma value

            dte_factor = max(0, (21 - dte) / 21) if dte < 21 else 0
            atm_factor = 1 - abs(delta - 50) / 50  # Max at delta=50
            gamma_factor = min(gamma * 10, 1)  # Normalize gamma

            position_gamma_risk = (dte_factor * 0.4 + atm_factor * 0.3 + gamma_factor * 0.3) * 100
            gamma_scores.append(position_gamma_risk)

        return round(sum(gamma_scores) / len(gamma_scores), 1) if gamma_scores else 0.0

    def _calculate_max_loss(self, options: List[Dict]) -> float:
        """Calculate maximum potential loss scenario"""
        max_loss = 0.0

        for opt in options:
            opt_type = opt.get("type")
            option_type = opt.get("option_type")
            strike = opt.get("strike", 0)
            qty = opt.get("quantity", 0)
            current_value = opt.get("current_value", 0)

            if opt_type == "short":
                if option_type == "put":
                    # Short put: max loss = strike * 100 * qty - premium received
                    premium = opt.get("total_premium", 0)
                    potential_loss = (strike * 100 * qty) - premium
                    max_loss += potential_loss
                else:
                    # Short call: theoretically unlimited, estimate 3x premium
                    premium = opt.get("total_premium", 0)
                    max_loss += premium * 3
            else:
                # Long options: max loss = premium paid
                max_loss += abs(opt.get("total_premium", 0))

        return round(max_loss, 2)

    def _calculate_risk_score(self, metrics: RiskMetrics) -> float:
        """
        Calculate overall portfolio risk score (0-100)

        Factors:
        - Greeks exposure (30%)
        - Concentration risk (25%)
        - Expiration risk (25%)
        - VaR relative to equity (20%)
        """
        score = 0.0

        # Greeks exposure risk (0-30)
        delta_risk = min(abs(metrics.greeks.net_delta) * 10, 30)
        score += delta_risk * 0.3

        # Concentration risk (0-25)
        concentration_score = metrics.top_3_concentration * 0.25
        score += min(concentration_score, 25)

        # Expiration risk (0-25)
        expiration_score = (
            metrics.expiring_this_week * 5 +
            metrics.expiring_next_week * 2 +
            metrics.assignment_risk_count * 8
        )
        score += min(expiration_score, 25)

        # Gamma risk contribution
        score += metrics.gamma_risk_score * 0.2

        return min(round(score, 1), 100)

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score"""
        if score < 20:
            return RiskLevel.LOW
        elif score < 40:
            return RiskLevel.MODERATE
        elif score < 60:
            return RiskLevel.ELEVATED
        elif score < 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    async def calculate_probability_metrics(
        self,
        positions: Dict[str, Any]
    ) -> ProbabilityMetrics:
        """
        Calculate probability-based metrics

        Args:
            positions: Portfolio positions data

        Returns:
            Probability metrics
        """
        options = positions.get("options", [])
        metrics = ProbabilityMetrics()

        if not options:
            return metrics

        # Portfolio Probability of Profit (weighted average)
        total_value = sum(abs(o.get("current_value", 0)) for o in options)
        if total_value > 0:
            weighted_pop = 0.0
            for opt in options:
                # Estimate PoP from delta
                delta = abs(opt.get("greeks", {}).get("delta", 0)) / 100
                opt_type = opt.get("type")

                if opt_type == "short":
                    # Short options: PoP ≈ 1 - delta
                    pop = 1 - delta
                else:
                    # Long options: PoP ≈ delta
                    pop = delta

                value = abs(opt.get("current_value", 0))
                weighted_pop += pop * value

            metrics.portfolio_pop = round((weighted_pop / total_value) * 100, 1)

        # Expected Value calculation
        for opt in options:
            max_profit = opt.get("total_premium", 0) if opt.get("type") == "short" else float("inf")
            pl = opt.get("pl", 0)
            pop = metrics.portfolio_pop / 100

            # Simplified EV: (PoP * avg_profit) - ((1-PoP) * avg_loss)
            ev = (pop * max(pl, 0)) - ((1 - pop) * abs(min(pl, 0)))
            metrics.expected_value += ev

        metrics.expected_value = round(metrics.expected_value, 2)

        # Theta efficiency (how efficiently capturing time decay)
        total_theta = sum(o.get("greeks", {}).get("theta", 0) * o.get("quantity", 0) for o in options)
        total_premium = sum(o.get("total_premium", 0) for o in options if o.get("type") == "short")
        if total_premium > 0:
            # Theta efficiency = (daily theta / total premium at risk) * DTE
            avg_dte = sum(o.get("dte", 30) for o in options) / len(options)
            metrics.theta_efficiency = round((total_theta / total_premium) * avg_dte * 100, 1)

        return metrics

    async def generate_multi_agent_consensus(
        self,
        position: Dict[str, Any],
        metadata: Optional[Dict] = None
    ) -> AIConsensus:
        """
        Generate multi-agent consensus recommendation

        Simulates multiple AI "agents" analyzing the position:
        - Technical Agent: Price action, trend, momentum
        - Greeks Agent: Options-specific metrics
        - Risk Agent: Risk/reward analysis
        - Sentiment Agent: News/market sentiment
        - Quantitative Agent: Rule-based signals

        Args:
            position: Position data
            metadata: Optional symbol metadata

        Returns:
            Consensus recommendation
        """
        agents_signals: List[Dict] = []

        # Technical Agent
        tech_signal = self._technical_agent_analyze(position, metadata)
        agents_signals.append(tech_signal)

        # Greeks Agent
        greeks_signal = self._greeks_agent_analyze(position)
        agents_signals.append(greeks_signal)

        # Risk Agent
        risk_signal = self._risk_agent_analyze(position)
        agents_signals.append(risk_signal)

        # Quantitative Agent
        quant_signal = self._quant_agent_analyze(position)
        agents_signals.append(quant_signal)

        # Aggregate signals
        return self._aggregate_agent_signals(agents_signals, position)

    def _technical_agent_analyze(self, position: Dict, metadata: Optional[Dict]) -> Dict:
        """Technical analysis agent"""
        signal = {
            "agent": "Technical",
            "action": "hold",
            "confidence": 50,
            "factors": []
        }

        # Use metadata for technical context
        if metadata:
            # Check 52-week range
            high_52w = metadata.get("52w_high", 0)
            low_52w = metadata.get("52w_low", 0)
            strike = position.get("strike", 0)

            if high_52w > 0 and low_52w > 0:
                range_52w = high_52w - low_52w
                if range_52w > 0:
                    # Where is strike relative to 52w range?
                    position_in_range = (strike - low_52w) / range_52w
                    if position_in_range < 0.3:
                        signal["action"] = "hold"
                        signal["confidence"] = 70
                        signal["factors"].append("Strike near 52-week low support")
                    elif position_in_range > 0.8:
                        signal["action"] = "close"
                        signal["confidence"] = 60
                        signal["factors"].append("Strike near 52-week high resistance")

            # Analyst rating
            rating = metadata.get("analyst_rating", "").lower()
            if "buy" in rating:
                signal["confidence"] += 10
                signal["factors"].append("Analyst rating positive")
            elif "sell" in rating:
                signal["confidence"] -= 10
                signal["factors"].append("Analyst rating negative")

        return signal

    def _greeks_agent_analyze(self, position: Dict) -> Dict:
        """Greeks-focused analysis agent"""
        signal = {
            "agent": "Greeks",
            "action": "hold",
            "confidence": 50,
            "factors": []
        }

        greeks = position.get("greeks", {})
        dte = position.get("dte", 999)
        opt_type = position.get("type")
        strategy = position.get("strategy", "")

        delta = abs(greeks.get("delta", 0))
        theta = greeks.get("theta", 0)
        iv = greeks.get("iv", 0)

        # Delta analysis
        if delta > 60:
            signal["factors"].append(f"High delta ({delta:.0f}%) - assignment risk")
            if dte <= 7:
                signal["action"] = "roll"
                signal["confidence"] = 75
        elif delta < 20 and opt_type == "short":
            signal["factors"].append(f"Low delta ({delta:.0f}%) - favorable for short")
            signal["confidence"] += 10

        # Theta analysis for short positions
        if opt_type == "short" and theta > 0:
            if dte <= 21:
                signal["factors"].append(f"Theta acceleration (${theta:.2f}/day)")
                signal["confidence"] += 15

        # IV analysis
        if iv > 80:
            if opt_type == "short":
                signal["factors"].append(f"High IV ({iv:.0f}%) - premium rich")
                signal["confidence"] += 10
            else:
                signal["factors"].append(f"High IV ({iv:.0f}%) - expensive premium")
                signal["confidence"] -= 5
        elif iv < 30:
            if opt_type == "short":
                signal["factors"].append(f"Low IV ({iv:.0f}%) - limited premium")
                signal["confidence"] -= 10

        return signal

    def _risk_agent_analyze(self, position: Dict) -> Dict:
        """Risk management agent"""
        signal = {
            "agent": "Risk",
            "action": "hold",
            "confidence": 50,
            "factors": []
        }

        pl_pct = position.get("pl_pct", 0)
        dte = position.get("dte", 999)
        strategy = position.get("strategy", "")

        # P/L-based signals
        if pl_pct > 50:
            signal["action"] = "close"
            signal["confidence"] = 70
            signal["factors"].append(f"Captured {pl_pct:.0f}% profit - take gains")
        elif pl_pct < -50:
            signal["action"] = "review"
            signal["confidence"] = 65
            signal["factors"].append(f"Down {abs(pl_pct):.0f}% - evaluate exit")

        # DTE-based risk
        if dte <= 3:
            signal["confidence"] += 20
            signal["factors"].append(f"Critical DTE ({dte} days) - high gamma risk")
            if strategy in ["CSP", "CC"] and pl_pct > 30:
                signal["action"] = "close"
        elif dte <= 7:
            signal["factors"].append(f"Expiration week ({dte} DTE) - monitor closely")
            signal["confidence"] += 10

        return signal

    def _quant_agent_analyze(self, position: Dict) -> Dict:
        """Quantitative rules-based agent"""
        signal = {
            "agent": "Quantitative",
            "action": "hold",
            "confidence": 60,  # Rule-based = higher base confidence
            "factors": []
        }

        pl_pct = position.get("pl_pct", 0)
        dte = position.get("dte", 999)
        strategy = position.get("strategy", "")
        delta = abs(position.get("greeks", {}).get("delta", 0))

        # Rule: Close at 50% profit for premium selling
        if strategy in ["CSP", "CC"] and pl_pct >= 50:
            signal["action"] = "close"
            signal["confidence"] = 85
            signal["factors"].append("Rule: Close at 50% profit target")

        # Rule: Roll at 21 DTE if still in position
        if dte <= 21 and pl_pct < 50 and strategy in ["CSP", "CC"]:
            signal["action"] = "roll"
            signal["confidence"] = 70
            signal["factors"].append("Rule: Consider rolling at 21 DTE")

        # Rule: Cut losses at 200% for premium selling
        if strategy in ["CSP", "CC"] and pl_pct <= -100:
            signal["action"] = "close"
            signal["confidence"] = 80
            signal["factors"].append("Rule: Cut loss at 2x premium")

        # Rule: High assignment risk
        if dte <= 7 and delta > 50 and strategy in ["CSP", "CC"]:
            signal["action"] = "roll"
            signal["confidence"] = 75
            signal["factors"].append("Rule: Roll to avoid assignment")

        return signal

    def _aggregate_agent_signals(
        self,
        signals: List[Dict],
        position: Dict
    ) -> AIConsensus:
        """Aggregate signals from multiple agents into consensus"""

        # Count actions
        action_counts: Dict[str, int] = {}
        action_confidences: Dict[str, List[float]] = {}
        all_factors: List[str] = []
        dissenting: List[str] = []

        for sig in signals:
            action = sig["action"]
            conf = sig["confidence"]
            factors = sig.get("factors", [])

            action_counts[action] = action_counts.get(action, 0) + 1
            if action not in action_confidences:
                action_confidences[action] = []
            action_confidences[action].append(conf)
            all_factors.extend(factors)

        # Determine majority action
        majority_action = max(action_counts.keys(), key=lambda k: action_counts[k])
        majority_count = action_counts[majority_action]

        # Calculate consensus confidence
        if majority_action in action_confidences:
            avg_confidence = sum(action_confidences[majority_action]) / len(action_confidences[majority_action])
            # Boost confidence if unanimous
            if majority_count == len(signals):
                avg_confidence = min(avg_confidence * 1.2, 95)
            # Reduce if split
            elif majority_count < len(signals) / 2:
                avg_confidence *= 0.8
        else:
            avg_confidence = 50

        # Identify dissenting opinions
        for sig in signals:
            if sig["action"] != majority_action:
                dissenting.append(f"{sig['agent']}: {sig['action']} ({sig['confidence']}%)")

        # Determine urgency
        dte = position.get("dte", 999)
        pl_pct = position.get("pl_pct", 0)

        if dte <= 3 or abs(pl_pct) > 75:
            urgency = "high"
        elif dte <= 7 or abs(pl_pct) > 50:
            urgency = "medium"
        else:
            urgency = "low"

        # Build rationale
        rationale = f"{majority_count}/{len(signals)} agents recommend '{majority_action}'. "
        if all_factors:
            rationale += "Key factors: " + "; ".join(list(set(all_factors))[:3])

        return AIConsensus(
            action=majority_action,
            confidence=round(avg_confidence, 1),
            urgency=urgency,
            agents_agree=majority_count,
            agents_total=len(signals),
            dissenting_opinions=dissenting,
            key_factors=list(set(all_factors))[:5],
            rationale=rationale
        )

    async def generate_position_alerts(
        self,
        positions: Dict[str, Any]
    ) -> List[PositionAlert]:
        """
        Generate proactive position alerts

        Args:
            positions: Portfolio positions data

        Returns:
            List of position alerts
        """
        alerts: List[PositionAlert] = []
        options = positions.get("options", [])

        for opt in options:
            symbol = opt.get("symbol", "")
            dte = opt.get("dte", 999)
            pl_pct = opt.get("pl_pct", 0)
            delta = abs(opt.get("greeks", {}).get("delta", 0))
            strategy = opt.get("strategy", "")

            # Expiration alerts
            if dte <= 1:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type="expiring",
                    severity="critical",
                    message=f"{symbol} {strategy} expires TODAY",
                    action_required="Close or let expire"
                ))
            elif dte <= 3:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type="expiring",
                    severity="warning",
                    message=f"{symbol} {strategy} expires in {dte} days",
                    action_required="Plan exit or roll"
                ))
            elif dte <= 7:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type="expiring",
                    severity="info",
                    message=f"{symbol} {strategy} expiration week ({dte} DTE)",
                    action_required="Monitor gamma risk"
                ))

            # Assignment risk alerts
            if dte <= 7 and delta > 50 and opt.get("type") == "short":
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type="assignment_risk",
                    severity="warning",
                    message=f"{symbol} has {delta:.0f}% delta with {dte} DTE",
                    action_required="Roll out or prepare for assignment"
                ))

            # Profit taking alerts
            if pl_pct >= 50 and strategy in ["CSP", "CC"]:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type="take_profit",
                    severity="info",
                    message=f"{symbol} at {pl_pct:.0f}% profit",
                    action_required="Consider closing for profit"
                ))

            # Stop loss alerts
            if pl_pct <= -100 and strategy in ["CSP", "CC"]:
                alerts.append(PositionAlert(
                    symbol=symbol,
                    alert_type="stop_loss",
                    severity="critical",
                    message=f"{symbol} down {abs(pl_pct):.0f}% - 2x premium loss",
                    action_required="Evaluate closing to limit loss"
                ))

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 3))

        return alerts

    async def stream_portfolio_insights(
        self,
        positions: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream portfolio insights for real-time UI updates

        Yields:
            Progressive insight updates
        """
        yield {
            "type": "status",
            "message": "Starting portfolio analysis...",
            "progress": 0
        }

        # Risk metrics
        yield {
            "type": "status",
            "message": "Calculating risk metrics...",
            "progress": 20
        }
        risk_metrics = await self.analyze_portfolio_risk(positions)
        yield {
            "type": "risk_metrics",
            "data": {
                "net_delta": risk_metrics.greeks.net_delta,
                "net_theta": risk_metrics.greeks.net_theta,
                "var_95": risk_metrics.var_1d_95,
                "risk_score": risk_metrics.portfolio_risk_score,
                "risk_level": risk_metrics.risk_level.value
            },
            "progress": 40
        }

        # Probability metrics
        yield {
            "type": "status",
            "message": "Calculating probability metrics...",
            "progress": 50
        }
        prob_metrics = await self.calculate_probability_metrics(positions)
        yield {
            "type": "probability_metrics",
            "data": {
                "portfolio_pop": prob_metrics.portfolio_pop,
                "expected_value": prob_metrics.expected_value,
                "theta_efficiency": prob_metrics.theta_efficiency
            },
            "progress": 60
        }

        # Position alerts
        yield {
            "type": "status",
            "message": "Generating alerts...",
            "progress": 70
        }
        alerts = await self.generate_position_alerts(positions)
        yield {
            "type": "alerts",
            "data": [
                {
                    "symbol": a.symbol,
                    "type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "action": a.action_required
                }
                for a in alerts
            ],
            "progress": 90
        }

        # Complete
        yield {
            "type": "complete",
            "message": "Analysis complete",
            "progress": 100
        }


# Singleton instance
_analytics_service: Optional[AdvancedPortfolioAnalytics] = None


def get_analytics_service() -> AdvancedPortfolioAnalytics:
    """Get singleton analytics service"""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = AdvancedPortfolioAnalytics()
    return _analytics_service
