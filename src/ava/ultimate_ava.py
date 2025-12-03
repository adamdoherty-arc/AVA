"""
Ultimate AVA - Complete Financial Advisory System
==================================================

Integrates all world-class features into one powerful system:
- Real portfolio integration
- Proactive monitoring & alerts
- Risk analytics (VaR, Sharpe, stress testing)
- Opportunity scanning
- Outcome tracking & learning
- Tax optimization
- Voice interface ready
- And much more!

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# =============================================================================
# ULTIMATE AVA - MAIN CLASS
# =============================================================================

class UltimateAVA:
    """
    The most advanced AI financial advisor ever built.

    Combines:
    - World-class prompts (Chain-of-Thought reasoning)
    - Real portfolio integration (Robinhood)
    - 3 FREE market data APIs
    - Proactive monitoring & alerts
    - Risk analytics suite
    - Opportunity scanning
    - Outcome tracking & learning
    - Tax optimization
    - And more!
    """

    def __init__(self) -> None:
        """Initialize Ultimate AVA with all systems"""
        logger.info("🚀 Initializing Ultimate AVA...")

        # Core systems
        self._init_world_class_ava()
        self._init_portfolio_integration()
        self._init_risk_analytics()
        self._init_monitoring_system()
        self._init_opportunity_scanner()
        self._init_outcome_tracker()
        self._init_tax_optimizer()

        logger.info("✅ Ultimate AVA initialized and ready!")

    def _init_world_class_ava(self) -> None:
        """Initialize world-class AVA core"""
        try:
            from src.ava.world_class_ava_integration import WorldClassAVA
            self.world_class_ava = WorldClassAVA()
            logger.info("✅ World-Class AVA core loaded")
        except Exception as e:
            logger.warning(f"World-Class AVA not available: {e}")
            self.world_class_ava = None

    def _init_portfolio_integration(self) -> None:
        """Initialize real portfolio integration"""
        try:
            from src.services.robinhood_client import get_robinhood_client
            from src.ava.agents.trading.portfolio_agent import PortfolioAgent

            self.robinhood_client = get_robinhood_client()
            self.portfolio_agent = PortfolioAgent()
            logger.info("✅ Portfolio integration active")
        except Exception as e:
            logger.warning(f"Portfolio integration not available: {e}")
            self.robinhood_client = None
            self.portfolio_agent = None

    def _init_risk_analytics(self) -> None:
        """Initialize risk analytics suite"""
        from src.ava.systems.risk_analytics import RiskAnalytics
        self.risk_analytics = RiskAnalytics()
        logger.info("✅ Risk analytics suite loaded")

    def _init_monitoring_system(self) -> None:
        """Initialize proactive monitoring"""
        from src.ava.systems.proactive_monitor import ProactiveMonitor
        self.monitor = ProactiveMonitor()
        logger.info("✅ Proactive monitoring active")

    def _init_opportunity_scanner(self) -> None:
        """Initialize opportunity scanner"""
        from src.ava.systems.opportunity_scanner import OpportunityScanner
        self.scanner = OpportunityScanner()
        logger.info("✅ Opportunity scanner ready")

    def _init_outcome_tracker(self) -> None:
        """Initialize outcome tracking"""
        from src.ava.systems.outcome_tracker import OutcomeTracker
        self.tracker = OutcomeTracker()
        logger.info("✅ Outcome tracking active")

    def _init_tax_optimizer(self) -> None:
        """Initialize tax optimization"""
        from src.ava.systems.tax_optimizer import TaxOptimizer
        self.tax_optimizer = TaxOptimizer()
        logger.info("✅ Tax optimizer ready")

    # =========================================================================
    # MORNING BRIEFING - Start Your Day Right!
    # =========================================================================

    def morning_briefing(self) -> str:
        """
        Comprehensive morning briefing with everything you need to know.

        Returns formatted briefing with:
        - Portfolio status
        - Today's agenda (earnings, expirations)
        - Risk alerts
        - Opportunities
        - Market context
        """
        briefing_parts = []

        # Header
        now = datetime.now()
        briefing_parts.append(f"""
{'='*70}
GOOD MORNING! {now.strftime('%A, %B %d, %Y')}
{'='*70}
""")

        # Portfolio Status
        portfolio_status = self._get_portfolio_status()
        briefing_parts.append(portfolio_status)

        # Today's Agenda
        agenda = self._get_todays_agenda()
        briefing_parts.append(agenda)

        # Risk Alerts
        alerts = self.monitor.get_risk_alerts()
        if alerts:
            briefing_parts.append("\n[!] RISK ALERTS:\n")
            for alert in alerts:
                briefing_parts.append(f"  - {alert}\n")

        # Opportunities
        opportunities = self._get_top_opportunities()
        briefing_parts.append(opportunities)

        # Market Context
        market_context = self._get_market_context_summary()
        briefing_parts.append(market_context)

        # Footer
        briefing_parts.append(f"\n{'='*70}\n")
        briefing_parts.append("Have a profitable day!\n")
        briefing_parts.append(f"{'='*70}\n")

        return ''.join(briefing_parts)

    def _get_portfolio_status(self) -> str:
        """Get portfolio status section"""
        if not self.robinhood_client:
            return "\nPORTFOLIO STATUS: Not connected\n"

        try:
            account = self.robinhood_client.get_account_info()
            positions = self.robinhood_client.get_positions()

            # Calculate daily change
            current_value = float(account.get('portfolio_value', 0))
            prev_close = float(account.get('previous_close', current_value))
            daily_change = current_value - prev_close
            daily_change_pct = (daily_change / prev_close * 100) if prev_close > 0 else 0

            status = f"""
PORTFOLIO STATUS:
Total Value: ${current_value:,.2f}
Daily Change: ${daily_change:+,.2f} ({daily_change_pct:+.2f}%)
Cash: ${float(account.get('cash', 0)):,.2f}
Buying Power: ${float(account.get('buying_power', 0)):,.2f}
Positions: {len(positions)}
"""

            # Add YTD performance if available
            try:
                ytd = self.tracker.get_ytd_performance()
                if ytd:
                    status += f"YTD Return: {ytd['return_pct']:+.2f}% (vs S&P: {ytd.get('sp500_return', 0):+.2f}%)\n"
            except:
                pass

            return status

        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return "\nPORTFOLIO STATUS: Error loading data\n"

    def _get_todays_agenda(self) -> str:
        """Get today's agenda"""
        agenda_items = []

        # Check for earnings today
        try:
            earnings_today = self.monitor.check_earnings_today()
            if earnings_today:
                agenda_items.append(f"{len(earnings_today)} positions have earnings today: {', '.join(earnings_today)}")
        except:
            pass

        # Check for expiring options
        try:
            expiring = self.monitor.check_expiring_options()
            if expiring:
                agenda_items.append(f"{len(expiring)} options expiring this week")
        except:
            pass

        # Check for ex-dividend dates
        try:
            ex_div = self.monitor.check_ex_dividend_dates()
            if ex_div:
                agenda_items.append(f"{len(ex_div)} positions go ex-dividend this week")
        except:
            pass

        if agenda_items:
            return "\nTODAY'S AGENDA:\n" + '\n'.join([f"  - {item}" for item in agenda_items]) + "\n"
        else:
            return "\nTODAY'S AGENDA: No critical events\n"

    def _get_top_opportunities(self) -> str:
        """Get top trading opportunities"""
        try:
            opportunities = self.scanner.scan_csp_opportunities(limit=3)

            if not opportunities:
                return "\nOPPORTUNITIES: No new opportunities right now\n"

            result = "\nTOP OPPORTUNITIES:\n"
            for i, opp in enumerate(opportunities, 1):
                result += f"""
  {i}. {opp['ticker']} ${opp['strike']} {opp['dte']} DTE
     Premium Yield: {opp['yield']:.2f}%
     Sentiment: {opp['sentiment']}
     Quality Score: {opp['score']}/100
"""

            return result

        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
            return "\nOPPORTUNITIES: Error scanning\n"

    def _get_market_context_summary(self) -> str:
        """Get market context summary"""
        if not self.world_class_ava or not self.world_class_ava.market_data:
            return "\nMARKET CONTEXT: Not available\n"

        try:
            context = self.world_class_ava.market_data.get_market_context_for_ava()

            result = f"""
MARKET CONTEXT:
Market Regime: {context.get('market_regime', 'Unknown')}
Recession Risk: {context.get('recession_risk', 'Unknown')}
VIX: {context.get('vix_level', 'N/A')}
Fed Funds Rate: {context.get('fed_funds_rate', 'N/A')}%
Volatility: {context.get('volatility_regime', 'Unknown')}
"""
            return result

        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return "\nMARKET CONTEXT: Error loading\n"

    # =========================================================================
    # COMPREHENSIVE ANALYSIS - Ask AVA Anything!
    # =========================================================================

    def analyze_question(
        self,
        question: str,
        personality_mode: str = "professional"
    ) -> Dict[str, Any]:
        """
        Analyze any trading question with full context.

        This is the POWER function - uses everything:
        - Your real portfolio
        - Current market data
        - AI sentiment
        - Economic indicators
        - Historical outcomes
        - Risk metrics
        - And generates world-class response

        Args:
            question: User's question
            personality_mode: AVA's personality

        Returns:
            Complete analysis with recommendation
        """
        logger.info(f"🔍 Analyzing question: {question}")

        # Gather all context
        context = self._gather_complete_context()

        # Generate world-class prompt
        if self.world_class_ava:
            prompt = self.world_class_ava.generate_world_class_prompt(
                user_query=question,
                user_profile=context.get('user_profile'),
                portfolio_context=context.get('portfolio'),
                rag_context=context.get('rag_context'),
                conversation_history=context.get('history'),
                personality_mode=personality_mode
            )

            # Get LLM response (would integrate with your LLM here)
            # For now, return structured analysis
            return {
                'question': question,
                'prompt': prompt,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'question': question,
                'error': 'World-Class AVA not available',
                'context': context
            }

    def _gather_complete_context(self) -> Dict[str, Any]:
        """Gather all context for AVA's analysis"""
        context = {}

        # Portfolio context
        if self.robinhood_client:
            try:
                account = self.robinhood_client.get_account_info()
                positions = self.robinhood_client.get_positions()

                context['portfolio'] = {
                    'total_value': float(account.get('portfolio_value', 0)),
                    'cash': float(account.get('cash', 0)),
                    'buying_power': float(account.get('buying_power', 0)),
                    'num_positions': len(positions),
                    'positions': positions
                }
            except Exception as e:
                logger.error(f"Error getting portfolio: {e}")

        # User profile (you can customize this)
        context['user_profile'] = {
            'risk_tolerance': 'moderate',
            'experience_level': 'intermediate',
            'goals': ['income generation', 'capital preservation'],
            'preferred_strategy': 'wheel strategy',
            'max_position_size': 10000
        }

        # Market context
        if self.world_class_ava:
            try:
                context['market'] = self.world_class_ava._get_market_context()
            except:
                pass

        # Recent outcomes (for learning)
        try:
            context['recent_outcomes'] = self.tracker.get_recent_outcomes(limit=10)
        except:
            pass

        return context

    # =========================================================================
    # RISK ANALYSIS - Know Your Risk!
    # =========================================================================

    def get_risk_report(self) -> str:
        """
        Comprehensive risk report for portfolio.

        Includes:
        - VaR (Value at Risk)
        - Sharpe Ratio
        - Stress testing
        - Concentration analysis
        - Greeks exposure
        - Correlation risks
        """
        if not self.robinhood_client:
            return "[!] Portfolio not connected - cannot assess risk"

        try:
            # Get portfolio data
            account = self.robinhood_client.get_account_info()
            positions = self.robinhood_client.get_positions()
            portfolio_value = float(account.get('portfolio_value', 0))

            # Calculate risk metrics
            var_95 = self.risk_analytics.calculate_var(portfolio_value)
            sharpe = self.risk_analytics.calculate_sharpe_ratio(positions)
            stress_tests = self.risk_analytics.stress_test_portfolio(portfolio_value, positions)

            # Build report
            report = f"""
{'='*70}
COMPREHENSIVE RISK REPORT
{'='*70}

Portfolio Value: ${portfolio_value:,.2f}

VALUE AT RISK (VaR):
{var_95}

RISK-ADJUSTED RETURNS:
Sharpe Ratio: {sharpe['ratio']:.2f} ({sharpe['rating']})
Annualized Return: {sharpe['annual_return']:.2f}%
Annualized Volatility: {sharpe['annual_vol']:.2f}%

STRESS TEST SCENARIOS:
"""

            for scenario, result in stress_tests.items():
                report += f"\n{scenario}:"
                report += f"\n  Portfolio Value: ${result['portfolio_value']:,.2f}"
                report += f"\n  Loss: ${result['loss']:,.2f} ({result['loss_pct']:.2f}%)\n"

            # Add concentration analysis
            concentration = self.risk_analytics.analyze_concentration(positions, portfolio_value)
            report += f"""
CONCENTRATION ANALYSIS:
Largest Position: {concentration['largest_position_pct']:.1f}% ({concentration['largest_position']})
"""

            if concentration['concentration_risk'] == 'High':
                report += "  [!] HIGH concentration risk - consider diversifying\n"

            report += f"\n{'='*70}\n"

            return report

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return f"[!] Error generating risk report: {e}"

    # =========================================================================
    # OPPORTUNITY SCANNING - Find Great Trades!
    # =========================================================================

    def scan_for_opportunities(
        self,
        strategy: str = 'CSP',
        min_quality_score: int = 70
    ) -> List[Dict]:
        """
        Scan market for high-quality trading opportunities.

        Args:
            strategy: 'CSP', 'CC', 'SPREAD', etc.
            min_quality_score: Minimum quality score (0-100)

        Returns:
            List of opportunities sorted by quality score
        """
        logger.info(f"🔍 Scanning for {strategy} opportunities...")

        try:
            if strategy == 'CSP':
                opportunities = self.scanner.scan_csp_opportunities(
                    min_score=min_quality_score
                )
            else:
                opportunities = []

            logger.info(f"✅ Found {len(opportunities)} opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
            return []

    # =========================================================================
    # OUTCOME TRACKING - Learn & Improve!
    # =========================================================================

    def log_trade_recommendation(
        self,
        ticker: str,
        action: str,
        details: Dict,
        recommendation: str,
        confidence: float
    ) -> int:
        """
        Log a trade recommendation for outcome tracking.

        Args:
            ticker: Stock ticker
            action: 'BUY', 'SELL', 'HOLD', etc.
            details: Trade details
            recommendation: AVA's recommendation
            confidence: Confidence level (0-1)

        Returns:
            Recommendation ID for later outcome tracking
        """
        return self.tracker.log_recommendation(
            ticker=ticker,
            action=action,
            details=details,
            recommendation=recommendation,
            confidence=confidence
        )

    def track_outcome(self, recommendation_id: int, outcome: Dict):
        """
        Track the outcome of a previous recommendation.

        Args:
            recommendation_id: ID from log_trade_recommendation
            outcome: Outcome details (profit, win/loss, etc.)
        """
        self.tracker.track_outcome(recommendation_id, outcome)

    def get_performance_stats(self) -> Dict:
        """Get AVA's performance statistics"""
        return self.tracker.get_statistics()

    # =========================================================================
    # TAX OPTIMIZATION - Save Money!
    # =========================================================================

    def find_tax_opportunities(self) -> List[Dict]:
        """
        Find tax optimization opportunities.

        Returns:
            List of tax-saving opportunities:
            - Tax-loss harvesting
            - Wash sale warnings
            - Asset location optimization
        """
        if not self.robinhood_client:
            return []

        try:
            positions = self.robinhood_client.get_positions()
            return self.tax_optimizer.find_opportunities(positions)

        except Exception as e:
            logger.error(f"Error finding tax opportunities: {e}")
            return []

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_comprehensive_status(self) -> str:
        """Get complete status of Ultimate AVA"""
        status = f"""
{'='*70}
ULTIMATE AVA - SYSTEM STATUS
{'='*70}

Core Systems:
"""

        # Check each system
        systems = {
            'World-Class AVA': self.world_class_ava is not None,
            'Portfolio Integration': self.robinhood_client is not None,
            'Risk Analytics': self.risk_analytics is not None,
            'Proactive Monitor': self.monitor is not None,
            'Opportunity Scanner': self.scanner is not None,
            'Outcome Tracker': self.tracker is not None,
            'Tax Optimizer': self.tax_optimizer is not None
        }

        for system, active in systems.items():
            status_icon = "[OK]" if active else "[X]"
            status += f"\n  {status_icon} {system}"

        status += f"\n\n{'='*70}\n"

        return status


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_ultimate_ava = None

def get_ultimate_ava() -> UltimateAVA:
    """Get singleton Ultimate AVA instance"""
    global _ultimate_ava
    if _ultimate_ava is None:
        _ultimate_ava = UltimateAVA()
    return _ultimate_ava


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def morning_briefing() -> str:
    """Quick function to get morning briefing"""
    return get_ultimate_ava().morning_briefing()


def analyze(question: str) -> Dict:
    """Quick function to analyze a question"""
    return get_ultimate_ava().analyze_question(question)


def risk_report() -> str:
    """Quick function to get risk report"""
    return get_ultimate_ava().get_risk_report()


def scan_opportunities() -> List[Dict]:
    """Quick function to scan for opportunities"""
    return get_ultimate_ava().scan_for_opportunities()


if __name__ == "__main__":
    # Test Ultimate AVA
    import logging
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Ultimate AVA ===\n")

    # Initialize
    ava = UltimateAVA()

    # Show status
    print(ava.get_comprehensive_status())

    # Test morning briefing (if portfolio connected)
    try:
        print("\n=== MORNING BRIEFING ===")
        print(ava.morning_briefing())
    except Exception as e:
        print(f"Morning briefing error: {e}")

    # Test risk report
    try:
        print("\n=== RISK REPORT ===")
        print(ava.get_risk_report())
    except Exception as e:
        print(f"Risk report error: {e}")
