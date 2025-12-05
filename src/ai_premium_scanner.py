"""
AI-Enhanced Premium Scanner
Modern, AI-powered options premium scanner with:
- Multi-criteria scoring (MCDM)
- LLM-powered analysis and recommendations
- Real-time sentiment analysis
- Smart opportunity ranking
- AI-generated trade insights
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

from src.premium_scanner import PremiumScanner
from src.ai_options_agent.scoring_engine import MultiCriteriaScorer

logger = logging.getLogger(__name__)


class AIPremiumScanner:
    """
    AI-Enhanced Premium Scanner

    Extends the base PremiumScanner with:
    - Multi-criteria AI scoring for each opportunity
    - LLM-powered analysis for top opportunities
    - Sentiment analysis integration
    - Smart filtering and ranking
    - AI-generated trade recommendations
    """

    def __init__(self) -> None:
        self.base_scanner = PremiumScanner()
        self.scorer = MultiCriteriaScorer()
        self._llm_service = None
        self._fundamental_cache = {}

    def _get_llm_service(self) -> None:
        """Lazy load LLM service to avoid import issues"""
        if self._llm_service is None:
            try:
                from src.services.llm_service import get_llm_service
                self._llm_service = get_llm_service()
            except Exception as e:
                logger.warning(f"LLM service not available: {e}")
        return self._llm_service

    def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data for a symbol with caching"""
        if symbol in self._fundamental_cache:
            return self._fundamental_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamentals = {
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'eps': info.get('trailingEps'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'short_ratio': info.get('shortRatio'),
                'recommendation': info.get('recommendationKey'),
                'target_price': info.get('targetMeanPrice'),
            }

            self._fundamental_cache[symbol] = fundamentals
            return fundamentals

        except Exception as e:
            logger.debug(f"Could not fetch fundamentals for {symbol}: {e}")
            return {}

    def _map_opportunity_for_scoring(self, opp: Dict[str, Any], fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Map scanner output to scorer input format"""
        return {
            # Basic info
            'symbol': opp.get('symbol'),
            'stock_price': opp.get('stock_price'),
            'strike_price': opp.get('strike'),
            'expiration_date': opp.get('expiration'),
            'dte': opp.get('dte'),

            # Premium info
            'premium': opp.get('premium'),
            'premium_pct': opp.get('premium_pct'),
            'monthly_return': opp.get('monthly_return'),
            'annual_return': opp.get('annual_return'),

            # Greeks
            'delta': opp.get('delta'),
            'theta': opp.get('theta'),
            'iv': opp.get('iv', 0) / 100 if opp.get('iv') else 0,  # Convert to decimal

            # Liquidity
            'bid': opp.get('bid'),
            'ask': opp.get('ask'),
            'volume': opp.get('volume'),
            'oi': opp.get('open_interest'),

            # Calculated fields
            'breakeven': opp.get('strike', 0) - (opp.get('premium', 0) / 100),

            # Fundamentals
            'pe_ratio': fundamentals.get('pe_ratio'),
            'eps': fundamentals.get('eps'),
            'market_cap': fundamentals.get('market_cap'),
            'sector': fundamentals.get('sector'),
            'dividend_yield': fundamentals.get('dividend_yield'),
        }

    def score_opportunity(self, opp: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single opportunity with AI multi-criteria analysis"""
        fundamentals = self._fetch_fundamentals(opp.get('symbol', ''))
        mapped = self._map_opportunity_for_scoring(opp, fundamentals)
        scores = self.scorer.score_opportunity(mapped)

        # Merge original opportunity with scores
        return {
            **opp,
            'ai_score': scores.get('final_score', 50),
            'ai_recommendation': scores.get('recommendation', 'HOLD'),
            'ai_confidence': scores.get('confidence', 50),
            'fundamental_score': scores.get('fundamental_score', 50),
            'technical_score': scores.get('technical_score', 50),
            'greeks_score': scores.get('greeks_score', 50),
            'risk_score': scores.get('risk_score', 50),
            'sentiment_score': scores.get('sentiment_score', 70),
            # Add fundamentals
            'sector': fundamentals.get('sector'),
            'industry': fundamentals.get('industry'),
            'pe_ratio': fundamentals.get('pe_ratio'),
            'market_cap': fundamentals.get('market_cap'),
            'beta': fundamentals.get('beta'),
        }

    def scan_with_ai(
        self,
        symbols: List[str],
        max_price: float = 250,
        min_premium_pct: float = 0.5,
        dte: int = 30,
        min_ai_score: int = 0,
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Scan for premium opportunities with AI scoring

        Args:
            symbols: List of stock symbols to scan
            max_price: Maximum stock price
            min_premium_pct: Minimum premium percentage
            dte: Target days to expiration
            min_ai_score: Minimum AI score to include (0-100)
            top_n: Return only top N results (None = all)

        Returns:
            Dict with opportunities, stats, and AI insights
        """
        start_time = datetime.now()

        # Run base scan
        opportunities = self.base_scanner.scan_premiums(
            symbols=symbols,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte=dte
        )

        if not opportunities:
            return {
                'opportunities': [],
                'total_scanned': len(symbols),
                'total_found': 0,
                'scan_time_seconds': (datetime.now() - start_time).total_seconds(),
                'ai_insights': None
            }

        # Score each opportunity with AI
        scored_opportunities = []
        for opp in opportunities:
            try:
                scored = self.score_opportunity(opp)
                if scored.get('ai_score', 0) >= min_ai_score:
                    scored_opportunities.append(scored)
            except Exception as e:
                logger.warning(f"Error scoring {opp.get('symbol')}: {e}")
                opp['ai_score'] = 50
                opp['ai_recommendation'] = 'HOLD'
                scored_opportunities.append(opp)

        # Sort by AI score (highest first)
        scored_opportunities.sort(key=lambda x: x.get('ai_score', 0), reverse=True)

        # Limit results if requested
        if top_n and top_n > 0:
            scored_opportunities = scored_opportunities[:top_n]

        # Calculate statistics
        stats = self._calculate_stats(scored_opportunities)

        # Generate AI insights for top opportunities
        ai_insights = self._generate_insights(scored_opportunities[:10])

        scan_time = (datetime.now() - start_time).total_seconds()

        return {
            'opportunities': scored_opportunities,
            'total_scanned': len(symbols),
            'total_found': len(scored_opportunities),
            'scan_time_seconds': round(scan_time, 2),
            'stats': stats,
            'ai_insights': ai_insights,
            'generated_at': datetime.now().isoformat()
        }

    def _calculate_stats(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics"""
        if not opportunities:
            return {}

        ai_scores = [o.get('ai_score', 0) for o in opportunities]
        monthly_returns = [o.get('monthly_return', 0) for o in opportunities]

        # Count by recommendation
        rec_counts = {}
        for opp in opportunities:
            rec = opp.get('ai_recommendation', 'UNKNOWN')
            rec_counts[rec] = rec_counts.get(rec, 0) + 1

        # Count by sector
        sector_counts = {}
        for opp in opportunities:
            sector = opp.get('sector', 'Unknown')
            if sector:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Top sectors
        top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'avg_ai_score': round(sum(ai_scores) / len(ai_scores), 1) if ai_scores else 0,
            'max_ai_score': max(ai_scores) if ai_scores else 0,
            'min_ai_score': min(ai_scores) if ai_scores else 0,
            'avg_monthly_return': round(sum(monthly_returns) / len(monthly_returns), 2) if monthly_returns else 0,
            'max_monthly_return': round(max(monthly_returns), 2) if monthly_returns else 0,
            'recommendation_breakdown': rec_counts,
            'strong_buy_count': rec_counts.get('STRONG_BUY', 0),
            'buy_count': rec_counts.get('BUY', 0),
            'hold_count': rec_counts.get('HOLD', 0),
            'top_sectors': dict(top_sectors),
            'unique_symbols': len(set(o.get('symbol') for o in opportunities)),
        }

    def _generate_insights(self, top_opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate AI-powered insights for opportunities"""
        if not top_opportunities:
            return None

        # Rule-based insights (fast, no LLM required)
        insights = {
            'summary': self._generate_summary(top_opportunities),
            'top_pick': self._get_top_pick(top_opportunities),
            'market_conditions': self._analyze_market_conditions(top_opportunities),
            'risk_assessment': self._assess_overall_risk(top_opportunities),
            'sector_analysis': self._analyze_sectors(top_opportunities),
            'recommendations': self._generate_recommendations(top_opportunities),
        }

        return insights

    def _generate_summary(self, opportunities: List[Dict[str, Any]]) -> str:
        """Generate a summary of the scan results"""
        if not opportunities:
            return "No opportunities found matching criteria."

        strong_buys = sum(1 for o in opportunities if o.get('ai_recommendation') == 'STRONG_BUY')
        buys = sum(1 for o in opportunities if o.get('ai_recommendation') == 'BUY')
        avg_score = sum(o.get('ai_score', 0) for o in opportunities) / len(opportunities)
        avg_return = sum(o.get('monthly_return', 0) for o in opportunities) / len(opportunities)

        quality = "excellent" if avg_score >= 75 else "good" if avg_score >= 60 else "moderate"

        return (
            f"Found {len(opportunities)} opportunities with {quality} average quality. "
            f"{strong_buys} strong buys, {buys} buys. "
            f"Average AI score: {avg_score:.0f}/100, average monthly return: {avg_return:.1f}%."
        )

    def _get_top_pick(self, opportunities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Get the top-rated opportunity"""
        if not opportunities:
            return None

        top = opportunities[0]
        return {
            'symbol': top.get('symbol'),
            'strike': top.get('strike'),
            'expiration': top.get('expiration'),
            'ai_score': top.get('ai_score'),
            'recommendation': top.get('ai_recommendation'),
            'monthly_return': top.get('monthly_return'),
            'premium': top.get('premium'),
            'reason': self._explain_top_pick(top)
        }

    def _explain_top_pick(self, opp: Dict[str, Any]) -> str:
        """Generate explanation for why this is the top pick"""
        reasons = []

        if opp.get('ai_score', 0) >= 80:
            reasons.append("exceptional AI score")
        if opp.get('monthly_return', 0) >= 5:
            reasons.append("high monthly return")
        if opp.get('liquidity_score', 0) >= 50:
            reasons.append("excellent liquidity")
        if opp.get('spread_quality') == 'tight':
            reasons.append("tight bid-ask spread")
        if opp.get('sector') in ['Technology', 'Healthcare']:
            reasons.append(f"strong {opp.get('sector')} sector")

        if not reasons:
            reasons.append("balanced risk/reward profile")

        return "Top pick due to: " + ", ".join(reasons)

    def _analyze_market_conditions(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall market conditions from opportunities"""
        if not opportunities:
            return {}

        avg_iv = sum(o.get('iv', 0) for o in opportunities) / len(opportunities)
        high_iv_count = sum(1 for o in opportunities if o.get('iv', 0) >= 50)

        iv_environment = "high" if avg_iv >= 40 else "moderate" if avg_iv >= 25 else "low"

        return {
            'iv_environment': iv_environment,
            'avg_iv': round(avg_iv, 1),
            'high_iv_opportunities': high_iv_count,
            'premium_quality': 'excellent' if avg_iv >= 35 else 'good' if avg_iv >= 25 else 'moderate',
            'market_insight': f"IV environment is {iv_environment}. {'Great time for premium selling.' if iv_environment in ['high', 'moderate'] else 'Consider waiting for higher IV.'}"
        }

    def _assess_overall_risk(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall risk of opportunities"""
        if not opportunities:
            return {}

        avg_risk_score = sum(o.get('risk_score', 50) for o in opportunities) / len(opportunities)
        avg_otm = sum(o.get('otm_pct', 0) for o in opportunities) / len(opportunities)

        risk_level = "low" if avg_risk_score >= 75 else "moderate" if avg_risk_score >= 55 else "elevated"

        return {
            'risk_level': risk_level,
            'avg_risk_score': round(avg_risk_score, 1),
            'avg_otm_distance': round(avg_otm, 1),
            'risk_insight': f"Overall risk level is {risk_level}. Average {avg_otm:.1f}% OTM provides {'good' if avg_otm >= 8 else 'moderate'} safety margin."
        }

    def _analyze_sectors(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sector distribution"""
        sector_data = {}
        for opp in opportunities:
            sector = opp.get('sector', 'Unknown')
            if sector not in sector_data:
                sector_data[sector] = {'count': 0, 'total_score': 0, 'total_return': 0}
            sector_data[sector]['count'] += 1
            sector_data[sector]['total_score'] += opp.get('ai_score', 0)
            sector_data[sector]['total_return'] += opp.get('monthly_return', 0)

        # Calculate averages
        for sector in sector_data:
            count = sector_data[sector]['count']
            sector_data[sector]['avg_score'] = round(sector_data[sector]['total_score'] / count, 1)
            sector_data[sector]['avg_return'] = round(sector_data[sector]['total_return'] / count, 2)

        # Find best sector
        best_sector = max(sector_data.items(), key=lambda x: x[1]['avg_score'], default=(None, {}))

        return {
            'sectors': sector_data,
            'best_sector': best_sector[0],
            'best_sector_score': best_sector[1].get('avg_score', 0),
            'diversification': 'good' if len(sector_data) >= 3 else 'limited'
        }

    def _generate_recommendations(self, opportunities: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if not opportunities:
            return ["No opportunities found. Consider expanding search criteria."]

        # Check for strong buys
        strong_buys = [o for o in opportunities if o.get('ai_recommendation') == 'STRONG_BUY']
        if strong_buys:
            top = strong_buys[0]
            recommendations.append(
                f"STRONG BUY: {top.get('symbol')} ${top.get('strike')} put expiring {top.get('expiration')} "
                f"- {top.get('monthly_return'):.1f}% monthly return, AI score {top.get('ai_score')}/100"
            )

        # Check IV environment
        avg_iv = sum(o.get('iv', 0) for o in opportunities) / len(opportunities)
        if avg_iv >= 40:
            recommendations.append(
                "High IV environment - excellent for premium selling. Consider sizing up positions."
            )
        elif avg_iv < 25:
            recommendations.append(
                "Low IV environment - premiums are compressed. Consider waiting or using tighter strikes."
            )

        # Check liquidity
        low_liq = [o for o in opportunities if o.get('liquidity_score', 0) < 30]
        if len(low_liq) > len(opportunities) * 0.3:
            recommendations.append(
                "Many opportunities have low liquidity. Prioritize high-liquidity options to minimize slippage."
            )

        # DTE advice
        avg_dte = sum(o.get('dte', 30) for o in opportunities) / len(opportunities)
        if avg_dte < 14:
            recommendations.append(
                "Short DTE opportunities available - rapid theta decay but higher gamma risk."
            )
        elif avg_dte > 40:
            recommendations.append(
                "Longer DTE positions - slower theta decay but more time for adjustment if needed."
            )

        return recommendations

    async def generate_llm_analysis(self, opportunity: Dict[str, Any]) -> Optional[str]:
        """Generate LLM-powered detailed analysis for an opportunity"""
        llm = self._get_llm_service()
        if not llm:
            return None

        try:
            prompt = f"""Analyze this cash-secured put opportunity for wheel strategy trading:

Symbol: {opportunity.get('symbol')}
Current Price: ${opportunity.get('stock_price')}
Strike: ${opportunity.get('strike')}
Expiration: {opportunity.get('expiration')} ({opportunity.get('dte')} DTE)
Premium: ${opportunity.get('premium')} ({opportunity.get('premium_pct'):.2f}%)
Monthly Return: {opportunity.get('monthly_return'):.2f}%
Annual Return: {opportunity.get('annual_return'):.1f}%
IV: {opportunity.get('iv')}%
Delta: {opportunity.get('delta')}
OTM Distance: {opportunity.get('otm_pct'):.1f}%
AI Score: {opportunity.get('ai_score')}/100
Sector: {opportunity.get('sector')}

Provide a concise 3-4 sentence analysis covering:
1. Risk/reward assessment
2. Key factors to watch
3. Trade recommendation"""

            result = llm.generate(prompt=prompt, max_tokens=300)
            return result.get('text') if result else None

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return None


# Singleton instance
_ai_scanner = None


def get_ai_scanner() -> AIPremiumScanner:
    """Get AI scanner singleton"""
    global _ai_scanner
    if _ai_scanner is None:
        _ai_scanner = AIPremiumScanner()
    return _ai_scanner


# Quick test
if __name__ == "__main__":
    scanner = AIPremiumScanner()
    results = scanner.scan_with_ai(
        symbols=['AAPL', 'AMD', 'SOFI', 'PLTR'],
        max_price=250,
        min_premium_pct=0.5,
        dte=30,
        top_n=20
    )

    print(f"\n=== AI Premium Scanner Results ===")
    print(f"Total found: {results['total_found']}")
    print(f"Scan time: {results['scan_time_seconds']}s")
    print(f"\nStats: {results['stats']}")
    print(f"\nInsights: {results['ai_insights']}")

    if results['opportunities']:
        print(f"\nTop 5 Opportunities:")
        for opp in results['opportunities'][:5]:
            print(f"  {opp['symbol']} ${opp['strike']} - AI: {opp['ai_score']}/100 ({opp['ai_recommendation']}) - {opp['monthly_return']:.1f}%/mo")
