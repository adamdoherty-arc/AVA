"""
Options-Specific Technical Indicators
Implied Volatility Rank, Expected Move, Greeks, Put/Call Ratio, Delta Hedging
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class OptionsIndicators:
    """
    Options-specific technical indicators

    Includes:
    - Implied Volatility Rank (IVR)
    - Implied Volatility Percentile (IVP)
    - Expected Move
    - Put/Call Ratio
    - Greeks (Delta, Gamma, Theta, Vega, Rho)
    - Delta Hedge calculations
    - Volatility Skew analysis
    """

    def __init__(self) -> None:
        try:
            import mibian
            self.mibian = mibian
        except ImportError:
            logger.warning("mibian not installed. Greeks calculation will be limited.")
            self.mibian = None

    def implied_volatility_rank(
        self,
        current_iv: float,
        iv_history: pd.Series,
        lookback: int = 252
    ) -> Dict:
        """
        Calculate Implied Volatility Rank (IVR)

        IVR = (Current IV - Min IV) / (Max IV - Min IV) × 100

        Args:
            current_iv: Current implied volatility (decimal, e.g., 0.25 for 25%)
            iv_history: Historical IV series
            lookback: Lookback period (default 252 trading days = 1 year)

        Returns:
            Dictionary with IVR, interpretation, and recommendation
        """
        if len(iv_history) < lookback:
            lookback = len(iv_history)

        recent_iv = iv_history.iloc[-lookback:]

        iv_min = recent_iv.min()
        iv_max = recent_iv.max()

        if iv_max == iv_min:
            ivr = 50.0
        else:
            ivr = ((current_iv - iv_min) / (iv_max - iv_min)) * 100

        # Interpretation
        if ivr > 80:
            interpretation = 'VERY_HIGH'
            recommendation = 'IV is extremely high - excellent for selling premium (credit spreads, iron condors)'
            strategy = 'SELL_PREMIUM'
        elif ivr > 50:
            interpretation = 'HIGH'
            recommendation = 'IV is elevated - good for selling premium strategies'
            strategy = 'SELL_PREMIUM'
        elif ivr < 20:
            interpretation = 'VERY_LOW'
            recommendation = 'IV is extremely low - excellent for buying premium (debit spreads, long options)'
            strategy = 'BUY_PREMIUM'
        elif ivr < 50:
            interpretation = 'LOW'
            recommendation = 'IV is below average - favorable for buying premium'
            strategy = 'BUY_PREMIUM'
        else:
            interpretation = 'NEUTRAL'
            recommendation = 'IV is near average - no strong edge'
            strategy = 'NEUTRAL'

        return {
            'ivr': float(ivr),
            'current_iv': float(current_iv),
            'iv_min': float(iv_min),
            'iv_max': float(iv_max),
            'interpretation': interpretation,
            'strategy': strategy,
            'recommendation': recommendation
        }

    def implied_volatility_percentile(
        self,
        current_iv: float,
        iv_history: pd.Series,
        lookback: int = 252
    ) -> Dict:
        """
        Calculate Implied Volatility Percentile (IVP)

        IVP = % of days in lookback period where IV was below current IV

        Args:
            current_iv: Current implied volatility
            iv_history: Historical IV series
            lookback: Lookback period

        Returns:
            Dictionary with IVP and interpretation
        """
        if len(iv_history) < lookback:
            lookback = len(iv_history)

        recent_iv = iv_history.iloc[-lookback:]

        days_below = (recent_iv < current_iv).sum()
        ivp = (days_below / len(recent_iv)) * 100

        # Interpretation
        if ivp > 80:
            interpretation = 'VERY_HIGH'
            recommendation = 'IV higher than 80% of recent period - sell premium'
        elif ivp > 50:
            interpretation = 'HIGH'
            recommendation = 'IV higher than average - consider selling premium'
        elif ivp < 20:
            interpretation = 'VERY_LOW'
            recommendation = 'IV lower than 80% of recent period - buy premium'
        elif ivp < 50:
            interpretation = 'LOW'
            recommendation = 'IV lower than average - consider buying premium'
        else:
            interpretation = 'NEUTRAL'
            recommendation = 'IV at average levels'

        return {
            'ivp': float(ivp),
            'current_iv': float(current_iv),
            'interpretation': interpretation,
            'recommendation': recommendation
        }

    def expected_move(
        self,
        price: float,
        iv: float,
        dte: int,
        confidence: float = 0.68
    ) -> Dict:
        """
        Calculate expected move based on implied volatility

        Expected Move = Price × IV × sqrt(DTE / 365)

        Args:
            price: Current stock price
            iv: Implied volatility (decimal, e.g., 0.25 for 25%)
            dte: Days to expiration
            confidence: Confidence level (0.68 = 1 SD, 0.95 = 2 SD)

        Returns:
            Dictionary with expected move and bounds
        """
        # Standard deviation multiplier
        if confidence >= 0.95:
            std_mult = 2.0
            conf_pct = 95
        elif confidence >= 0.68:
            std_mult = 1.0
            conf_pct = 68
        else:
            std_mult = 1.0
            conf_pct = 68

        # Expected move (1 standard deviation)
        base_move = price * iv * np.sqrt(dte / 365)

        # Adjust for confidence level
        expected_move = base_move * std_mult

        upper_bound = price + expected_move
        lower_bound = price - expected_move

        move_pct = (expected_move / price) * 100

        return {
            'expected_move': float(expected_move),
            'upper_bound': float(upper_bound),
            'lower_bound': float(lower_bound),
            'move_pct': float(move_pct),
            'confidence': conf_pct,
            'dte': dte,
            'current_price': float(price),
            'iv': float(iv)
        }

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        rate: float,
        dte: int,
        iv: float,
        option_type: str = 'call'
    ) -> Dict:
        """
        Calculate option Greeks using Black-Scholes

        Args:
            spot: Current stock price
            strike: Option strike price
            rate: Risk-free interest rate (decimal, e.g., 0.05 for 5%)
            dte: Days to expiration
            iv: Implied volatility (decimal, e.g., 0.25 for 25%)
            option_type: 'call' or 'put'

        Returns:
            Dictionary with all Greeks and theoretical price
        """
        if self.mibian is None:
            return {
                'error': 'mibian library not installed',
                'delta': None,
                'gamma': None,
                'theta': None,
                'vega': None,
                'rho': None,
                'price': None
            }

        try:
            # Convert IV to percentage for mibian
            iv_pct = iv * 100

            # Create Black-Scholes option
            option = self.mibian.BS(
                [spot, strike, rate * 100, dte],
                volatility=iv_pct
            )

            if option_type.lower() == 'call':
                greeks = {
                    'delta': option.callDelta,
                    'gamma': option.gamma,
                    'theta': option.callTheta,
                    'vega': option.vega,
                    'rho': option.callRho,
                    'price': option.callPrice
                }
            else:  # put
                greeks = {
                    'delta': option.putDelta,
                    'gamma': option.gamma,
                    'theta': option.putTheta,
                    'vega': option.vega,
                    'rho': option.putRho,
                    'price': option.putPrice
                }

            # Add interpretation
            greeks['delta_interpretation'] = self._interpret_delta(
                greeks['delta'],
                option_type
            )
            greeks['gamma_interpretation'] = self._interpret_gamma(greeks['gamma'])
            greeks['theta_interpretation'] = self._interpret_theta(greeks['theta'])
            greeks['vega_interpretation'] = self._interpret_vega(greeks['vega'])

            return greeks

        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            return {
                'error': str(e),
                'delta': None,
                'gamma': None,
                'theta': None,
                'vega': None,
                'rho': None,
                'price': None
            }

    def _interpret_delta(self, delta: float, option_type: str) -> str:
        """Interpret delta value"""
        abs_delta = abs(delta)

        if abs_delta > 0.7:
            return f"Deep ITM - behaves like stock (Delta: {delta:.3f})"
        elif abs_delta > 0.5:
            return f"ITM - high probability of expiring ITM (Delta: {delta:.3f})"
        elif abs_delta > 0.3:
            return f"ATM - 50/50 probability (Delta: {delta:.3f})"
        else:
            return f"OTM - low probability of expiring ITM (Delta: {delta:.3f})"

    def _interpret_gamma(self, gamma: float) -> str:
        """Interpret gamma value"""
        if gamma > 0.05:
            return f"High gamma - delta changes rapidly (Gamma: {gamma:.4f})"
        elif gamma > 0.02:
            return f"Moderate gamma (Gamma: {gamma:.4f})"
        else:
            return f"Low gamma - delta stable (Gamma: {gamma:.4f})"

    def _interpret_theta(self, theta: float) -> str:
        """Interpret theta value"""
        if abs(theta) > 0.05:
            return f"High time decay - ${abs(theta):.2f}/day"
        elif abs(theta) > 0.02:
            return f"Moderate time decay - ${abs(theta):.2f}/day"
        else:
            return f"Low time decay - ${abs(theta):.2f}/day"

    def _interpret_vega(self, vega: float) -> str:
        """Interpret vega value"""
        if vega > 0.15:
            return f"High vega - sensitive to IV changes (Vega: {vega:.4f})"
        elif vega > 0.05:
            return f"Moderate vega (Vega: {vega:.4f})"
        else:
            return f"Low vega - less sensitive to IV (Vega: {vega:.4f})"

    def put_call_ratio(
        self,
        put_volume: float,
        call_volume: float
    ) -> Dict:
        """
        Calculate Put/Call Ratio

        PCR > 1.0 = More puts than calls (bearish sentiment)
        PCR < 1.0 = More calls than puts (bullish sentiment)

        Args:
            put_volume: Total put volume or open interest
            call_volume: Total call volume or open interest

        Returns:
            Dictionary with PCR and sentiment
        """
        if call_volume == 0:
            return {
                'pcr': None,
                'sentiment': 'UNKNOWN',
                'interpretation': 'No call volume data'
            }

        pcr = put_volume / call_volume

        # Sentiment interpretation
        if pcr > 1.5:
            sentiment = 'VERY_BEARISH'
            interpretation = 'Extremely high put activity - very bearish sentiment'
        elif pcr > 1.0:
            sentiment = 'BEARISH'
            interpretation = 'More puts than calls - bearish sentiment'
        elif pcr < 0.5:
            sentiment = 'VERY_BULLISH'
            interpretation = 'Extremely high call activity - very bullish sentiment'
        elif pcr < 0.7:
            sentiment = 'BULLISH'
            interpretation = 'More calls than puts - bullish sentiment'
        else:
            sentiment = 'NEUTRAL'
            interpretation = 'Balanced put/call activity'

        # Contrarian interpretation
        if pcr > 1.5:
            contrarian = 'Excessive bearishness may indicate bottom'
        elif pcr < 0.5:
            contrarian = 'Excessive bullishness may indicate top'
        else:
            contrarian = 'No extreme sentiment'

        return {
            'pcr': float(pcr),
            'put_volume': float(put_volume),
            'call_volume': float(call_volume),
            'sentiment': sentiment,
            'interpretation': interpretation,
            'contrarian_view': contrarian
        }

    def delta_hedge_calculation(
        self,
        position_delta: float,
        shares_per_contract: int = 100,
        current_price: float = None
    ) -> Dict:
        """
        Calculate shares needed to delta-hedge an options position

        Args:
            position_delta: Total delta of options position
            shares_per_contract: Shares per contract (default 100)
            current_price: Current stock price (optional)

        Returns:
            Dictionary with hedge details
        """
        shares_to_hedge = -position_delta * shares_per_contract

        action = 'BUY' if shares_to_hedge > 0 else 'SELL'
        quantity = abs(shares_to_hedge)

        result = {
            'shares_to_hedge': float(shares_to_hedge),
            'action': action,
            'quantity': float(quantity),
            'position_delta': float(position_delta)
        }

        if current_price:
            cost = abs(shares_to_hedge) * current_price
            result['hedge_cost'] = float(cost)
            result['current_price'] = float(current_price)

        return result

    def volatility_skew_analysis(
        self,
        otm_put_iv: float,
        atm_iv: float,
        otm_call_iv: float
    ) -> Dict:
        """
        Analyze volatility skew across strike prices

        Args:
            otm_put_iv: IV of OTM put (e.g., 95% strike)
            atm_iv: IV of ATM option
            otm_call_iv: IV of OTM call (e.g., 105% strike)

        Returns:
            Dictionary with skew analysis
        """
        put_skew = otm_put_iv - atm_iv
        call_skew = otm_call_iv - atm_iv

        # Interpretation
        if put_skew > 0.05:  # 5% higher
            put_interpretation = 'HIGH_PUT_SKEW'
            put_recommendation = 'OTM puts expensive - market fears downside'
        elif put_skew > 0.02:
            put_interpretation = 'MODERATE_PUT_SKEW'
            put_recommendation = 'Moderate downside protection premium'
        else:
            put_interpretation = 'NORMAL_PUT_SKEW'
            put_recommendation = 'Normal put pricing'

        if call_skew > 0.05:
            call_interpretation = 'HIGH_CALL_SKEW'
            call_recommendation = 'OTM calls expensive - market expects upside'
        elif call_skew > 0.02:
            call_interpretation = 'MODERATE_CALL_SKEW'
            call_recommendation = 'Moderate upside premium'
        else:
            call_interpretation = 'NORMAL_CALL_SKEW'
            call_recommendation = 'Normal call pricing'

        # Overall skew
        if put_skew > call_skew:
            overall_skew = 'PUT_SKEW'
            overall_interpretation = 'Market fears downside more than upside'
        elif call_skew > put_skew:
            overall_skew = 'CALL_SKEW'
            overall_interpretation = 'Market expects upside more than downside'
        else:
            overall_skew = 'BALANCED'
            overall_interpretation = 'Balanced skew'

        return {
            'put_skew': float(put_skew),
            'call_skew': float(call_skew),
            'put_interpretation': put_interpretation,
            'call_interpretation': call_interpretation,
            'overall_skew': overall_skew,
            'overall_interpretation': overall_interpretation,
            'put_recommendation': put_recommendation,
            'call_recommendation': call_recommendation
        }

    def option_strategy_recommendation(
        self,
        ivr: float,
        trend: str,
        expected_move: Dict
    ) -> Dict:
        """
        Recommend option strategy based on multiple factors

        Args:
            ivr: Implied Volatility Rank (0-100)
            trend: Market trend ('BULLISH', 'BEARISH', 'NEUTRAL')
            expected_move: Expected move dictionary

        Returns:
            Strategy recommendation
        """
        strategies = []

        # High IV strategies (sell premium)
        if ivr > 50:
            if trend == 'BULLISH':
                strategies.append({
                    'strategy': 'Bull Put Spread',
                    'reason': 'High IV + Bullish trend - sell premium below support',
                    'profit_potential': 'Limited',
                    'risk': 'Limited',
                    'ideal_for': 'High probability income'
                })
                strategies.append({
                    'strategy': 'Cash-Secured Put',
                    'reason': 'Collect premium while waiting to buy stock',
                    'profit_potential': 'Limited',
                    'risk': 'Substantial',
                    'ideal_for': 'Stock acquisition at discount'
                })
            elif trend == 'BEARISH':
                strategies.append({
                    'strategy': 'Bear Call Spread',
                    'reason': 'High IV + Bearish trend - sell premium above resistance',
                    'profit_potential': 'Limited',
                    'risk': 'Limited',
                    'ideal_for': 'High probability income'
                })
                strategies.append({
                    'strategy': 'Covered Call',
                    'reason': 'Collect premium on existing shares',
                    'profit_potential': 'Limited',
                    'risk': 'Substantial (owns stock)',
                    'ideal_for': 'Income generation'
                })
            else:  # NEUTRAL
                strategies.append({
                    'strategy': 'Iron Condor',
                    'reason': 'High IV + Range-bound - sell premium on both sides',
                    'profit_potential': 'Limited',
                    'risk': 'Limited',
                    'ideal_for': 'Range-bound high IV environment'
                })
                strategies.append({
                    'strategy': 'Strangle (Short)',
                    'reason': 'Sell OTM puts and calls',
                    'profit_potential': 'Limited',
                    'risk': 'Substantial',
                    'ideal_for': 'Experienced traders in range'
                })

        # Low IV strategies (buy premium)
        else:
            if trend == 'BULLISH':
                strategies.append({
                    'strategy': 'Long Call',
                    'reason': 'Low IV + Bullish trend - buy cheap premium',
                    'profit_potential': 'Unlimited',
                    'risk': 'Limited to premium',
                    'ideal_for': 'Strong directional conviction'
                })
                strategies.append({
                    'strategy': 'Bull Call Spread',
                    'reason': 'Lower cost bullish play',
                    'profit_potential': 'Limited',
                    'risk': 'Limited',
                    'ideal_for': 'Moderate bullish outlook'
                })
            elif trend == 'BEARISH':
                strategies.append({
                    'strategy': 'Long Put',
                    'reason': 'Low IV + Bearish trend - buy cheap puts',
                    'profit_potential': 'Substantial',
                    'risk': 'Limited to premium',
                    'ideal_for': 'Strong bearish conviction'
                })
                strategies.append({
                    'strategy': 'Bear Put Spread',
                    'reason': 'Lower cost bearish play',
                    'profit_potential': 'Limited',
                    'risk': 'Limited',
                    'ideal_for': 'Moderate bearish outlook'
                })
            else:  # NEUTRAL
                strategies.append({
                    'strategy': 'Straddle (Long)',
                    'reason': 'Low IV - buy cheap premium, expect big move',
                    'profit_potential': 'Unlimited',
                    'risk': 'Limited to premium',
                    'ideal_for': 'Expecting volatility expansion'
                })
                strategies.append({
                    'strategy': 'Strangle (Long)',
                    'reason': 'Cheaper than straddle, expect big move',
                    'profit_potential': 'Unlimited',
                    'risk': 'Limited to premium',
                    'ideal_for': 'Volatility play with less cost'
                })

        return {
            'ivr': ivr,
            'trend': trend,
            'expected_move_pct': expected_move['move_pct'],
            'strategies': strategies,
            'top_recommendation': strategies[0] if strategies else None
        }


if __name__ == "__main__":
    # Test
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("OPTIONS INDICATORS TEST")
    print("=" * 80)

    options = OptionsIndicators()

    # Test IVR
    print("\n1. Implied Volatility Rank:")
    print("-" * 80)
    iv_history = pd.Series([0.15, 0.18, 0.22, 0.25, 0.20, 0.30, 0.28, 0.24])
    current_iv = 0.27
    ivr_result = options.implied_volatility_rank(current_iv, iv_history)
    print(f"IVR: {ivr_result['ivr']:.1f}%")
    print(f"Interpretation: {ivr_result['interpretation']}")
    print(f"Strategy: {ivr_result['strategy']}")
    print(f"{ivr_result['recommendation']}")

    # Test Expected Move
    print("\n2. Expected Move:")
    print("-" * 80)
    expected = options.expected_move(price=150, iv=0.25, dte=30)
    print(f"Current Price: ${expected['current_price']:.2f}")
    print(f"Expected Move: ${expected['expected_move']:.2f} ({expected['move_pct']:.1f}%)")
    print(f"Upper Bound: ${expected['upper_bound']:.2f}")
    print(f"Lower Bound: ${expected['lower_bound']:.2f}")
    print(f"Confidence: {expected['confidence']}%")

    # Test Greeks
    print("\n3. Option Greeks (Call):")
    print("-" * 80)
    greeks = options.calculate_greeks(
        spot=150,
        strike=155,
        rate=0.05,
        dte=30,
        iv=0.25,
        option_type='call'
    )
    if 'error' not in greeks:
        print(f"Theoretical Price: ${greeks['price']:.2f}")
        print(f"Delta: {greeks['delta']:.3f}")
        print(f"  {greeks['delta_interpretation']}")
        print(f"Gamma: {greeks['gamma']:.4f}")
        print(f"  {greeks['gamma_interpretation']}")
        print(f"Theta: ${greeks['theta']:.4f}")
        print(f"  {greeks['theta_interpretation']}")
        print(f"Vega: {greeks['vega']:.4f}")
        print(f"  {greeks['vega_interpretation']}")
    else:
        print(f"Error: {greeks['error']}")

    # Test Put/Call Ratio
    print("\n4. Put/Call Ratio:")
    print("-" * 80)
    pcr = options.put_call_ratio(put_volume=15000, call_volume=10000)
    print(f"PCR: {pcr['pcr']:.2f}")
    print(f"Sentiment: {pcr['sentiment']}")
    print(f"{pcr['interpretation']}")
    print(f"Contrarian View: {pcr['contrarian_view']}")

    # Test Delta Hedge
    print("\n5. Delta Hedge Calculation:")
    print("-" * 80)
    hedge = options.delta_hedge_calculation(
        position_delta=0.65,
        current_price=150
    )
    print(f"Position Delta: {hedge['position_delta']}")
    print(f"Action: {hedge['action']} {hedge['quantity']:.0f} shares")
    print(f"Hedge Cost: ${hedge['hedge_cost']:,.2f}")

    # Test Strategy Recommendation
    print("\n6. Strategy Recommendation:")
    print("-" * 80)
    recommendation = options.option_strategy_recommendation(
        ivr=75,
        trend='BULLISH',
        expected_move=expected
    )
    print(f"IVR: {recommendation['ivr']:.1f}%")
    print(f"Trend: {recommendation['trend']}")
    print(f"Expected Move: {recommendation['expected_move_pct']:.1f}%")
    print(f"\nTop Recommendation:")
    top = recommendation['top_recommendation']
    if top:
        print(f"  Strategy: {top['strategy']}")
        print(f"  Reason: {top['reason']}")
        print(f"  Profit Potential: {top['profit_potential']}")
        print(f"  Risk: {top['risk']}")
        print(f"  Ideal For: {top['ideal_for']}")

    print("\n" + "=" * 80)
    print("Options Indicators Test Complete")
    print("=" * 80)
