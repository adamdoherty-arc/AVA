"""
Test All Technical Indicators
Comprehensive test of all indicator modules
"""

import yfinance as yf
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.momentum_indicators import MomentumIndicators
from src.standard_indicators import StandardIndicators
from src.smart_money_indicators import SmartMoneyIndicators
from src.advanced_technical_indicators import VolumeProfileCalculator, OrderFlowAnalyzer
from src.options_indicators import OptionsIndicators


def test_all_indicators():
    """Test all indicator modules"""

    print("=" * 80)
    print("COMPREHENSIVE TECHNICAL INDICATORS TEST")
    print("=" * 80)

    # Fetch test data
    print("\nFetching AAPL data...")
    ticker = yf.Ticker('AAPL')
    df = ticker.history(period='3mo', interval='1d')
    df.columns = [col.lower() for col in df.columns]
    current_price = df['close'].iloc[-1]

    print(f"✅ Fetched {len(df)} days of data")
    print(f"Current Price: ${current_price:.2f}")

    results = {
        'passed': [],
        'failed': []
    }

    # Test 1: Momentum Indicators
    print("\n" + "=" * 80)
    print("1. Testing Momentum Indicators")
    print("-" * 80)
    try:
        mi = MomentumIndicators()
        momentum = mi.get_all_momentum_indicators(df, current_price)

        # Display key results
        rsi_sig = momentum['rsi']['signal']
        macd_sig = momentum['macd']['signal']
        ema_align = momentum['emas']['alignment']
        atr_stops = momentum['atr']['stops']

        print(f"RSI: {rsi_sig['value']:.1f} ({rsi_sig['signal']})")
        print(f"MACD: {macd_sig['signal']}")
        print(f"EMA Alignment: {ema_align['alignment']} ({ema_align['strength']})")
        print(f"ATR: ${atr_stops['atr']:.2f}")
        print(f"Stop Loss (Long): ${atr_stops['long_stop']:.2f}")

        print("✅ PASSED - Momentum Indicators")
        results['passed'].append('Momentum Indicators')

    except Exception as e:
        print(f"❌ FAILED - Momentum Indicators: {e}")
        results['failed'].append(('Momentum Indicators', str(e)))

    # Test 2: Standard Indicators
    print("\n" + "=" * 80)
    print("2. Testing Standard Indicators")
    print("-" * 80)
    try:
        si = StandardIndicators()
        standard = si.get_all_indicators(df, current_price)

        # Display key results
        bb_sig = standard['bollinger']['signal']
        stoch_sig = standard['stochastic']['signal']
        adx_sig = standard['adx']['signal']
        ich_sig = standard['ichimoku']['signal']

        print(f"Bollinger: {bb_sig['signal']} ({bb_sig['position']})")
        print(f"  Volatility: {bb_sig['volatility_state']}")
        print(f"Stochastic: {stoch_sig['signal']} ({stoch_sig['zone']})")
        print(f"  %K: {stoch_sig['k']:.1f}, %D: {stoch_sig['d']:.1f}")
        print(f"ADX: {adx_sig['signal']}")
        print(f"  Trend Strength: {adx_sig['trend_strength']} (ADX: {adx_sig['adx']:.1f})")
        print(f"Ichimoku: {ich_sig['signal']} ({ich_sig['cloud_position']})")

        print("✅ PASSED - Standard Indicators")
        results['passed'].append('Standard Indicators')

    except Exception as e:
        print(f"❌ FAILED - Standard Indicators: {e}")
        results['failed'].append(('Standard Indicators', str(e)))

    # Test 3: Smart Money Indicators
    print("\n" + "=" * 80)
    print("3. Testing Smart Money Indicators")
    print("-" * 80)
    try:
        smc = SmartMoneyIndicators()
        smart_money = smc.get_all_smc_indicators(df)

        # Display key results
        order_blocks = smart_money['order_blocks']
        fvgs = smart_money['fair_value_gaps']
        structure = smart_money['market_structure']
        liquidity = smart_money['liquidity_pools']

        print(f"Order Blocks: {len(order_blocks)} detected")
        if order_blocks:
            latest_ob = order_blocks[-1]
            print(f"  Latest: {latest_ob['type']} at ${latest_ob['midpoint']:.2f}")

        print(f"Fair Value Gaps: {len(fvgs)} detected")
        if fvgs:
            unfilled_fvgs = [f for f in fvgs if not f['filled']]
            print(f"  Unfilled: {len(unfilled_fvgs)}")

        print(f"Market Structure:")
        print(f"  Current Trend: {structure['current_trend']}")
        print(f"  BOS: {len(structure['bos'])}")
        print(f"  CHoCH: {len(structure['choch'])}")

        print(f"Liquidity Pools: {len(liquidity)} detected")

        print("✅ PASSED - Smart Money Indicators")
        results['passed'].append('Smart Money Indicators')

    except Exception as e:
        print(f"❌ FAILED - Smart Money Indicators: {e}")
        results['failed'].append(('Smart Money Indicators', str(e)))

    # Test 4: Volume Profile & Order Flow
    print("\n" + "=" * 80)
    print("4. Testing Volume Profile & Order Flow")
    print("-" * 80)
    try:
        vp_calc = VolumeProfileCalculator()
        volume_profile = vp_calc.calculate_volume_profile(df, price_bins=30)
        vp_signals = vp_calc.get_trading_signals(current_price, volume_profile)

        print(f"Volume Profile:")
        print(f"  POC: ${volume_profile['poc']['price']:.2f}")
        print(f"  VAH: ${volume_profile['vah']:.2f}")
        print(f"  VAL: ${volume_profile['val']:.2f}")
        print(f"  Value Area Width: {volume_profile['value_area_width_pct']:.2f}%")

        print(f"\nCurrent Price Analysis:")
        print(f"  Position: {vp_signals['position']}")
        print(f"  Bias: {vp_signals['bias']}")
        print(f"  Setup Quality: {vp_signals['setup_quality']}")

        # Order Flow
        of_analyzer = OrderFlowAnalyzer()
        cvd = of_analyzer.calculate_cvd(df)
        divergences = of_analyzer.find_cvd_divergences(df, lookback=10)

        print(f"\nOrder Flow (CVD):")
        print(f"  Latest CVD: {cvd.iloc[-1]:,.0f}")
        print(f"  CVD Trend: {'RISING' if cvd.iloc[-1] > cvd.iloc[-10] else 'FALLING'}")
        print(f"  Divergences: {len(divergences)}")

        print("✅ PASSED - Volume Profile & Order Flow")
        results['passed'].append('Volume Profile & Order Flow')

    except Exception as e:
        print(f"❌ FAILED - Volume Profile & Order Flow: {e}")
        results['failed'].append(('Volume Profile & Order Flow', str(e)))

    # Test 5: Options Indicators
    print("\n" + "=" * 80)
    print("5. Testing Options Indicators")
    print("-" * 80)
    try:
        oi = OptionsIndicators()

        # Test IVR
        iv_history = pd.Series([0.15, 0.18, 0.22, 0.25, 0.20, 0.30, 0.28, 0.24])
        current_iv = 0.27
        ivr = oi.implied_volatility_rank(current_iv, iv_history)

        print(f"Implied Volatility Rank:")
        print(f"  IVR: {ivr['ivr']:.1f}%")
        print(f"  Interpretation: {ivr['interpretation']}")
        print(f"  Strategy: {ivr['strategy']}")

        # Test Expected Move
        expected = oi.expected_move(price=current_price, iv=current_iv, dte=30)

        print(f"\nExpected Move (30 DTE):")
        print(f"  Move: ${expected['expected_move']:.2f} ({expected['move_pct']:.1f}%)")
        print(f"  Upper: ${expected['upper_bound']:.2f}")
        print(f"  Lower: ${expected['lower_bound']:.2f}")

        # Test Greeks
        greeks = oi.calculate_greeks(
            spot=current_price,
            strike=current_price * 1.05,
            rate=0.05,
            dte=30,
            iv=current_iv,
            option_type='call'
        )

        if 'error' not in greeks:
            print(f"\nOption Greeks (Call at 105% strike):")
            print(f"  Theoretical Price: ${greeks['price']:.2f}")
            print(f"  Delta: {greeks['delta']:.3f}")
            print(f"  Gamma: {greeks['gamma']:.4f}")
            print(f"  Theta: ${greeks['theta']:.4f}/day")
            print(f"  Vega: {greeks['vega']:.4f}")
        else:
            print(f"  Greeks calculation: {greeks['error']}")

        # Test Put/Call Ratio
        pcr = oi.put_call_ratio(put_volume=15000, call_volume=10000)

        print(f"\nPut/Call Ratio:")
        print(f"  PCR: {pcr['pcr']:.2f}")
        print(f"  Sentiment: {pcr['sentiment']}")

        # Test Strategy Recommendation
        recommendation = oi.option_strategy_recommendation(
            ivr=ivr['ivr'],
            trend='BULLISH',
            expected_move=expected
        )

        print(f"\nStrategy Recommendation:")
        top = recommendation['top_recommendation']
        if top:
            print(f"  Strategy: {top['strategy']}")
            print(f"  Reason: {top['reason']}")

        print("✅ PASSED - Options Indicators")
        results['passed'].append('Options Indicators')

    except Exception as e:
        print(f"❌ FAILED - Options Indicators: {e}")
        results['failed'].append(('Options Indicators', str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(f"\n✅ PASSED: {len(results['passed'])}/{len(results['passed']) + len(results['failed'])}")
    for module in results['passed']:
        print(f"  ✓ {module}")

    if results['failed']:
        print(f"\n❌ FAILED: {len(results['failed'])}/{len(results['passed']) + len(results['failed'])}")
        for module, error in results['failed']:
            print(f"  ✗ {module}")
            print(f"    Error: {error}")
    else:
        print("\n🎉 ALL TESTS PASSED!")

    print("\n" + "=" * 80)

    return len(results['failed']) == 0


if __name__ == "__main__":
    success = test_all_indicators()
    sys.exit(0 if success else 1)
