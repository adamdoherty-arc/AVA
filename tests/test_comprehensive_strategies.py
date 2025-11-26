"""
Test script for Comprehensive Strategy Analyzer
Tests all 10 strategies with SOFI and AAPL
"""

import sys
from datetime import datetime
from src.ai_options_agent.comprehensive_strategy_analyzer import ComprehensiveStrategyAnalyzer
from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager


def test_comprehensive_analyzer():
    """Test the comprehensive strategy analyzer"""

    print("\n" + "="*80)
    print("COMPREHENSIVE STRATEGY ANALYZER TEST")
    print("="*80 + "\n")

    analyzer = ComprehensiveStrategyAnalyzer()
    db_manager = AIOptionsDBManager()

    # Test stocks
    test_symbols = ['SOFI', 'AAPL']

    for symbol in test_symbols:
        print(f"\n{'='*80}")
        print(f"Testing {symbol}")
        print(f"{'='*80}\n")

        try:
            # Get stock data from database
            opportunities = db_manager.get_opportunities(
                symbols=[symbol],
                dte_range=(20, 45),
                delta_range=(-0.50, -0.10),
                min_premium=0,
                limit=1
            )

            if not opportunities:
                print(f"[X] No options data found for {symbol} in database")
                print(f"   Skipping {symbol}\n")
                continue

            opp = opportunities[0]

            # Prepare data (convert Decimals to floats)
            current_price = float(opp.get('current_price', 0) or opp.get('stock_price', 100))
            iv_value = float(opp.get('iv', 0.35))
            market_cap_value = float(opp.get('market_cap', 0)) if opp.get('market_cap') else 0
            pe_ratio_value = float(opp.get('pe_ratio', 25)) if opp.get('pe_ratio') else 25

            stock_data = {
                'symbol': symbol,
                'current_price': current_price,
                'iv': iv_value,
                'price_52w_high': current_price * 1.2,
                'price_52w_low': current_price * 0.8,
                'market_cap': market_cap_value,
                'pe_ratio': pe_ratio_value,
                'sector': opp.get('sector', 'Unknown')
            }

            strike_price_value = float(opp.get('strike_price') or (current_price * 0.95))
            dte_value = int(opp.get('dte') or 30)
            delta_value = float(opp.get('delta') or -0.30)
            premium_value = float(opp.get('premium') or 0) / 100

            options_data = {
                'strike_price': strike_price_value,
                'dte': dte_value,
                'delta': delta_value,
                'premium': premium_value
            }

            print(f"[DATA] Stock Data:")
            print(f"   Current Price: ${stock_data['current_price']:.2f}")
            print(f"   IV: {stock_data['iv']*100:.1f}%")
            print(f"   Sector: {stock_data['sector']}")
            print(f"   Market Cap: ${stock_data['market_cap']/1e9:.2f}B")
            print(f"\n[DATA] Options Data:")
            print(f"   Strike: ${options_data['strike_price']:.2f}")
            print(f"   DTE: {options_data['dte']} days")
            print(f"   Delta: {options_data['delta']:.3f}")
            print(f"   Premium: ${options_data['premium']:.2f}")

            # Run analysis
            print(f"\n[RUNNING] Running comprehensive strategy analysis...")
            result = analyzer.analyze_stock(
                symbol=symbol,
                stock_data=stock_data,
                options_data=options_data
            )

            # Display results
            print(f"\n{'='*80}")
            print(f"RESULTS FOR {symbol}")
            print(f"{'='*80}\n")

            # Market Environment
            env = result['environment_analysis']
            print(f"[ENVIRONMENT] MARKET ENVIRONMENT:")
            print(f"   Volatility Regime: {env['volatility_regime'].upper()}")
            print(f"   Trend: {env['trend'].replace('_', ' ').title()}")
            print(f"   IV: {env['iv']*100:.1f}%")
            print(f"   Market Regime: {env['market_regime'].replace('_', ' ').title()}")
            print(f"   Price vs 52W High: {env.get('price_vs_52w_high', 0):.1f}%")
            print(f"   Price vs 52W Low: {env.get('price_vs_52w_low', 0):.1f}%")

            # Top 3 Strategies
            print(f"\n[TOP 3] TOP 3 STRATEGIES:\n")

            strategies = result['strategy_rankings']
            for idx, strategy in enumerate(strategies[:3], 1):
                print(f"#{idx}: {strategy['name']} - Score: {strategy['score']}/100")
                print(f"    Type: {strategy['type']} | Outlook: {strategy['outlook']}")
                print(f"    Best When: {strategy['best_when']}")
                print(f"    Win Rate: {strategy['win_rate']}")

                metrics = strategy['metrics']
                print(f"    Trade Legs:")
                for leg in metrics['legs']:
                    action = leg.get('action', '')
                    leg_type = leg.get('type', '')
                    strike = leg.get('strike', 0)
                    premium = leg.get('premium', 0)
                    if premium > 0:
                        print(f"      - {action} {leg_type} @ ${strike:.2f} (${premium:.2f} premium)")
                    else:
                        print(f"      - {action} {leg_type} @ ${strike:.2f}")

                # Show key metrics
                max_profit = metrics.get('max_profit', 0)
                max_loss = metrics.get('max_loss', 0)
                capital = metrics.get('capital_required', 0)

                if isinstance(max_profit, (int, float)):
                    print(f"    Max Profit: ${max_profit:.2f}")
                else:
                    print(f"    Max Profit: {max_profit}")

                if isinstance(max_loss, (int, float)):
                    print(f"    Max Loss: ${max_loss:.2f}")
                else:
                    print(f"    Max Loss: {max_loss}")

                if isinstance(capital, (int, float)):
                    print(f"    Capital Required: ${capital:.2f}")
                else:
                    print(f"    Capital Required: {capital}")

                print()

            # All 10 Strategies Summary
            print(f"[ALL STRATEGIES] ALL 10 STRATEGIES RANKED:\n")
            print(f"{'Rank':<6} {'Strategy':<30} {'Score':<8} {'Type':<10} {'Win Rate':<10}")
            print(f"{'-'*80}")

            for idx, strategy in enumerate(strategies, 1):
                print(f"{idx:<6} {strategy['name']:<30} {strategy['score']:<8} {strategy['type']:<10} {strategy['win_rate']:<10}")

            print(f"\n[OK] {symbol} analysis complete!")

        except Exception as e:
            print(f"[ERROR] Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_comprehensive_analyzer()
