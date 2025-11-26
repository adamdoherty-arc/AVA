"""Test TradingView integration to ensure it's working"""

from src.tradingview_real_integration import TradingViewRealIntegration
import json

def test_tradingview():
    print("Testing TradingView Integration...")
    print("=" * 50)

    # Initialize integration
    tv = TradingViewRealIntegration()

    # Test loading watchlists
    print("\n1. Current Watchlists:")
    for name, symbols in tv.watchlists.items():
        print(f"   - {name}: {len(symbols)} stocks")
        if symbols:
            print(f"     First 5: {symbols[:5]}")

    # Test importing NVDA watchlist
    nvda_symbols = "NVDA, AMD, AVGO, QCOM, MU, INTC, AMAT, LRCX, KLAC, ASML"
    print("\n2. Importing NVDA Watchlist:")
    imported = tv.import_from_text(nvda_symbols, "NVDA Watchlist")
    print(f"   Imported {len(imported)} symbols: {imported}")

    # Test getting stock data with changes
    print("\n3. Getting Stock Data with Price Changes:")
    stock_data = tv.get_stock_data_with_changes(imported[:3])
    for stock in stock_data:
        print(f"   {stock['symbol']}: ${stock['price']:.2f} ({stock['pct_change']:+.2f}%)")

    # Test options data
    print("\n4. Testing Options Premium Calculation:")
    if imported:
        test_symbol = imported[0]
        print(f"   Getting options for {test_symbol}...")

        # Use yfinance since we're testing
        options = tv.get_options_from_yfinance(test_symbol, stock_data[0]['price'] if stock_data else 100)

        if options and 'error' not in options:
            for exp_date, opt_data in list(options.items())[:2]:
                print(f"\n   Expiration: {exp_date} ({opt_data.get('days_to_expiry', 0)} days)")
                print(f"   Strike: ${opt_data.get('strike', 0):.2f}")
                print(f"   Premium: ${opt_data.get('premium', 0)*100:.2f}")
                print(f"   Return: {(opt_data.get('premium', 0)*100 / (opt_data.get('strike', 0)*100) * 100):.2f}%")
        else:
            print(f"   Error getting options: {options.get('error', 'Unknown')}")

    print("\n5. Testing Comprehensive Options Table:")
    table = tv.get_comprehensive_options_table(imported[:2])
    if table:
        print(f"   Generated table with {len(table)} rows")
        for row in table[:1]:  # Show first row
            print(f"   {row.get('Symbol')}: Price=${row.get('Price', 0):.2f}")
            # Show any option columns
            for key in row:
                if 'd Strike' in key or 'd Premium' in key or 'd Return%' in key:
                    print(f"     {key}: {row[key]}")

    print("\n✅ TradingView integration test complete!")

if __name__ == "__main__":
    test_tradingview()