"""Test the database scanner fix"""
from src.database_scanner import DatabaseScanner

print("Testing Database Scanner Fix")
print("=" * 50)

scanner = DatabaseScanner()

if scanner.connect():
    print("\n[OK] Database connection successful")

    # Test get_all_stocks
    print("\nTesting get_all_stocks()...")
    stocks = scanner.get_all_stocks()

    if stocks:
        print(f"[OK] Found {len(stocks)} stocks")

        # Show first stock to verify column mapping
        print("\nFirst stock data:")
        first_stock = stocks[0]
        for key, value in first_stock.items():
            print(f"  - {key}: {value}")

        # Verify expected columns exist
        print("\nVerifying column mapping...")
        expected_columns = ['symbol', 'name', 'sector', 'current_price', 'market_cap', 'avg_volume']
        for col in expected_columns:
            if col in first_stock:
                print(f"  [OK] {col}: {first_stock[col]}")
            else:
                print(f"  [ERROR] {col}: MISSING")

        # Test filtering
        print("\nTesting get_stocks_by_criteria()...")
        filtered = scanner.get_stocks_by_criteria(max_price=100, min_volume=1000000)
        print(f"[OK] Filtered to {len(filtered)} stocks (price <= $100, volume >= 1M)")

        print("\n[OK] All tests passed!")

    else:
        print("[ERROR] No stocks returned")

    scanner.disconnect()
else:
    print("[ERROR] Failed to connect to database")
