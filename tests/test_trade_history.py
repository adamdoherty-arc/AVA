"""Test script for TradeHistoryManager"""
from src.trade_history_manager import TradeHistoryManager

# Initialize manager
th = TradeHistoryManager()
print("TradeHistoryManager imported successfully!")

# Test 1: Add a trade
trade_id = th.add_trade(
    symbol='NVDA',
    strike_price=180.00,
    expiration_date='2024-12-20',
    premium_collected=610.00,
    contracts=1,
    open_date='2024-10-01'
)
print(f"Test trade created with ID: {trade_id}")

# Test 2: Close the trade
result = th.close_trade(
    trade_id=trade_id,
    close_price=305.00,
    close_reason='early_close',
    close_date='2024-10-15'
)
print(f"Trade closed: P/L ${result['profit_loss']:.2f} ({result['profit_loss_percent']:.1f}%), {result['days_held']} days")
print(f"Annualized Return: {result['annualized_return']:.1f}%")

# Test 3: Get stats
stats = th.get_trade_stats()
print(f"\nTrade Statistics:")
print(f"  Total Trades: {stats['total_trades']}")
print(f"  Total P/L: ${stats['total_pl']:.2f}")
print(f"  Win Rate: {stats['win_rate']:.1f}%")
print(f"  Avg Days Held: {stats['avg_days_held']}")

# Test 4: Get closed trades
closed = th.get_closed_trades(limit=10)
print(f"\nClosed Trades: {len(closed)}")
for trade in closed:
    print(f"  {trade['symbol']} - P/L: ${trade['profit_loss']:.2f}")

print("\n✅ All tests passed!")
