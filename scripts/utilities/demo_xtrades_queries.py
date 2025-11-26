"""
Demo script showing Xtrades schema in action
Runs sample queries to demonstrate the database capabilities
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'magnus'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres123!')
}

def print_section(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def demo_queries():
    """Run demonstration queries"""

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    print_section("XTRADES WATCHLISTS - DATABASE DEMONSTRATION")

    # Query 1: Profile Overview
    print_section("1. PROFILE OVERVIEW WITH STATISTICS")
    cursor.execute("""
        SELECT
            p.id,
            p.username,
            p.display_name,
            p.active,
            COUNT(t.id) as total_trades,
            SUM(CASE WHEN t.status = 'open' THEN 1 ELSE 0 END) as open_positions,
            SUM(CASE WHEN t.status = 'closed' THEN 1 ELSE 0 END) as closed_trades,
            COALESCE(SUM(t.pnl), 0) as total_pnl
        FROM xtrades_profiles p
        LEFT JOIN xtrades_trades t ON p.id = t.profile_id
        GROUP BY p.id, p.username, p.display_name, p.active
        ORDER BY total_pnl DESC
    """)

    profiles = cursor.fetchall()
    print(f"{'ID':<5} {'Username':<20} {'Active':<8} {'Trades':<8} {'Open':<8} {'Closed':<8} {'P&L':<12}")
    print("-" * 80)
    for p in profiles:
        print(f"{p['id']:<5} {p['username']:<20} {'Yes' if p['active'] else 'No':<8} "
              f"{p['total_trades']:<8} {p['open_positions']:<8} {p['closed_trades']:<8} "
              f"${p['total_pnl']:,.2f}")

    # Query 2: Open Positions
    print_section("2. CURRENT OPEN POSITIONS")
    cursor.execute("""
        SELECT
            p.username,
            t.ticker,
            t.strategy,
            t.strike_price,
            t.expiration_date,
            t.entry_price,
            EXTRACT(DAY FROM (t.expiration_date::timestamp - CURRENT_DATE::timestamp)) as days_to_exp
        FROM xtrades_trades t
        JOIN xtrades_profiles p ON t.profile_id = p.id
        WHERE t.status = 'open'
        ORDER BY t.expiration_date
    """)

    positions = cursor.fetchall()
    if positions:
        print(f"{'Profile':<20} {'Ticker':<8} {'Strategy':<12} {'Strike':<10} "
              f"{'Expiration':<15} {'Premium':<10} {'DTE':<5}")
        print("-" * 80)
        for pos in positions:
            print(f"{pos['username']:<20} {pos['ticker']:<8} {pos['strategy']:<12} "
                  f"${pos['strike_price']:<9.2f} {str(pos['expiration_date']):<15} "
                  f"${pos['entry_price']:<9.2f} {int(pos['days_to_exp']):<5}")
    else:
        print("No open positions")

    # Query 3: Closed Trades Performance
    print_section("3. CLOSED TRADES WITH P&L")
    cursor.execute("""
        SELECT
            p.username,
            t.ticker,
            t.strategy,
            t.entry_date::date,
            t.exit_date::date,
            t.entry_price,
            t.exit_price,
            t.pnl,
            t.pnl_percent,
            EXTRACT(DAY FROM (t.exit_date::timestamp - t.entry_date::timestamp)) as days_held
        FROM xtrades_trades t
        JOIN xtrades_profiles p ON t.profile_id = p.id
        WHERE t.status = 'closed'
        ORDER BY t.exit_date DESC
    """)

    closed = cursor.fetchall()
    if closed:
        print(f"{'Profile':<20} {'Ticker':<8} {'Strategy':<12} {'Entry':<12} "
              f"{'Exit':<12} {'P&L':<12} {'Return %':<10} {'Days':<5}")
        print("-" * 80)
        for trade in closed:
            print(f"{trade['username']:<20} {trade['ticker']:<8} {trade['strategy']:<12} "
                  f"{str(trade['entry_date']):<12} {str(trade['exit_date']):<12} "
                  f"${trade['pnl']:>10.2f} {trade['pnl_percent']:>8.1f}% {int(trade['days_held']):<5}")
    else:
        print("No closed trades")

    # Query 4: Strategy Performance
    print_section("4. PERFORMANCE BY STRATEGY")
    cursor.execute("""
        SELECT
            strategy,
            COUNT(*) as total_trades,
            COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            AVG(pnl_percent) as avg_pnl_percent,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losers
        FROM xtrades_trades
        WHERE status = 'closed' AND pnl IS NOT NULL
        GROUP BY strategy
        ORDER BY total_pnl DESC
    """)

    strategies = cursor.fetchall()
    if strategies:
        print(f"{'Strategy':<15} {'Total':<8} {'Closed':<8} {'Winners':<10} "
              f"{'Total P&L':<15} {'Avg P&L':<12} {'Avg %':<10}")
        print("-" * 80)
        for strat in strategies:
            win_rate = (strat['winners'] / strat['closed'] * 100) if strat['closed'] > 0 else 0
            print(f"{strat['strategy']:<15} {strat['total_trades']:<8} {strat['closed']:<8} "
                  f"{strat['winners']}/{strat['losers']:<9} ${strat['total_pnl']:>12.2f} "
                  f"${strat['avg_pnl']:>10.2f} {strat['avg_pnl_percent']:>8.1f}%")
    else:
        print("No closed trades with P&L data")

    # Query 5: Recent Sync History
    print_section("5. RECENT SYNC HISTORY")
    cursor.execute("""
        SELECT
            sync_timestamp,
            profiles_synced,
            trades_found,
            new_trades,
            updated_trades,
            status,
            duration_seconds
        FROM xtrades_sync_log
        ORDER BY sync_timestamp DESC
        LIMIT 5
    """)

    syncs = cursor.fetchall()
    if syncs:
        print(f"{'Timestamp':<25} {'Profiles':<10} {'Found':<8} {'New':<8} "
              f"{'Updated':<10} {'Status':<10} {'Duration':<10}")
        print("-" * 80)
        for sync in syncs:
            print(f"{str(sync['sync_timestamp']):<25} {sync['profiles_synced']:<10} "
                  f"{sync['trades_found']:<8} {sync['new_trades']:<8} "
                  f"{sync['updated_trades']:<10} {sync['status']:<10} {sync['duration_seconds']:<10}s")
    else:
        print("No sync history")

    # Query 6: Overall System Stats
    print_section("6. SYSTEM STATISTICS DASHBOARD")
    cursor.execute("""
        SELECT
            (SELECT COUNT(*) FROM xtrades_profiles WHERE active = TRUE) as active_profiles,
            (SELECT COUNT(*) FROM xtrades_profiles WHERE active = FALSE) as inactive_profiles,
            (SELECT COUNT(*) FROM xtrades_trades) as total_trades,
            (SELECT COUNT(*) FROM xtrades_trades WHERE status = 'open') as open_positions,
            (SELECT COUNT(*) FROM xtrades_trades WHERE status = 'closed') as closed_trades,
            (SELECT COALESCE(SUM(pnl), 0) FROM xtrades_trades WHERE status = 'closed') as total_pnl,
            (SELECT COALESCE(AVG(pnl), 0) FROM xtrades_trades WHERE status = 'closed') as avg_pnl,
            (SELECT COALESCE(AVG(pnl_percent), 0) FROM xtrades_trades WHERE status = 'closed') as avg_return_pct,
            (SELECT COUNT(*) FROM xtrades_trades WHERE status = 'closed' AND pnl > 0) as winning_trades,
            (SELECT COUNT(*) FROM xtrades_trades WHERE status = 'closed' AND pnl < 0) as losing_trades,
            (SELECT MAX(sync_timestamp) FROM xtrades_sync_log) as last_sync,
            (SELECT COUNT(*) FROM xtrades_notifications) as total_notifications
    """)

    stats = cursor.fetchone()
    total_closed = stats['closed_trades']
    win_rate = (stats['winning_trades'] / total_closed * 100) if total_closed > 0 else 0

    print(f"Active Profiles:        {stats['active_profiles']}")
    print(f"Inactive Profiles:      {stats['inactive_profiles']}")
    print(f"Total Trades:           {stats['total_trades']}")
    print(f"  - Open:               {stats['open_positions']}")
    print(f"  - Closed:             {stats['closed_trades']}")
    print(f"\nPerformance Metrics:")
    print(f"Total P&L:              ${stats['total_pnl']:,.2f}")
    print(f"Average P&L:            ${stats['avg_pnl']:,.2f}")
    print(f"Average Return:         {stats['avg_return_pct']:.2f}%")
    print(f"Winning Trades:         {stats['winning_trades']}")
    print(f"Losing Trades:          {stats['losing_trades']}")
    print(f"Win Rate:               {win_rate:.1f}%")
    print(f"\nSystem Status:")
    print(f"Last Sync:              {stats['last_sync']}")
    print(f"Total Notifications:    {stats['total_notifications']}")

    print_section("DEMONSTRATION COMPLETE")
    print("All queries executed successfully!")
    print("\nThis demonstrates the Xtrades Watchlists schema is fully operational.")
    print("Ready for production use!")
    print("\n" + "="*80 + "\n")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    demo_queries()
