#!/usr/bin/env python3
"""
Earnings Sync CLI Tool

Command-line interface for syncing earnings data from Robinhood API

Usage:
    python sync_earnings.py --all                    # Sync all stocks
    python sync_earnings.py --symbol AAPL            # Sync single symbol
    python sync_earnings.py --symbols AAPL,NVDA,TSLA # Sync multiple symbols
    python sync_earnings.py --limit 100              # Sync first 100 stocks
    python sync_earnings.py --upcoming 7             # Show upcoming earnings (7 days)
    python sync_earnings.py --history AAPL           # Show earnings history
    python sync_earnings.py --beat-rate AAPL         # Show beat rate
    python sync_earnings.py --stats                  # Show sync statistics

Examples:
    # Full sync of all stocks in database
    python sync_earnings.py --all

    # Sync specific symbols with custom delay
    python sync_earnings.py --symbols AAPL,MSFT,GOOGL --delay 2

    # Show upcoming earnings in next 14 days
    python sync_earnings.py --upcoming 14

    # Get detailed history and beat rate for a stock
    python sync_earnings.py --symbol NVDA --show-history --show-beat-rate
"""

import argparse
import sys
import json
from datetime import datetime, date
from typing import List
from src.earnings_sync_service import EarningsSyncService
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_table_row(*columns, widths=None):
    """Print a formatted table row"""
    if widths is None:
        widths = [15] * len(columns)

    row = ""
    for col, width in zip(columns, widths):
        col_str = str(col)[:width]
        row += col_str.ljust(width) + " "
    print(row)


def sync_all_stocks(service: EarningsSyncService, limit: int = None, delay: float = 1.0):
    """Sync earnings for all stocks"""
    print_section("SYNCING ALL STOCKS")

    if limit:
        print(f"Syncing first {limit} stocks from database...")
    else:
        print("Syncing all stocks from database...")

    print(f"Rate limit delay: {delay}s between requests\n")

    # Run sync
    summary = service.sync_all_stocks_earnings(limit=limit, rate_limit_delay=delay)

    # Print results
    print("\n" + "-"*80)
    print("SYNC SUMMARY")
    print("-"*80)
    print(f"Total Stocks:          {summary['total_stocks']}")
    print(f"Successful:            {summary['successful']} ({summary['success_rate']})")
    print(f"Failed:                {summary['failed']}")
    print(f"No Data:               {summary['no_data']}")
    print(f"Historical Records:    {summary['total_historical_records']}")
    print(f"Upcoming Events:       {summary['total_upcoming_events']}")
    print("-"*80)


def sync_symbols(service: EarningsSyncService, symbols: List[str], show_details: bool = False):
    """Sync earnings for specific symbols"""
    print_section(f"SYNCING {len(symbols)} SYMBOLS")

    results = []
    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] Syncing {symbol}...", end=" ")
        result = service.sync_symbol_earnings(symbol)

        if result['status'] == 'success':
            print(f"OK - {result['historical_count']} historical, {result['upcoming_count']} upcoming")
        elif result['status'] == 'no_data':
            print("No data available")
        else:
            print(f"FAILED - {result.get('error', 'Unknown error')}")

        results.append(result)

    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    no_data = sum(1 for r in results if r['status'] == 'no_data')

    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Successful: {successful}/{len(symbols)}")
    print(f"Failed:     {failed}/{len(symbols)}")
    print(f"No Data:    {no_data}/{len(symbols)}")
    print("-"*80)


def show_upcoming_earnings(service: EarningsSyncService, days: int = 30):
    """Show upcoming earnings events"""
    print_section(f"UPCOMING EARNINGS (Next {days} Days)")

    events = service.get_upcoming_earnings(days_ahead=days)

    if not events:
        print("\nNo upcoming earnings found.")
        return

    # Group by date
    events_by_date = {}
    for event in events:
        event_date = event['earnings_date']
        if event_date not in events_by_date:
            events_by_date[event_date] = []
        events_by_date[event_date].append(event)

    # Print grouped events
    for event_date in sorted(events_by_date.keys()):
        print(f"\n{event_date} ({event_date.strftime('%A')})")
        print("-" * 80)

        print_table_row(
            "Symbol", "Time", "EPS Est", "Beat Rate", "Last Q Surprise",
            widths=[10, 8, 12, 12, 15]
        )
        print_table_row("-"*9, "-"*7, "-"*11, "-"*11, "-"*14, widths=[10, 8, 12, 12, 15])

        for event in events_by_date[event_date]:
            symbol = event['symbol']
            time = event.get('earnings_time', 'N/A').upper()
            eps_est = f"${event['eps_estimate']:.2f}" if event.get('eps_estimate') else "N/A"
            beat_rate = f"{event.get('historical_beat_rate_pct', 0)}%" if event.get('historical_beat_rate_pct') else "N/A"
            last_surprise = f"{event.get('last_quarter_surprise_pct', 0):+.1f}%" if event.get('last_quarter_surprise_pct') else "N/A"

            print_table_row(symbol, time, eps_est, beat_rate, last_surprise, widths=[10, 8, 12, 12, 15])

    print(f"\nTotal: {len(events)} upcoming earnings")


def show_earnings_history(service: EarningsSyncService, symbol: str, quarters: int = 8):
    """Show earnings history for a symbol"""
    print_section(f"EARNINGS HISTORY - {symbol} (Last {quarters} Quarters)")

    history = service.get_historical_earnings(symbol, limit=quarters)

    if not history:
        print(f"\nNo earnings history found for {symbol}")
        return

    # Print table header
    print_table_row(
        "Date", "Quarter", "Actual", "Estimate", "Surprise", "Beat/Miss",
        widths=[12, 10, 10, 10, 12, 10]
    )
    print_table_row("-"*11, "-"*9, "-"*9, "-"*9, "-"*11, "-"*9, widths=[12, 10, 10, 10, 12, 10])

    # Print records
    for record in history:
        date_str = str(record['report_date'])
        quarter = f"Q{record.get('fiscal_quarter', '?')} {record.get('fiscal_year', '?')}"
        actual = f"${record['eps_actual']:.2f}" if record.get('eps_actual') is not None else "N/A"
        estimate = f"${record['eps_estimate']:.2f}" if record.get('eps_estimate') is not None else "N/A"
        surprise = f"{record.get('eps_surprise_percent', 0):+.1f}%" if record.get('eps_surprise_percent') is not None else "N/A"
        beat_miss = record.get('beat_miss', 'unknown').upper()

        print_table_row(date_str, quarter, actual, estimate, surprise, beat_miss, widths=[12, 10, 10, 10, 12, 10])

    print(f"\nTotal: {len(history)} quarters")


def show_beat_rate(service: EarningsSyncService, symbol: str, quarters: int = 8):
    """Show beat rate for a symbol"""
    print_section(f"BEAT RATE ANALYSIS - {symbol}")

    beat_rate = service.calculate_beat_rate(symbol, lookback_quarters=quarters)

    print(f"\nSymbol:           {symbol}")
    print(f"Lookback:         {quarters} quarters")
    print(f"Beat Rate:        {beat_rate:.1f}%")

    # Interpretation
    if beat_rate >= 75:
        interpretation = "Strong consistent beater"
    elif beat_rate >= 60:
        interpretation = "Generally beats estimates"
    elif beat_rate >= 40:
        interpretation = "Mixed results"
    elif beat_rate >= 25:
        interpretation = "Generally misses estimates"
    else:
        interpretation = "Consistent misser"

    print(f"Interpretation:   {interpretation}")


def show_sync_stats(service: EarningsSyncService):
    """Show sync statistics"""
    print_section("SYNC STATISTICS")

    conn = service.get_db_connection()
    cur = conn.cursor()

    try:
        # Overall stats
        cur.execute("""
            SELECT
                COUNT(*) as total_symbols,
                COUNT(*) FILTER (WHERE last_sync_status = 'success') as successful,
                COUNT(*) FILTER (WHERE last_sync_status = 'failed') as failed,
                COUNT(*) FILTER (WHERE last_sync_status = 'no_data') as no_data,
                SUM(historical_quarters_found) as total_historical,
                SUM(upcoming_events_found) as total_upcoming,
                MAX(last_sync_at) as latest_sync,
                MIN(last_sync_at) as earliest_sync
            FROM earnings_sync_status
        """)

        stats = cur.fetchone()

        print("\nOverall Statistics:")
        print("-" * 80)
        print(f"Total Symbols Synced:     {stats[0]}")
        print(f"Successful Syncs:         {stats[1]}")
        print(f"Failed Syncs:             {stats[2]}")
        print(f"No Data:                  {stats[3]}")
        print(f"Total Historical Records: {stats[4] or 0}")
        print(f"Total Upcoming Events:    {stats[5] or 0}")
        print(f"Latest Sync:              {stats[6]}")
        print(f"Earliest Sync:            {stats[7]}")

        # Recent syncs
        cur.execute("""
            SELECT symbol, last_sync_at, last_sync_status, historical_quarters_found, upcoming_events_found
            FROM earnings_sync_status
            ORDER BY last_sync_at DESC
            LIMIT 10
        """)

        print("\nRecent Syncs (Last 10):")
        print("-" * 80)
        print_table_row("Symbol", "Sync Time", "Status", "Historical", "Upcoming", widths=[10, 20, 12, 12, 10])
        print_table_row("-"*9, "-"*19, "-"*11, "-"*11, "-"*9, widths=[10, 20, 12, 12, 10])

        for row in cur.fetchall():
            symbol, sync_time, status, hist, upcoming = row
            sync_time_str = sync_time.strftime("%Y-%m-%d %H:%M") if sync_time else "Never"
            print_table_row(symbol, sync_time_str, status, hist or 0, upcoming or 0, widths=[10, 20, 12, 12, 10])

    except Exception as e:
        print(f"Error fetching statistics: {e}")

    finally:
        cur.close()
        conn.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Earnings Sync Tool - Sync earnings data from Robinhood API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Sync operations
    parser.add_argument('--all', action='store_true', help='Sync all stocks')
    parser.add_argument('--symbol', type=str, help='Sync single symbol')
    parser.add_argument('--symbols', type=str, help='Sync multiple symbols (comma-separated)')
    parser.add_argument('--limit', type=int, help='Limit number of stocks to sync')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls (seconds)')

    # Query operations
    parser.add_argument('--upcoming', type=int, metavar='DAYS', help='Show upcoming earnings (days ahead)')
    parser.add_argument('--history', type=str, metavar='SYMBOL', help='Show earnings history')
    parser.add_argument('--beat-rate', type=str, metavar='SYMBOL', help='Show beat rate')
    parser.add_argument('--stats', action='store_true', help='Show sync statistics')

    # Options
    parser.add_argument('--quarters', type=int, default=8, help='Number of quarters for history/beat-rate (default: 8)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize service
    try:
        service = EarningsSyncService()
    except Exception as e:
        print(f"ERROR: Failed to initialize service: {e}")
        sys.exit(1)

    # Execute operations
    try:
        if args.all:
            sync_all_stocks(service, limit=args.limit, delay=args.delay)

        elif args.symbol:
            sync_symbols(service, [args.symbol.upper()])
            if args.history or args.beat_rate:
                if args.history:
                    show_earnings_history(service, args.symbol.upper(), quarters=args.quarters)
                if args.beat_rate:
                    show_beat_rate(service, args.symbol.upper(), quarters=args.quarters)

        elif args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            sync_symbols(service, symbols)

        elif args.upcoming is not None:
            show_upcoming_earnings(service, days=args.upcoming)

        elif args.history:
            show_earnings_history(service, args.history.upper(), quarters=args.quarters)

        elif args.beat_rate:
            show_beat_rate(service, args.beat_rate.upper(), quarters=args.quarters)

        elif args.stats:
            show_sync_stats(service)

        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
