#!/usr/bin/env python3
"""
Xtrades.net Scraper - Command Line Interface
=============================================
Simple CLI tool for scraping Xtrades.net profiles.

Usage:
    python xtrades_cli.py scrape <username> [--max-alerts=N] [--output=file.json]
    python xtrades_cli.py batch <file.txt> [--output=results.json]
    python xtrades_cli.py test [--username=behappy]
    python xtrades_cli.py --help

Examples:
    # Scrape single profile
    python xtrades_cli.py scrape behappy --max-alerts=20

    # Scrape multiple profiles from file
    python xtrades_cli.py batch profiles.txt --output=alerts.json

    # Test connection
    python xtrades_cli.py test
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from xtrades_scraper import (
    XtradesScraper,
    LoginFailedException,
    ProfileNotFoundException,
    scrape_profile
)


class XtradesCLI:
    """Command-line interface for Xtrades scraper"""

    def __init__(self) -> None:
        self.scraper = None

    def scrape_command(self, args):
        """Scrape a single profile"""
        print(f"\nScraping profile: {args.username}")
        print("="*60)

        try:
            # Scrape profile
            alerts = scrape_profile(args.username, max_alerts=args.max_alerts)

            print(f"\nFound {len(alerts)} alerts\n")

            # Display alerts
            for i, alert in enumerate(alerts, 1):
                self._display_alert(i, alert)

            # Save to file if requested
            if args.output:
                self._save_json(alerts, args.output)
                print(f"\nSaved to: {args.output}")

            # Summary
            self._display_summary(alerts)

            return 0

        except LoginFailedException as e:
            print(f"\n❌ Login failed: {e}")
            print("Please check your credentials in .env file")
            return 1
        except ProfileNotFoundException as e:
            print(f"\n❌ Profile not found: {e}")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def batch_command(self, args):
        """Scrape multiple profiles from file"""
        print(f"\nBatch scraping from: {args.file}")
        print("="*60)

        try:
            # Read usernames from file
            with open(args.file, 'r') as f:
                usernames = [line.strip() for line in f if line.strip()]

            print(f"Found {len(usernames)} profiles to scrape\n")

            # Initialize scraper (reuse session)
            self.scraper = XtradesScraper()
            self.scraper.login()

            # Scrape each profile
            results = {}
            total_alerts = 0

            for i, username in enumerate(usernames, 1):
                print(f"\n[{i}/{len(usernames)}] Scraping: {username}")
                print("-"*60)

                try:
                    alerts = self.scraper.get_profile_alerts(
                        username,
                        max_alerts=args.max_alerts
                    )

                    results[username] = {
                        'success': True,
                        'alert_count': len(alerts),
                        'alerts': alerts,
                        'scraped_at': datetime.now().isoformat()
                    }

                    total_alerts += len(alerts)
                    print(f"✓ Found {len(alerts)} alerts")

                except ProfileNotFoundException:
                    print(f"✗ Profile not found")
                    results[username] = {
                        'success': False,
                        'error': 'Profile not found'
                    }
                except Exception as e:
                    print(f"✗ Error: {e}")
                    results[username] = {
                        'success': False,
                        'error': str(e)
                    }

            # Save results
            if args.output:
                self._save_json(results, args.output)
                print(f"\n\nSaved results to: {args.output}")

            # Summary
            print("\n" + "="*60)
            print("BATCH SUMMARY")
            print("="*60)
            successful = sum(1 for r in results.values() if r.get('success'))
            print(f"Profiles processed: {len(usernames)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(usernames) - successful}")
            print(f"Total alerts: {total_alerts}")

            return 0

        except FileNotFoundError:
            print(f"\n❌ File not found: {args.file}")
            return 1
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            if self.scraper:
                self.scraper.close()

    def test_command(self, args):
        """Test scraper functionality"""
        print("\nTesting Xtrades scraper...")
        print("="*60)

        try:
            # Test login
            print("\n1. Testing login...")
            scraper = XtradesScraper()
            scraper.login()
            print("   ✓ Login successful")

            # Test scraping
            print(f"\n2. Testing profile scrape ({args.username})...")
            alerts = scraper.get_profile_alerts(args.username, max_alerts=5)
            print(f"   ✓ Found {len(alerts)} alerts")

            # Test parsing
            print("\n3. Testing alert parsing...")
            if alerts:
                alert = alerts[0]
                print(f"   ✓ Ticker: {alert['ticker']}")
                print(f"   ✓ Strategy: {alert['strategy']}")
                print(f"   ✓ Action: {alert['action']}")

            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED")
            print("="*60)

            scraper.close()
            return 0

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def _display_alert(self, number: int, alert: Dict):
        """Display single alert"""
        print(f"Alert #{number}:")
        print(f"  Ticker:    {alert['ticker'] or 'N/A'}")
        print(f"  Strategy:  {alert['strategy'] or 'N/A'}")
        print(f"  Action:    {alert['action'] or 'N/A'}")

        if alert['entry_price']:
            print(f"  Entry:     ${alert['entry_price']:.2f}")
        if alert['strike_price']:
            print(f"  Strike:    ${alert['strike_price']:.2f}")
        if alert['expiration_date']:
            print(f"  Expiry:    {alert['expiration_date']}")
        if alert['quantity']:
            print(f"  Quantity:  {alert['quantity']}")
        if alert['pnl']:
            print(f"  P&L:       ${alert['pnl']:.2f}")
        if alert['pnl_percent']:
            print(f"  P&L %:     {alert['pnl_percent']:+.1f}%")

        print(f"  Status:    {alert['status']}")
        print(f"  Text:      {alert['alert_text'][:80]}...")
        print()

    def _display_summary(self, alerts: List[Dict]):
        """Display summary statistics"""
        if not alerts:
            return

        print("="*60)
        print("SUMMARY")
        print("="*60)

        # Count by status
        open_trades = sum(1 for a in alerts if a['status'] == 'open')
        closed_trades = sum(1 for a in alerts if a['status'] == 'closed')

        print(f"Total alerts:   {len(alerts)}")
        print(f"Open trades:    {open_trades}")
        print(f"Closed trades:  {closed_trades}")

        # Count by strategy
        strategies = {}
        for alert in alerts:
            strat = alert['strategy'] or 'Unknown'
            strategies[strat] = strategies.get(strat, 0) + 1

        if strategies:
            print("\nStrategies:")
            for strat, count in sorted(strategies.items(), key=lambda x: x[1], reverse=True):
                print(f"  {strat}: {count}")

        # Total P&L
        total_pnl = sum(a['pnl'] for a in alerts if a['pnl'])
        if total_pnl != 0:
            print(f"\nTotal P&L: ${total_pnl:.2f}")

    def _save_json(self, data, filepath: str):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Xtrades.net Scraper CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scrape behappy --max-alerts=20
  %(prog)s scrape behappy --output=alerts.json
  %(prog)s batch profiles.txt --output=results.json
  %(prog)s test
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Scrape a single profile')
    scrape_parser.add_argument('username', help='Xtrades.net username')
    scrape_parser.add_argument(
        '--max-alerts',
        type=int,
        default=None,
        help='Maximum number of alerts to retrieve (default: all)'
    )
    scrape_parser.add_argument(
        '--output',
        help='Output JSON file'
    )

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Scrape multiple profiles')
    batch_parser.add_argument(
        'file',
        help='Text file with usernames (one per line)'
    )
    batch_parser.add_argument(
        '--max-alerts',
        type=int,
        default=None,
        help='Maximum alerts per profile (default: all)'
    )
    batch_parser.add_argument(
        '--output',
        default='batch_results.json',
        help='Output JSON file (default: batch_results.json)'
    )

    # Test command
    test_parser = subparsers.add_parser('test', help='Test scraper functionality')
    test_parser.add_argument(
        '--username',
        default='behappy',
        help='Username to test with (default: behappy)'
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run command
    cli = XtradesCLI()

    if args.command == 'scrape':
        return cli.scrape_command(args)
    elif args.command == 'batch':
        return cli.batch_command(args)
    elif args.command == 'test':
        return cli.test_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
