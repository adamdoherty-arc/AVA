"""
Scheduled Premium Scanner Service
Runs background premium scanning twice daily (9:30 AM and 4:00 PM EST)

Can be run as:
1. Continuous service: python scheduled_premium_scanner.py
2. One-shot for Task Scheduler: python scheduled_premium_scanner.py --once
3. Test mode: python scheduled_premium_scanner.py --test
"""

import asyncio
import subprocess
import sys
import os
from datetime import datetime, time
from pathlib import Path
import argparse
import logging

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / 'logs' / 'scheduled_scanner.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
(Path(__file__).parent.parent / 'logs').mkdir(exist_ok=True)


class ScheduledPremiumScanner:
    """Scheduler for running premium scans at set times"""

    # Schedule times (Eastern Time)
    SCAN_TIMES = [
        time(9, 30),   # 9:30 AM - Market open
        time(12, 0),   # 12:00 PM - Midday
        time(16, 0),   # 4:00 PM - Market close
    ]

    def __init__(self) -> None:
        self.script_dir = Path(__file__).parent
        self.robinhood_scanner = self.script_dir / 'background_premium_scanner_robinhood.py'
        self.yfinance_scanner = self.script_dir / 'background_premium_scanner.py'

    async def run_scan(self, use_robinhood: bool = True, tradingview_only: bool = True):
        """Run a premium scan"""
        logger.info(f"Starting scheduled scan at {datetime.now()}")

        try:
            if use_robinhood and self.robinhood_scanner.exists():
                # Use Robinhood scanner (no rate limits)
                cmd = [sys.executable, str(self.robinhood_scanner)]
                if tradingview_only:
                    cmd.append('--tradingview')
                logger.info(f"Running Robinhood scanner: {' '.join(cmd)}")
            else:
                # Fallback to yfinance scanner
                cmd = [sys.executable, str(self.yfinance_scanner)]
                logger.info(f"Running yfinance scanner: {' '.join(cmd)}")

            # Run the scanner as subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.script_dir.parent)
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"Scan completed successfully")
                if stdout:
                    # Log last few lines of output
                    lines = stdout.decode().strip().split('\n')
                    for line in lines[-10:]:
                        logger.info(f"  {line}")
            else:
                logger.error(f"Scan failed with code {process.returncode}")
                if stderr:
                    logger.error(stderr.decode())

        except Exception as e:
            logger.error(f"Error running scan: {e}")

    def is_market_day(self) -> bool:
        """Check if today is a market day (weekday, not holiday)"""
        today = datetime.now()
        # Skip weekends
        if today.weekday() >= 5:
            return False
        # TODO: Add holiday calendar check if needed
        return True

    async def run_continuous(self) -> None:
        """Run continuously, checking for scheduled times"""
        logger.info("Starting scheduled premium scanner service")
        logger.info(f"Scan times: {[t.strftime('%H:%M') for t in self.SCAN_TIMES]}")

        last_scan_date = None
        scans_today = set()

        while True:
            now = datetime.now()
            current_time = now.time()
            current_date = now.date()

            # Reset daily tracking
            if current_date != last_scan_date:
                scans_today = set()
                last_scan_date = current_date
                logger.info(f"New day: {current_date}")

            # Check if we should scan
            if self.is_market_day():
                for scan_time in self.SCAN_TIMES:
                    scan_key = scan_time.strftime('%H:%M')

                    # Check if within 5 minutes of scheduled time and haven't scanned yet
                    time_diff_minutes = (
                        current_time.hour * 60 + current_time.minute -
                        scan_time.hour * 60 - scan_time.minute
                    )

                    if 0 <= time_diff_minutes < 5 and scan_key not in scans_today:
                        logger.info(f"Triggering scheduled scan for {scan_key}")
                        await self.run_scan()
                        scans_today.add(scan_key)

            # Sleep for 1 minute before checking again
            await asyncio.sleep(60)

    async def run_once(self) -> None:
        """Run a single scan (for Windows Task Scheduler)"""
        logger.info("Running one-shot scan")
        await self.run_scan()


async def main():
    parser = argparse.ArgumentParser(description='Scheduled Premium Scanner')
    parser.add_argument('--once', action='store_true', help='Run single scan and exit')
    parser.add_argument('--test', action='store_true', help='Run test scan immediately')
    parser.add_argument('--no-robinhood', action='store_true', help='Use yfinance instead of Robinhood')
    parser.add_argument('--all-symbols', action='store_true', help='Scan all symbols, not just TradingView')
    args = parser.parse_args()

    scanner = ScheduledPremiumScanner()

    if args.once or args.test:
        await scanner.run_scan(
            use_robinhood=not args.no_robinhood,
            tradingview_only=not args.all_symbols
        )
    else:
        await scanner.run_continuous()


if __name__ == '__main__':
    asyncio.run(main())
