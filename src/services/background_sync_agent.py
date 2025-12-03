"""
Background Sync Agent - Continuous Data Synchronization Service

Runs continuously in the background to sync:
- Robinhood positions (every 5 minutes)
- Portfolio balance snapshots
- Trade history
- Account information

Usage:
    python -m src.services.background_sync_agent

    Or with custom interval:
    python -m src.services.background_sync_agent --interval 300
"""

import time
import logging
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('background_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BackgroundSyncAgent:
    """
    Background service that continuously syncs Robinhood data.

    Syncs every 5 minutes (configurable):
    - Stock positions
    - Option positions
    - Account info (buying power, portfolio value)
    - Trade history (closed positions)
    - Portfolio balance snapshots
    """

    def __init__(self, sync_interval: int = 300):
        """
        Initialize background sync agent.

        Args:
            sync_interval: Seconds between sync cycles (default 300 = 5 minutes)
        """
        self.sync_interval = sync_interval
        self.is_running = False
        self.last_sync_time: Optional[datetime] = None
        self.sync_count = 0
        self.error_count = 0

        # Services (lazy initialization)
        self.rh_client = None
        self.trade_sync_service = None
        self.balance_tracker = None
        self.positions_connector = None

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Background Sync Agent initialized (interval: {sync_interval}s)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def _initialize_services(self) -> None:
        """Lazy initialization of services."""
        try:
            # Import here to avoid circular dependencies
            from src.services.robinhood_client import RobinhoodClient
            from src.trade_history_sync import TradeHistorySyncService
            from src.portfolio_balance_tracker import PortfolioBalanceTracker
            from src.services.positions_connector import PositionsConnector

            # Initialize services
            self.rh_client = RobinhoodClient()
            self.trade_sync_service = TradeHistorySyncService()
            self.balance_tracker = PortfolioBalanceTracker()
            self.positions_connector = PositionsConnector()

            logger.info("Services initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            return False

    def _sync_positions(self) -> Dict[str, Any]:
        """
        Sync Robinhood positions (stocks and options).

        Returns:
            Dictionary with sync results
        """
        results = {
            'success': False,
            'stock_positions': 0,
            'option_positions': 0,
            'error': None
        }

        try:
            logger.info("Syncing positions...")

            # Ensure connected
            if not self.rh_client.logged_in:
                logger.info("Connecting to Robinhood...")
                self.rh_client.login()

            # Fetch positions
            positions = self.rh_client.get_positions()

            if positions:
                stock_positions = positions.get('stocks', [])
                option_positions = positions.get('options', [])

                results['stock_positions'] = len(stock_positions)
                results['option_positions'] = len(option_positions)
                results['success'] = True

                logger.info(f"✓ Synced {len(stock_positions)} stock positions")
                logger.info(f"✓ Synced {len(option_positions)} option positions")

                # Cache positions for fast access
                self.positions_connector.invalidate_cache()

            else:
                logger.warning("No positions returned from Robinhood")
                results['success'] = True  # Not an error, just no positions

        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
            results['error'] = str(e)

        return results

    def _sync_account_info(self) -> Dict[str, Any]:
        """
        Sync Robinhood account information.

        Returns:
            Dictionary with sync results
        """
        results = {
            'success': False,
            'buying_power': None,
            'portfolio_value': None,
            'error': None
        }

        try:
            logger.info("Syncing account info...")

            account_info = self.rh_client.get_account_info()

            if account_info:
                results['buying_power'] = account_info.get('buying_power')
                results['portfolio_value'] = account_info.get('portfolio_value')
                results['success'] = True

                logger.info(f"✓ Buying Power: ${results['buying_power']:.2f}")
                logger.info(f"✓ Portfolio Value: ${results['portfolio_value']:.2f}")
            else:
                logger.warning("No account info returned")

        except Exception as e:
            logger.error(f"Error syncing account info: {e}")
            results['error'] = str(e)

        return results

    def _sync_trade_history(self) -> Dict[str, Any]:
        """
        Sync closed trades to database.

        Returns:
            Dictionary with sync results
        """
        results = {
            'success': False,
            'trades_synced': 0,
            'error': None
        }

        try:
            # Only sync trade history once per hour (not every 5 min)
            if self.last_sync_time:
                time_since_last = datetime.now() - self.last_sync_time
                if time_since_last < timedelta(hours=1):
                    logger.debug("Skipping trade history sync (synced recently)")
                    results['success'] = True
                    results['trades_synced'] = 0
                    return results

            logger.info("Syncing trade history...")

            # Import robin_stocks for trade sync (requires robin_stocks API calls)
            import robin_stocks.robinhood as rh

            # Sync trades
            trades_synced = self.trade_sync_service.sync_trades_from_robinhood(rh)

            results['trades_synced'] = trades_synced
            results['success'] = True

            logger.info(f"✓ Synced {trades_synced} trades to database")

        except Exception as e:
            logger.error(f"Error syncing trade history: {e}")
            results['error'] = str(e)

        return results

    def _sync_portfolio_balance(self) -> Dict[str, Any]:
        """
        Record portfolio balance snapshot.

        Returns:
            Dictionary with sync results
        """
        results = {
            'success': False,
            'balance_recorded': False,
            'error': None
        }

        try:
            logger.info("Recording portfolio balance...")

            # Get account info
            account_info = self.rh_client.get_account_info()

            if account_info:
                # Record daily balance
                from datetime import date
                self.balance_tracker.record_daily_balance(
                    balance_date=date.today(),
                    ending_balance=float(account_info.get('portfolio_value', 0)),
                    buying_power=float(account_info.get('buying_power', 0))
                )

                results['success'] = True
                results['balance_recorded'] = True

                logger.info("✓ Portfolio balance recorded")
            else:
                logger.warning("No account info for balance recording")

        except Exception as e:
            logger.error(f"Error recording portfolio balance: {e}")
            results['error'] = str(e)

        return results

    def _run_sync_cycle(self) -> None:
        """Execute one complete sync cycle."""
        cycle_start = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting sync cycle #{self.sync_count + 1}")
        logger.info(f"{'='*60}")

        try:
            # Sync positions (always)
            position_results = self._sync_positions()

            # Sync account info (always)
            account_results = self._sync_account_info()

            # Sync trade history (hourly)
            trade_results = self._sync_trade_history()

            # Record portfolio balance (always)
            balance_results = self._sync_portfolio_balance()

            # Check for errors
            all_success = all([
                position_results['success'],
                account_results['success'],
                trade_results['success'],
                balance_results['success']
            ])

            if all_success:
                self.sync_count += 1
                self.last_sync_time = datetime.now()
                logger.info(f"✓ Sync cycle #{self.sync_count} completed successfully")
            else:
                self.error_count += 1
                logger.warning(f"⚠ Sync cycle completed with errors (total errors: {self.error_count})")

            # Summary
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Cycle duration: {cycle_duration:.2f}s")
            logger.info(f"Next sync in {self.sync_interval}s")

        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in sync cycle: {e}", exc_info=True)

    def start(self) -> None:
        """Start the background sync agent."""
        if self.is_running:
            logger.warning("Agent is already running")
            return

        logger.info("Starting Background Sync Agent...")
        logger.info(f"Sync interval: {self.sync_interval}s ({self.sync_interval/60:.1f} minutes)")

        # Initialize services
        if not self._initialize_services():
            logger.error("Failed to start agent - service initialization failed")
            return

        self.is_running = True

        try:
            # Run first sync immediately
            self._run_sync_cycle()

            # Main loop
            while self.is_running:
                # Wait for next sync
                time.sleep(self.sync_interval)

                # Run sync cycle
                if self.is_running:  # Check again in case stopped during sleep
                    self._run_sync_cycle()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping...")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            self.stop()

    def stop(self) -> None:
        """Stop the background sync agent."""
        if not self.is_running:
            logger.warning("Agent is not running")
            return

        logger.info("Stopping Background Sync Agent...")
        self.is_running = False

        # Cleanup
        if self.rh_client:
            try:
                self.rh_client.logout()
                logger.info("✓ Logged out from Robinhood")
            except Exception as e:
                logger.warning(f"Error during logout: {e}")

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Agent Statistics:")
        logger.info(f"  Total sync cycles: {self.sync_count}")
        logger.info(f"  Total errors: {self.error_count}")
        if self.last_sync_time:
            logger.info(f"  Last successful sync: {self.last_sync_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")

        logger.info("Background Sync Agent stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            'is_running': self.is_running,
            'sync_interval': self.sync_interval,
            'sync_count': self.sync_count,
            'error_count': self.error_count,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'uptime': (datetime.now() - self.last_sync_time).total_seconds() if self.last_sync_time else None
        }


def main():
    """Main entry point for background sync agent."""
    parser = argparse.ArgumentParser(description='Background Sync Agent for Magnus Trading Platform')
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Sync interval in seconds (default: 300 = 5 minutes)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run sync once and exit (useful for testing)'
    )

    args = parser.parse_args()

    # Create agent
    agent = BackgroundSyncAgent(sync_interval=args.interval)

    if args.once:
        # Run once and exit
        logger.info("Running single sync cycle...")
        if agent._initialize_services():
            agent._run_sync_cycle()
        logger.info("Single sync completed, exiting")
    else:
        # Run continuously
        agent.start()


if __name__ == '__main__':
    main()
