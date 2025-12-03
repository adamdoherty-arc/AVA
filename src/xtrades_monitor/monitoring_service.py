"""
Xtrades Real-Time Monitoring Service
====================================

Main orchestrator that runs every 2.5 minutes to:
1. Scrape Xtrades profiles
2. Detect alert events (new/update/close)
3. Evaluate alerts with AI consensus
4. Send Telegram notifications for high-quality alerts

This is the entry point for the background monitoring system.
"""

import logging
import time
from typing import Dict, List, Any
from datetime import datetime
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.xtrades_monitor.alert_processor import AlertProcessor
from src.xtrades_monitor.ai_consensus import AIConsensusEngine
from src.xtrades_monitor.notification_service import TelegramNotificationService
from src.xtrades_scraper import XtradesScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xtrades_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class XtradesMonitoringService:
    """
    Main monitoring service orchestrator.

    Coordinates:
    - Scraping (XtradesScraper)
    - Alert processing (AlertProcessor)
    - AI evaluation (AIConsensusEngine)
    - Notifications (TelegramNotificationService)
    """

    def __init__(self, scrape_interval_seconds: int = 150):
        """
        Initialize monitoring service.

        Args:
            scrape_interval_seconds: How often to scrape (default: 150 = 2.5 minutes)
        """
        self.scrape_interval = scrape_interval_seconds

        logger.info("🚀 Initializing Xtrades Monitoring Service...")

        # Initialize components
        try:
            self.scraper = XtradesScraper()
            self.alert_processor = AlertProcessor()
            self.ai_engine = AIConsensusEngine()
            self.notification_service = TelegramNotificationService()

            logger.info("✅ All components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}", exc_info=True)
            raise

        # Track service stats
        self.stats = {
            'total_scrapes': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0,
            'total_alerts_processed': 0,
            'new_alerts': 0,
            'updated_alerts': 0,
            'closed_alerts': 0,
            'evaluations_completed': 0,
            'notifications_sent': 0,
            'high_quality_alerts': 0,  # Score >= 80
            'started_at': datetime.now()
        }

    def run_single_cycle(self) -> Dict[str, Any]:
        """
        Run a single monitoring cycle.

        Returns:
            Dict with cycle results and statistics
        """
        cycle_start = datetime.now()
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 Starting monitoring cycle at {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}\n")

        cycle_results = {
            'cycle_start': cycle_start,
            'profiles_scraped': 0,
            'alerts_detected': 0,
            'alerts_evaluated': 0,
            'notifications_sent': 0,
            'errors': []
        }

        try:
            # Step 1: Get list of profiles to monitor
            profiles = self._get_profiles_to_monitor()
            logger.info(f"📋 Monitoring {len(profiles)} profiles...")

            # Step 2: Scrape each profile
            all_alerts = []

            for profile_username in profiles:
                try:
                    logger.info(f"\n🔍 Scraping profile: @{profile_username}")
                    scraped_trades = self._scrape_profile(profile_username)

                    if scraped_trades:
                        logger.info(f"✅ Scraped {len(scraped_trades)} trades from @{profile_username}")

                        # Process alerts
                        alerts = self.alert_processor.process_scrape_results(
                            profile_username, scraped_trades
                        )

                        # Collect new alerts for evaluation
                        all_alerts.extend(alerts['new_alerts'])

                        cycle_results['profiles_scraped'] += 1
                        cycle_results['alerts_detected'] += (
                            len(alerts['new_alerts']) +
                            len(alerts['updated_alerts']) +
                            len(alerts['closed_alerts'])
                        )

                        self.stats['new_alerts'] += len(alerts['new_alerts'])
                        self.stats['updated_alerts'] += len(alerts['updated_alerts'])
                        self.stats['closed_alerts'] += len(alerts['closed_alerts'])

                    self.stats['successful_scrapes'] += 1

                except Exception as e:
                    logger.error(f"❌ Error scraping @{profile_username}: {e}")
                    cycle_results['errors'].append(f"Scrape error for @{profile_username}: {str(e)}")
                    self.stats['failed_scrapes'] += 1

            self.stats['total_scrapes'] += len(profiles)

            # Step 3: Evaluate new alerts with AI
            logger.info(f"\n🤖 Evaluating {len(all_alerts)} new alerts with AI consensus...")

            for alert in all_alerts:
                try:
                    # Enrich with market data
                    enriched_alert = self.alert_processor.enrich_alert_with_market_data(alert)

                    # Prepare for evaluation
                    prepared_alert = self.alert_processor.prepare_for_evaluation(enriched_alert)

                    # Evaluate with AI consensus
                    evaluation = self.ai_engine.evaluate_alert(prepared_alert)

                    # Save to database
                    alert_id = self.ai_engine.save_evaluation_to_database(
                        evaluation, alert['trade_id']
                    )

                    cycle_results['alerts_evaluated'] += 1
                    self.stats['evaluations_completed'] += 1

                    # Track high-quality alerts
                    if evaluation['consensus_score'] >= 80:
                        self.stats['high_quality_alerts'] += 1

                        # Queue notification if qualifies
                        if self.ai_engine.should_send_notification(evaluation):
                            notification_id = self.notification_service.queue_notification(
                                alert_id, evaluation
                            )

                            if notification_id:
                                logger.info(f"📬 Queued notification for alert {alert_id}")

                except Exception as e:
                    logger.error(f"❌ Error evaluating alert {alert['trade_id']}: {e}")
                    cycle_results['errors'].append(f"Evaluation error: {str(e)}")

            # Step 4: Send pending notifications
            logger.info(f"\n📤 Processing notification queue...")

            notification_stats = self.notification_service.send_pending_notifications()

            cycle_results['notifications_sent'] = notification_stats['sent']
            self.stats['notifications_sent'] += notification_stats['sent']

            # Step 5: Log cycle summary
            cycle_duration = (datetime.now() - cycle_start).total_seconds()

            logger.info(f"\n{'='*80}")
            logger.info(f"✅ Cycle complete in {cycle_duration:.1f}s")
            logger.info(f"   Profiles scraped: {cycle_results['profiles_scraped']}")
            logger.info(f"   Alerts detected: {cycle_results['alerts_detected']}")
            logger.info(f"   Alerts evaluated: {cycle_results['alerts_evaluated']}")
            logger.info(f"   Notifications sent: {cycle_results['notifications_sent']}")

            if cycle_results['errors']:
                logger.warning(f"   Errors: {len(cycle_results['errors'])}")

            logger.info(f"{'='*80}\n")

            cycle_results['cycle_duration_seconds'] = cycle_duration
            cycle_results['success'] = True

            return cycle_results

        except Exception as e:
            logger.error(f"❌ Critical error in monitoring cycle: {e}", exc_info=True)
            cycle_results['success'] = False
            cycle_results['errors'].append(f"Critical error: {str(e)}")
            return cycle_results

    def run_continuous(self) -> None:
        """
        Run monitoring service continuously.

        Executes cycles every scrape_interval seconds until interrupted.
        """
        logger.info(f"🚀 Starting continuous monitoring (interval: {self.scrape_interval}s)")
        logger.info(f"Press Ctrl+C to stop\n")

        cycle_number = 0

        try:
            while True:
                cycle_number += 1
                logger.info(f"📊 Cycle #{cycle_number}")

                # Run cycle
                results = self.run_single_cycle()

                # Print cumulative stats
                self._print_cumulative_stats()

                # Wait for next cycle
                logger.info(f"⏰ Waiting {self.scrape_interval}s until next cycle...\n")
                time.sleep(self.scrape_interval)

        except KeyboardInterrupt:
            logger.info(f"\n🛑 Monitoring service stopped by user")
            self._print_final_stats()

        except Exception as e:
            logger.error(f"❌ Fatal error in continuous monitoring: {e}", exc_info=True)
            self._print_final_stats()
            raise

    def _get_profiles_to_monitor(self) -> List[str]:
        """
        Get list of profile usernames to monitor.

        Queries database for active profiles.
        """
        try:
            import psycopg2
            import os

            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT username
                FROM xtrades_profiles
                WHERE is_active = TRUE
                ORDER BY username
            """)

            profiles = [row[0] for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            return profiles

        except Exception as e:
            logger.error(f"Error getting profiles: {e}")
            return []

    def _scrape_profile(self, profile_username: str) -> List[Dict[str, Any]]:
        """
        Scrape a single profile.

        Returns:
            List of trade dictionaries
        """
        try:
            # Use existing Xtrades scraper
            trades = self.scraper.scrape_following_alerts(profile_username)
            return trades

        except Exception as e:
            logger.error(f"Error scraping profile {profile_username}: {e}")
            return []

    def _print_cumulative_stats(self) -> None:
        """Print cumulative statistics"""
        uptime = (datetime.now() - self.stats['started_at']).total_seconds()
        uptime_hours = uptime / 3600

        logger.info(f"\n📊 Cumulative Statistics:")
        logger.info(f"   Uptime: {uptime_hours:.1f} hours")
        logger.info(f"   Total scrapes: {self.stats['total_scrapes']}")
        logger.info(f"   Success rate: {(self.stats['successful_scrapes'] / max(self.stats['total_scrapes'], 1)) * 100:.1f}%")
        logger.info(f"   New alerts: {self.stats['new_alerts']}")
        logger.info(f"   High-quality alerts: {self.stats['high_quality_alerts']} (score >= 80)")
        logger.info(f"   Evaluations: {self.stats['evaluations_completed']}")
        logger.info(f"   Notifications sent: {self.stats['notifications_sent']}")
        logger.info(f"")

    def _print_final_stats(self) -> None:
        """Print final statistics on shutdown"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Final Statistics")
        logger.info(f"{'='*80}")

        uptime = (datetime.now() - self.stats['started_at']).total_seconds()
        uptime_hours = uptime / 3600

        logger.info(f"\nSession Duration: {uptime_hours:.2f} hours")
        logger.info(f"\nScraping:")
        logger.info(f"  Total attempts: {self.stats['total_scrapes']}")
        logger.info(f"  Successful: {self.stats['successful_scrapes']}")
        logger.info(f"  Failed: {self.stats['failed_scrapes']}")
        logger.info(f"  Success rate: {(self.stats['successful_scrapes'] / max(self.stats['total_scrapes'], 1)) * 100:.1f}%")

        logger.info(f"\nAlerts:")
        logger.info(f"  Total processed: {self.stats['total_alerts_processed']}")
        logger.info(f"  New: {self.stats['new_alerts']}")
        logger.info(f"  Updated: {self.stats['updated_alerts']}")
        logger.info(f"  Closed: {self.stats['closed_alerts']}")
        logger.info(f"  High-quality (>=80): {self.stats['high_quality_alerts']}")

        logger.info(f"\nEvaluation & Notifications:")
        logger.info(f"  Evaluations completed: {self.stats['evaluations_completed']}")
        logger.info(f"  Notifications sent: {self.stats['notifications_sent']}")

        if self.stats['evaluations_completed'] > 0:
            avg_quality = (self.stats['high_quality_alerts'] / self.stats['evaluations_completed']) * 100
            logger.info(f"  Avg alert quality: {avg_quality:.1f}% high-quality")

        logger.info(f"\n{'='*80}\n")


def main():
    """Main entry point for monitoring service"""
    import argparse

    parser = argparse.ArgumentParser(description='Xtrades Real-Time Monitoring Service')
    parser.add_argument(
        '--interval',
        type=int,
        default=150,
        help='Scrape interval in seconds (default: 150 = 2.5 minutes)'
    )
    parser.add_argument(
        '--single-cycle',
        action='store_true',
        help='Run a single cycle and exit (for testing)'
    )

    args = parser.parse_args()

    # Initialize service
    service = XtradesMonitoringService(scrape_interval_seconds=args.interval)

    if args.single_cycle:
        # Run single cycle
        logger.info("🧪 Running single test cycle...")
        results = service.run_single_cycle()

        if results['success']:
            logger.info("✅ Test cycle completed successfully")
        else:
            logger.error("❌ Test cycle failed")
            sys.exit(1)
    else:
        # Run continuously
        service.run_continuous()


if __name__ == "__main__":
    main()
