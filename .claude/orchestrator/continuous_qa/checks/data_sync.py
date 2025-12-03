"""
Data Sync Check Module

Validates data freshness from all sources to ensure:
1. Data is not stale beyond acceptable thresholds
2. Sync mechanisms are working properly
3. No data sources are disconnected
4. Cache is being properly invalidated
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)

# Try to import database libraries
try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class DataSyncCheck(BaseCheck):
    """
    Validates data freshness and sync status across all data sources.

    CRITICAL check - Stale data in a financial tool is dangerous.
    """

    # Default freshness thresholds
    DEFAULT_THRESHOLDS = {
        'robinhood_positions': {'max_stale_minutes': 5, 'table': 'robinhood_positions'},
        'robinhood_options': {'max_stale_minutes': 5, 'table': 'robinhood_options'},
        'kalshi_markets': {'max_stale_minutes': 2, 'table': 'kalshi_markets'},
        'kalshi_positions': {'max_stale_minutes': 5, 'table': 'kalshi_positions'},
        'espn_games': {'max_stale_minutes': 5, 'table': 'espn_games'},
        'nfl_games': {'max_stale_minutes': 10, 'table': 'nfl_games'},
        'earnings_calendar': {'max_stale_hours': 24, 'table': 'earnings_calendar'},
        'xtrades_alerts': {'max_stale_hours': 1, 'table': 'xtrades_alerts'},
        'tradingview_watchlist': {'max_stale_hours': 4, 'table': 'tradingview_watchlist'},
    }

    def __init__(self, project_root: Path = None, db_url: str = None):
        """
        Initialize data sync check.

        Args:
            project_root: Root directory of the project
            db_url: Database connection URL
        """
        super().__init__()
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent.parent
        self.db_url = db_url
        self._connection = None

    @property
    def name(self) -> str:
        return "data_sync"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.CRITICAL

    def get_checks_list(self) -> List[str]:
        return [
            "database_connectivity",
            "robinhood_data_freshness",
            "kalshi_data_freshness",
            "sports_data_freshness",
            "earnings_data_freshness",
            "cache_health",
        ]

    def run(self) -> ModuleCheckResult:
        """Run data sync checks."""
        self._start_module()

        # Check database connectivity
        db_ok = self._check_database_connectivity()
        if not db_ok:
            self._fail(
                "database_connectivity",
                "Cannot connect to database",
                details={'error': 'Connection failed'},
            )
            # Skip all data freshness checks
            for check in self.get_checks()[1:]:
                self._skip(check, "Database not available")
            return self._end_module()

        self._pass("database_connectivity", "Database connection successful")

        # Check Robinhood data freshness
        rh_result = self._check_data_freshness([
            'robinhood_positions',
            'robinhood_options',
        ])
        self._report_freshness("robinhood_data_freshness", "Robinhood", rh_result)

        # Check Kalshi data freshness
        kalshi_result = self._check_data_freshness([
            'kalshi_markets',
            'kalshi_positions',
        ])
        self._report_freshness("kalshi_data_freshness", "Kalshi", kalshi_result)

        # Check sports data freshness
        sports_result = self._check_data_freshness([
            'espn_games',
            'nfl_games',
        ])
        self._report_freshness("sports_data_freshness", "Sports", sports_result)

        # Check earnings data freshness
        earnings_result = self._check_data_freshness([
            'earnings_calendar',
        ])
        self._report_freshness("earnings_data_freshness", "Earnings", earnings_result)

        # Check cache health
        cache_result = self._check_cache_health()
        if cache_result['healthy']:
            self._pass("cache_health", f"Cache healthy: {cache_result.get('message', 'OK')}")
        else:
            self._warn(
                "cache_health",
                f"Cache issues: {cache_result.get('message', 'Unknown')}",
                details=cache_result,
            )

        # Close connection
        self._close_connection()

        return self._end_module()

    def _check_database_connectivity(self) -> bool:
        """Check if we can connect to the database."""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 not available")
            return False

        try:
            # Try to get connection from environment or config
            db_url = self.db_url or self._get_db_url()
            if not db_url:
                logger.warning("No database URL configured")
                return False

            self._connection = psycopg2.connect(db_url)
            return True

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def _get_db_url(self) -> Optional[str]:
        """Get database URL from environment or config."""
        import os

        # Check environment
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
            return db_url

        # Check .env file
        env_file = self.project_root / '.env'
        if env_file.exists():
            try:
                content = env_file.read_text()
                for line in content.split('\n'):
                    if line.startswith('DATABASE_URL='):
                        return line.split('=', 1)[1].strip().strip('"\'')
            except Exception:
                pass

        # Default local postgres
        return "postgresql://localhost/magnus"

    def _check_data_freshness(self, data_sources: List[str]) -> Dict[str, Any]:
        """Check freshness of specified data sources."""
        result = {
            'sources_checked': 0,
            'stale_sources': [],
            'fresh_sources': [],
            'missing_sources': [],
        }

        for source in data_sources:
            config = self.DEFAULT_THRESHOLDS.get(source)
            if not config:
                continue

            table = config.get('table', source)
            max_stale_minutes = config.get('max_stale_minutes')
            max_stale_hours = config.get('max_stale_hours')

            if max_stale_hours:
                max_stale_minutes = max_stale_hours * 60

            # Check last update time
            last_update = self._get_last_update_time(table)

            result['sources_checked'] += 1

            if last_update is None:
                result['missing_sources'].append({
                    'source': source,
                    'table': table,
                    'issue': 'Table not found or empty',
                })
            else:
                age_minutes = (datetime.utcnow() - last_update).total_seconds() / 60

                if age_minutes > max_stale_minutes:
                    result['stale_sources'].append({
                        'source': source,
                        'table': table,
                        'age_minutes': round(age_minutes, 1),
                        'max_minutes': max_stale_minutes,
                        'last_update': last_update.isoformat(),
                    })
                else:
                    result['fresh_sources'].append({
                        'source': source,
                        'age_minutes': round(age_minutes, 1),
                    })

        return result

    def _get_last_update_time(self, table: str) -> Optional[datetime]:
        """Get the last update time for a table."""
        if not self._connection:
            return None

        try:
            cursor = self._connection.cursor()

            # Try common timestamp columns
            for col in ['updated_at', 'created_at', 'timestamp', 'last_updated', 'sync_time']:
                try:
                    cursor.execute(f"""
                        SELECT MAX({col}) FROM {table}
                        WHERE {col} IS NOT NULL
                    """)
                    result = cursor.fetchone()
                    if result and result[0]:
                        return result[0]
                except Exception:
                    continue

            cursor.close()
            return None

        except Exception as e:
            logger.warning(f"Error checking {table}: {e}")
            return None

    def _report_freshness(self, check_name: str, category: str, result: Dict[str, Any]):
        """Report freshness check results."""
        stale = result.get('stale_sources', [])
        missing = result.get('missing_sources', [])
        fresh = result.get('fresh_sources', [])

        if not stale and not missing:
            if fresh:
                self._pass(
                    check_name,
                    f"{category} data is fresh ({len(fresh)} sources checked)"
                )
            else:
                self._warn(
                    check_name,
                    f"No {category} data sources found to check",
                )
        elif stale:
            self._fail(
                check_name,
                f"{len(stale)} {category} data sources are stale",
                details={
                    'stale': stale,
                    'missing': missing,
                },
            )
        else:
            self._warn(
                check_name,
                f"{len(missing)} {category} data sources are missing",
                details={'missing': missing},
            )

    def _check_cache_health(self) -> Dict[str, Any]:
        """Check Redis cache health if available."""
        result = {'healthy': True, 'message': 'Cache not configured'}

        if not REDIS_AVAILABLE:
            return result

        try:
            import os
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')

            r = redis.from_url(redis_url, socket_connect_timeout=5)
            info = r.info()

            # Check memory usage
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)

            if max_memory > 0:
                memory_pct = (used_memory / max_memory) * 100
                if memory_pct > 90:
                    result['healthy'] = False
                    result['message'] = f'Cache memory at {memory_pct:.1f}%'
                    return result

            # Check hit rate
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses

            if total > 100:
                hit_rate = hits / total
                if hit_rate < 0.5:
                    result['healthy'] = False
                    result['message'] = f'Low cache hit rate: {hit_rate:.1%}'
                    return result

            result['message'] = f'Cache healthy (hit rate: {hits}/{total})'
            return result

        except redis.ConnectionError:
            result['message'] = 'Redis not available'
            return result
        except Exception as e:
            result['message'] = f'Cache check error: {str(e)[:50]}'
            return result

    def _close_connection(self) -> None:
        """Close database connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def can_auto_fix(self, check_name: str) -> bool:
        """Data sync issues generally cannot be auto-fixed."""
        # Could potentially trigger a sync, but that's risky
        return False

    def trigger_sync(self, source: str) -> bool:
        """
        Manually trigger a sync for a data source.

        This is NOT used in auto-fix but can be called manually.
        """
        sync_commands = {
            'robinhood': 'python -m src.robinhood_sync',
            'kalshi': 'python -m src.kalshi_sync',
            'espn': 'python -m src.espn_sync',
            'earnings': 'python -m src.earnings_sync',
        }

        cmd = sync_commands.get(source)
        if not cmd:
            return False

        try:
            import subprocess
            result = subprocess.run(
                cmd.split(),
                cwd=self.project_root,
                capture_output=True,
                timeout=60,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Sync trigger failed for {source}: {e}")
            return False
