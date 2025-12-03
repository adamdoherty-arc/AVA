"""
Sync Status Service - Unified sync status tracking for all finance pages

Provides centralized access to sync status from all database tables.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Sync status levels"""
    FRESH = "fresh"      # < 1 hour
    RECENT = "recent"    # 1-24 hours
    STALE = "stale"      # > 24 hours
    NEVER = "never"      # No sync data


@dataclass
class SyncStatusResult:
    """Result of sync status query"""
    last_sync: Optional[datetime]
    total_items: int
    synced_today: int
    synced_this_week: int
    oldest_sync: Optional[datetime]
    status: SyncStatus
    hours_ago: float
    status_text: str
    status_color: str


class SyncStatusService:
    """Unified service for querying sync status from all tables"""
    
    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
    
    def _get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def _calculate_freshness(self, last_sync: Optional[datetime]) -> Tuple[SyncStatus, float, str, str]:
        """Calculate freshness status from last sync time"""
        if not last_sync:
            return SyncStatus.NEVER, 0.0, "Never synced", "⚪"
        
        # Handle timezone-aware datetimes
        if hasattr(last_sync, 'tzinfo') and last_sync.tzinfo:
            last_sync = last_sync.replace(tzinfo=None)
        
        now = datetime.now()
        hours_ago = (now - last_sync).total_seconds() / 3600
        
        if hours_ago < 1:
            minutes = int(hours_ago * 60)
            return SyncStatus.FRESH, hours_ago, f"Fresh ({minutes}m ago)", "🟢"
        elif hours_ago < 24:
            return SyncStatus.RECENT, hours_ago, f"Recent ({int(hours_ago)}h ago)", "🟡"
        else:
            days = int(hours_ago / 24)
            return SyncStatus.STALE, hours_ago, f"Stale ({days}d ago)", "🔴"
    
    def get_stock_data_sync_status(self) -> Optional[SyncStatusResult]:
        """Get sync status for stock_data table"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT 
                    MAX(last_updated) as last_sync,
                    COUNT(*) as total_items,
                    COUNT(CASE WHEN last_updated > NOW() - INTERVAL '24 hours' THEN 1 END) as synced_today,
                    COUNT(CASE WHEN last_updated > NOW() - INTERVAL '7 days' THEN 1 END) as synced_this_week,
                    MIN(last_updated) as oldest_sync
                FROM stock_data
            """)
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result and result['last_sync']:
                last_sync = result['last_sync']
                status, hours_ago, status_text, status_color = self._calculate_freshness(last_sync)
                
                return SyncStatusResult(
                    last_sync=last_sync,
                    total_items=result['total_items'] or 0,
                    synced_today=result['synced_today'] or 0,
                    synced_this_week=result['synced_this_week'] or 0,
                    oldest_sync=result['oldest_sync'],
                    status=status,
                    hours_ago=hours_ago,
                    status_text=status_text,
                    status_color=status_color
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock_data sync status: {e}")
            return None
    
    def get_stock_premiums_sync_status(self) -> Optional[SyncStatusResult]:
        """Get sync status for stock_premiums table"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT 
                    MAX(last_updated) as last_sync,
                    COUNT(DISTINCT symbol) as total_items,
                    COUNT(DISTINCT CASE WHEN last_updated > NOW() - INTERVAL '24 hours' THEN symbol END) as synced_today,
                    COUNT(DISTINCT CASE WHEN last_updated > NOW() - INTERVAL '7 days' THEN symbol END) as synced_this_week,
                    MIN(last_updated) as oldest_sync
                FROM stock_premiums
            """)
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result and result['last_sync']:
                last_sync = result['last_sync']
                status, hours_ago, status_text, status_color = self._calculate_freshness(last_sync)
                
                return SyncStatusResult(
                    last_sync=last_sync,
                    total_items=result['total_items'] or 0,
                    synced_today=result['synced_today'] or 0,
                    synced_this_week=result['synced_this_week'] or 0,
                    oldest_sync=result['oldest_sync'],
                    status=status,
                    hours_ago=hours_ago,
                    status_text=status_text,
                    status_color=status_color
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting stock_premiums sync status: {e}")
            return None
    
    def get_xtrades_sync_status(self) -> Optional[SyncStatusResult]:
        """Get sync status for xtrades_profiles table"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT 
                    MAX(last_sync) as last_sync,
                    COUNT(*) as total_items,
                    COUNT(CASE WHEN last_sync > NOW() - INTERVAL '24 hours' THEN 1 END) as synced_today,
                    COUNT(CASE WHEN last_sync > NOW() - INTERVAL '7 days' THEN 1 END) as synced_this_week,
                    MIN(last_sync) as oldest_sync
                FROM xtrades_profiles
                WHERE active = true
            """)
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result and result['last_sync']:
                last_sync = result['last_sync']
                status, hours_ago, status_text, status_color = self._calculate_freshness(last_sync)
                
                return SyncStatusResult(
                    last_sync=last_sync,
                    total_items=result['total_items'] or 0,
                    synced_today=result['synced_today'] or 0,
                    synced_this_week=result['synced_this_week'] or 0,
                    oldest_sync=result['oldest_sync'],
                    status=status,
                    hours_ago=hours_ago,
                    status_text=status_text,
                    status_color=status_color
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting xtrades sync status: {e}")
            return None
    
    def get_tradingview_sync_status(self) -> Optional[SyncStatusResult]:
        """Get sync status for TradingView watchlists"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT 
                    MAX(last_refresh) as last_sync,
                    COUNT(*) as total_items,
                    COUNT(CASE WHEN last_refresh > NOW() - INTERVAL '24 hours' THEN 1 END) as synced_today,
                    COUNT(CASE WHEN last_refresh > NOW() - INTERVAL '7 days' THEN 1 END) as synced_this_week,
                    MIN(last_refresh) as oldest_sync
                FROM tv_watchlists_api
            """)
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result and result['last_sync']:
                last_sync = result['last_sync']
                status, hours_ago, status_text, status_color = self._calculate_freshness(last_sync)
                
                return SyncStatusResult(
                    last_sync=last_sync,
                    total_items=result['total_items'] or 0,
                    synced_today=result['synced_today'] or 0,
                    synced_this_week=result['synced_this_week'] or 0,
                    oldest_sync=result['oldest_sync'],
                    status=status,
                    hours_ago=hours_ago,
                    status_text=status_text,
                    status_color=status_color
                )
            return None
            
        except Exception as e:
            logger.error(f"Error getting TradingView sync status: {e}")
            return None
    
    def get_all_sync_status(self) -> Dict[str, Optional[SyncStatusResult]]:
        """Get sync status for all tables"""
        return {
            'stock_data': self.get_stock_data_sync_status(),
            'stock_premiums': self.get_stock_premiums_sync_status(),
            'xtrades': self.get_xtrades_sync_status(),
            'tradingview': self.get_tradingview_sync_status()
        }
    
    def format_relative_time(self, dt: Optional[datetime]) -> str:
        """Format datetime as relative time string"""
        if not dt:
            return "Never"
        
        # Handle timezone-aware datetimes
        if hasattr(dt, 'tzinfo') and dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        
        if isinstance(dt, str):
            try:
                from dateutil import parser
                dt = parser.parse(dt)
            except:
                return "Unknown"
        
        now = datetime.now()
        diff = now - dt
        hours = diff.total_seconds() / 3600
        
        if hours < 1:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"
        elif hours < 24:
            return f"{int(hours)}h ago"
        else:
            days = int(hours / 24)
            return f"{days}d ago"

