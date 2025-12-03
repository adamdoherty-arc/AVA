"""
Sync Log Service - Track sync operations and failures

Provides logging and monitoring for all sync operations.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SyncLogService:
    """Service for logging sync operations"""
    
    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self._ensure_table_exists()
    
    def _ensure_table_exists(self) -> None:
        """Ensure sync_log table exists"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'sync_log'
                )
            """)
            
            if not cur.fetchone()[0]:
                # Create table
                schema_path = os.path.join(os.path.dirname(__file__), 'sync_log_schema.sql')
                if os.path.exists(schema_path):
                    with open(schema_path, 'r') as f:
                        schema_sql = f.read()
                    cur.execute(schema_sql)
                    conn.commit()
                    logger.info("Created sync_log table")
            
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error ensuring sync_log table exists: {e}")
    
    def start_sync(self, sync_type: str, metadata: Optional[Dict] = None) -> int:
        """Start a sync operation and return log ID"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO sync_log (sync_type, status, metadata)
                VALUES (%s, 'in_progress', %s)
                RETURNING id
            """, (sync_type, psycopg2.extras.Json(metadata or {})))
            
            log_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            
            return log_id
        except Exception as e:
            logger.error(f"Error starting sync log: {e}")
            return -1
    
    def complete_sync(
        self,
        log_id: int,
        status: str,
        items_processed: int = 0,
        items_successful: int = 0,
        items_failed: int = 0,
        error_message: Optional[str] = None
    ):
        """Complete a sync operation"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Calculate duration
            cur.execute("SELECT started_at FROM sync_log WHERE id = %s", (log_id,))
            started_at = cur.fetchone()
            duration = None
            if started_at:
                duration = int((datetime.now() - started_at[0]).total_seconds())
            
            cur.execute("""
                UPDATE sync_log
                SET status = %s,
                    items_processed = %s,
                    items_successful = %s,
                    items_failed = %s,
                    error_message = %s,
                    completed_at = NOW(),
                    duration_seconds = %s
                WHERE id = %s
            """, (status, items_processed, items_successful, items_failed, error_message, duration, log_id))
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error completing sync log: {e}")
    
    def get_recent_failures(self, sync_type: Optional[str] = None, limit: int = 10) -> list:
        """Get recent sync failures"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            if sync_type:
                cur.execute("""
                    SELECT * FROM sync_log
                    WHERE status = 'failed'
                      AND sync_type = %s
                      AND started_at > NOW() - INTERVAL '7 days'
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (sync_type, limit))
            else:
                cur.execute("""
                    SELECT * FROM sync_log
                    WHERE status = 'failed'
                      AND started_at > NOW() - INTERVAL '7 days'
                    ORDER BY started_at DESC
                    LIMIT %s
                """, (limit,))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting recent failures: {e}")
            return []
    
    def get_last_successful_sync(self, sync_type: str) -> Optional[Dict]:
        """Get last successful sync for a type"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT * FROM sync_log
                WHERE sync_type = %s
                  AND status = 'success'
                ORDER BY completed_at DESC
                LIMIT 1
            """, (sync_type,))
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting last successful sync: {e}")
            return None

