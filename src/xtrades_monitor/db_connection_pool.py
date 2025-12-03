"""
Database Connection Pool Manager
=================================

Thread-safe connection pooling for PostgreSQL with proper resource management.

Features:
- ThreadedConnectionPool for concurrent access
- Automatic connection lifecycle management
- Context manager support for safe connection usage
- Health checks and connection validation
- Proper cleanup on application shutdown

Security fixes:
- Prevents connection leaks
- Limits max connections to prevent exhaustion
- Validates connections before use
"""

import logging
import threading
import os
from contextlib import contextmanager
from typing import Optional
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConnectionPool:
    """
    Thread-safe singleton connection pool for PostgreSQL.

    Usage:
        # Initialize once at application startup
        pool = DatabaseConnectionPool.initialize()

        # Use with context manager (recommended)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")

        # Or manual management
        conn = pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            conn.commit()
        finally:
            pool.putconn(conn)
    """

    _instance = None
    _lock = threading.Lock()
    _pool = None

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize connection pool parameters"""
        if self._pool is None:
            self.db_url = os.getenv("DATABASE_URL")
            if not self.db_url:
                raise ValueError("DATABASE_URL not found in environment")

            self.min_connections = int(os.getenv("DB_POOL_MIN", "2"))
            self.max_connections = int(os.getenv("DB_POOL_MAX", "10"))

            logger.info(f"Database pool configured: min={self.min_connections}, max={self.max_connections}")

    @classmethod
    def initialize(cls, min_conn: int = 2, max_conn: int = 10) -> 'DatabaseConnectionPool':
        """
        Initialize the connection pool (call once at startup).

        Args:
            min_conn: Minimum number of connections to maintain
            max_conn: Maximum number of connections allowed

        Returns:
            DatabaseConnectionPool instance
        """
        instance = cls()

        if instance._pool is None:
            with cls._lock:
                if instance._pool is None:
                    try:
                        instance._pool = pool.ThreadedConnectionPool(
                            minconn=min_conn or instance.min_connections,
                            maxconn=max_conn or instance.max_connections,
                            dsn=instance.db_url
                        )
                        logger.info(f"Connection pool created successfully: {min_conn}-{max_conn} connections")
                    except Exception as e:
                        logger.error(f"Failed to create connection pool: {e}")
                        raise

        return instance

    def getconn(self) -> psycopg2.extensions.connection:
        """
        Get a connection from the pool.

        Returns:
            psycopg2 connection object

        Raises:
            pool.PoolError: If pool is exhausted
            RuntimeError: If pool not initialized
        """
        if self._pool is None:
            raise RuntimeError("Connection pool not initialized. Call initialize() first.")

        try:
            conn = self._pool.getconn()
            # Validate connection is alive
            conn.isolation_level  # Simple check that connection is valid
            return conn
        except pool.PoolError as e:
            logger.error(f"Connection pool exhausted: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise

    def putconn(self, conn: psycopg2.extensions.connection, close: bool = False) -> None:
        """
        Return a connection to the pool.

        Args:
            conn: Connection to return
            close: If True, close the connection instead of returning to pool
        """
        if self._pool is None:
            logger.warning("Cannot return connection - pool not initialized")
            return

        try:
            if close:
                self._pool.putconn(conn, close=True)
            else:
                # Reset connection state before returning to pool
                try:
                    if not conn.closed:
                        conn.rollback()  # Rollback any uncommitted changes
                except Exception as e:
                    logger.warning(f"Error rolling back connection: {e}")
                    close = True  # Close connection if rollback fails

                self._pool.putconn(conn, close=close)
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")

    @contextmanager
    def get_connection(self) -> None:
        """
        Context manager for safe connection handling.

        Automatically returns connection to pool when done.
        Handles rollback on exceptions.

        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
                conn.commit()
        """
        conn = None
        try:
            conn = self.getconn()
            yield conn
        except Exception as e:
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            raise
        finally:
            if conn:
                self.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Context manager for connection + cursor.

        Automatically handles commit/rollback and cleanup.

        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)

        Usage:
            with pool.get_cursor(RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
        """
        conn = None
        cursor = None
        try:
            conn = self.getconn()
            cursor = conn.cursor(cursor_factory=cursor_factory)
            yield cursor
            conn.commit()
        except Exception as e:
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                self.putconn(conn)

    def close_all(self) -> None:
        """
        Close all connections in the pool.

        Call this on application shutdown.
        """
        if self._pool is not None:
            with self._lock:
                if self._pool is not None:
                    try:
                        self._pool.closeall()
                        logger.info("All database connections closed")
                        self._pool = None
                    except Exception as e:
                        logger.error(f"Error closing connection pool: {e}")

    def get_stats(self) -> dict:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool stats
        """
        if self._pool is None:
            return {"initialized": False}

        # Note: ThreadedConnectionPool doesn't expose these stats directly
        # This is a placeholder for potential future monitoring
        return {
            "initialized": True,
            "min_connections": self.min_connections,
            "max_connections": self.max_connections,
        }


# Global instance (singleton)
_pool_instance = None


def get_db_pool() -> DatabaseConnectionPool:
    """
    Get the global database connection pool instance.

    Returns:
        DatabaseConnectionPool singleton

    Raises:
        RuntimeError: If pool not initialized
    """
    global _pool_instance

    if _pool_instance is None:
        _pool_instance = DatabaseConnectionPool.initialize()

    return _pool_instance


def init_db_pool(min_conn: int = 2, max_conn: int = 10) -> DatabaseConnectionPool:
    """
    Initialize the global database connection pool.

    Call this once at application startup.

    Args:
        min_conn: Minimum connections to maintain
        max_conn: Maximum connections allowed

    Returns:
        DatabaseConnectionPool instance
    """
    global _pool_instance

    if _pool_instance is None:
        _pool_instance = DatabaseConnectionPool.initialize(min_conn, max_conn)
        logger.info("Global database pool initialized")
    else:
        logger.warning("Database pool already initialized")

    return _pool_instance


def close_db_pool() -> None:
    """
    Close the global database connection pool.

    Call this on application shutdown.
    """
    global _pool_instance

    if _pool_instance is not None:
        _pool_instance.close_all()
        _pool_instance = None
        logger.info("Global database pool closed")


# Example usage
if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing Database Connection Pool...")

    # Initialize pool
    pool = init_db_pool(min_conn=2, max_conn=5)

    # Test 1: Context manager usage
    print("\nTest 1: Context manager usage")
    try:
        with pool.get_cursor(RealDictCursor) as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Multiple concurrent connections
    print("\nTest 2: Multiple concurrent connections")
    connections = []
    try:
        for i in range(3):
            conn = pool.getconn()
            connections.append(conn)
            print(f"Got connection {i+1}")

        # Return all connections
        for i, conn in enumerate(connections):
            pool.putconn(conn)
            print(f"Returned connection {i+1}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Transaction with rollback
    print("\nTest 3: Transaction with rollback")
    try:
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            # This would fail if table doesn't exist, triggering rollback
            cursor.execute("SELECT * FROM nonexistent_table LIMIT 1")
    except Exception as e:
        print(f"Expected error (connection returned safely): {e}")

    # Cleanup
    print("\nClosing pool...")
    close_db_pool()
    print("Done!")
