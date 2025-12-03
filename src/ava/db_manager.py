"""
AVA Database Manager
====================

Thread-safe connection pooling for PostgreSQL with comprehensive error handling.

Features:
- ThreadedConnectionPool for concurrent access (min 2, max 10 connections)
- Context managers for automatic cleanup and rollback
- Proper error handling with user-friendly messages
- Connection validation and health checks
- Automatic retry logic with exponential backoff
- Connection lifecycle management

Usage:
    from src.ava.db_manager import get_db_manager

    # Get a cursor with automatic commit/rollback
    with get_db_manager().get_cursor() as cursor:
        cursor.execute("SELECT * FROM portfolio_balances")
        results = cursor.fetchall()

    # Or get a connection for transactions
    with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT ...")
        cursor.execute("UPDATE ...")
        conn.commit()  # Explicit commit
"""

import os
import logging
import time
import threading
from contextlib import contextmanager
from typing import Optional, Any, Dict
from functools import wraps

import psycopg2
from psycopg2 import pool, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database errors"""
    pass


class ConnectionPoolExhausted(DatabaseError):
    """Raised when connection pool is exhausted"""
    pass


class DatabaseConnectionManager:
    """
    Thread-safe singleton database connection manager.

    Provides connection pooling with automatic retry and error handling
    specifically designed for the AVA Telegram bot.
    """

    _instance: Optional['DatabaseConnectionManager'] = None
    _lock = threading.Lock()
    _pool: Optional[pool.ThreadedConnectionPool] = None

    # Configuration - Use environment variables with sensible defaults
    MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN_CONNECTIONS", "2"))
    MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX_CONNECTIONS", "10"))
    MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("DB_RETRY_DELAY", "1.0"))  # seconds
    CONNECTION_TIMEOUT = int(os.getenv("DB_CONNECTION_TIMEOUT", "10"))  # seconds

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize connection manager (only once)"""
        if self._pool is not None:
            return  # Already initialized

        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool"""
        try:
            with self._lock:
                if self._pool is None:
                    self._pool = pool.ThreadedConnectionPool(
                        minconn=self.MIN_CONNECTIONS,
                        maxconn=self.MAX_CONNECTIONS,
                        dsn=self.db_url,
                        connect_timeout=self.CONNECTION_TIMEOUT
                    )
                    logger.info(
                        f"Database connection pool initialized: "
                        f"{self.MIN_CONNECTIONS}-{self.MAX_CONNECTIONS} connections"
                    )
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    def _get_connection(self) -> psycopg2.extensions.connection:
        """
        Get a connection from the pool with retry logic.

        Returns:
            Database connection

        Raises:
            ConnectionPoolExhausted: If pool is exhausted after retries
            DatabaseError: If connection fails
        """
        if self._pool is None:
            self._initialize_pool()

        retries = 0
        last_error = None

        while retries < self.MAX_RETRIES:
            try:
                conn = self._pool.getconn()

                # Validate connection is alive
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                except (OperationalError, InterfaceError):
                    # Connection is dead, close and retry
                    self._pool.putconn(conn, close=True)
                    raise

                return conn

            except pool.PoolError as e:
                last_error = e
                wait_time = self.RETRY_DELAY * (2 ** retries)  # Exponential backoff
                logger.warning(f"Connection pool exhausted, retry {retries + 1}/{self.MAX_RETRIES} in {wait_time}s")
                time.sleep(wait_time)
                retries += 1

            except (OperationalError, InterfaceError) as e:
                last_error = e
                wait_time = self.RETRY_DELAY * (2 ** retries)
                logger.warning(f"Connection error, retry {retries + 1}/{self.MAX_RETRIES} in {wait_time}s: {e}")
                time.sleep(wait_time)
                retries += 1

        # All retries failed
        if isinstance(last_error, pool.PoolError):
            raise ConnectionPoolExhausted("Database connection pool is exhausted. Please try again later.")
        else:
            raise DatabaseError(f"Failed to connect to database: {str(last_error)}")

    def _return_connection(self, conn: psycopg2.extensions.connection, close: bool = False):
        """
        Return a connection to the pool.

        Args:
            conn: Connection to return
            close: If True, close the connection instead of returning to pool
        """
        if self._pool is None or conn is None:
            return

        try:
            if not conn.closed and not close:
                # Reset connection state before returning to pool
                try:
                    conn.rollback()
                except Exception as e:
                    logger.warning(f"Error rolling back connection: {e}")
                    close = True  # Close if rollback fails

            self._pool.putconn(conn, close=close)

        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for safe connection handling.

        Automatically handles rollback on errors and returns connection to pool.

        Usage:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
                conn.commit()

        Yields:
            Database connection
        """
        conn = None
        try:
            conn = self._get_connection()
            yield conn
        except DatabaseError:
            # Re-raise database errors as-is (already user-friendly)
            if conn and not conn.closed:
                conn.rollback()
            raise
        except Exception as e:
            # Wrap other exceptions
            if conn and not conn.closed:
                conn.rollback()
            logger.error(f"Unexpected error in database operation: {e}", exc_info=True)
            raise DatabaseError("An unexpected database error occurred")
        finally:
            if conn:
                self._return_connection(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """
        Context manager for connection + cursor.

        Automatically handles commit/rollback and cleanup.

        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)

        Usage:
            with db_manager.get_cursor(RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()

        Yields:
            Database cursor
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=cursor_factory)
            yield cursor
            conn.commit()
        except DatabaseError:
            # Re-raise database errors as-is
            if conn and not conn.closed:
                conn.rollback()
            raise
        except Exception as e:
            if conn and not conn.closed:
                conn.rollback()
            logger.error(f"Unexpected error in database operation: {e}", exc_info=True)
            raise DatabaseError("An unexpected database error occurred")
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                self._return_connection(conn)

    def execute_query(
        self,
        query: str,
        params: tuple = None,
        fetch_one: bool = False,
        cursor_factory=None
    ) -> Optional[Any]:
        """
        Execute a query and return results (convenience method).

        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: If True, return single row; if False, return all rows
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)

        Returns:
            Query results or None

        Raises:
            DatabaseError: If query fails
        """
        try:
            with self.get_cursor(cursor_factory=cursor_factory) as cursor:
                cursor.execute(query, params)

                # Check if this is a SELECT query
                if cursor.description is not None:
                    if fetch_one:
                        return cursor.fetchone()
                    else:
                        return cursor.fetchall()

                return None

        except psycopg2.Error as e:
            logger.error(f"Database query error: {e}")
            raise DatabaseError("Failed to execute database query")

    def close_all(self):
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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool stats
        """
        return {
            "initialized": self._pool is not None,
            "min_connections": self.MIN_CONNECTIONS,
            "max_connections": self.MAX_CONNECTIONS,
        }


# Global singleton instance
_db_manager_instance: Optional[DatabaseConnectionManager] = None
_init_lock = threading.Lock()


def get_db_manager() -> DatabaseConnectionManager:
    """
    Get the global database manager instance.

    Thread-safe singleton accessor.

    Returns:
        DatabaseConnectionManager instance
    """
    global _db_manager_instance

    if _db_manager_instance is None:
        with _init_lock:
            if _db_manager_instance is None:
                _db_manager_instance = DatabaseConnectionManager()

    return _db_manager_instance


def close_db_manager():
    """
    Close the global database manager.

    Call this on application shutdown.
    """
    global _db_manager_instance

    if _db_manager_instance is not None:
        with _init_lock:
            if _db_manager_instance is not None:
                _db_manager_instance.close_all()
                _db_manager_instance = None


def with_db_error_handling(func):
    """
    Decorator for database operations with error handling.

    Converts database exceptions to user-friendly messages.

    Usage:
        @with_db_error_handling
        def get_portfolio():
            with get_db_manager().get_cursor() as cursor:
                cursor.execute("SELECT * FROM portfolio")
                return cursor.fetchall()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionPoolExhausted as e:
            logger.error(f"Connection pool exhausted in {func.__name__}: {e}")
            raise DatabaseError("The system is currently busy. Please try again in a moment.")
        except DatabaseError:
            # Re-raise as-is (already user-friendly)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise DatabaseError("An unexpected error occurred while accessing the database.")

    return wrapper


# Example usage
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing AVA Database Manager...\n")

    try:
        # Get database manager
        db = get_db_manager()

        # Test 1: Simple query with context manager
        print("Test 1: Simple query")
        with db.get_cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            print(f"Result: {result}")

        # Test 2: Query with RealDictCursor
        print("\nTest 2: Query with RealDictCursor")
        with db.get_cursor(RealDictCursor) as cursor:
            cursor.execute("SELECT 1 as test, 2 as value")
            result = cursor.fetchone()
            print(f"Result: {result}")

        # Test 3: Convenience method
        print("\nTest 3: Convenience method")
        result = db.execute_query(
            "SELECT $1::int as number, $2::text as text",
            params=(42, "hello"),
            fetch_one=True,
            cursor_factory=RealDictCursor
        )
        print(f"Result: {result}")

        # Test 4: Error handling
        print("\nTest 4: Error handling")
        try:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM nonexistent_table")
        except DatabaseError as e:
            print(f"Caught expected error: {e}")

        # Test 5: Stats
        print("\nTest 5: Connection pool stats")
        stats = db.get_stats()
        print(f"Stats: {stats}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        close_db_manager()
        print("\nDatabase manager closed")
