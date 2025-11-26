import os
import logging
from contextlib import contextmanager
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._pool = pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    host=os.getenv("DB_HOST", "localhost"),
                    port=int(os.getenv("DB_PORT", 5432)),
                    database=os.getenv("DB_NAME", "trading"),
                    user=os.getenv("DB_USER", "postgres"),
                    password=os.getenv("DB_PASSWORD")
                )
                logger.info("Database connection pool initialized.")
            except Exception as e:
                logger.error(f"Error initializing database connection pool: {e}")
                cls._pool = None
        return cls._instance

    @contextmanager
    def get_connection(self):
        if self._pool is None:
            raise Exception("Database connection pool is not initialized.")
        
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction error: {e}")
            raise
        finally:
            self._pool.putconn(conn)

    def close_all(self):
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed.")

db_pool = DatabaseConnectionPool()

def get_db_connection():
    """Dependency for FastAPI to get a database connection."""
    with db_pool.get_connection() as conn:
        yield conn
