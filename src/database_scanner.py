"""Database Scanner - Pull and analyze stocks from PostgreSQL database"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
# PERFORMANCE: Using centralized rate-limited wrapper
from src.yfinance_wrapper import get_ticker, get_ticker_info, get_ticker_history
from src.yfinance_utils import (
    safe_get_ticker,
    safe_get_history,
    safe_get_current_price,
    safe_get_info,
    safe_get_options_expirations,
    safe_get_option_chain,
    is_symbol_delisted
)

# Load environment variables
load_dotenv(override=True)

class DatabaseScanner:
    """Scan PostgreSQL database for stocks and analyze option premiums"""

    def __init__(self) -> None:
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),  # Fixed: was 'wheel_strategy', should be 'magnus'
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!')
        }
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def create_tables(self) -> None:
        """Create necessary tables if they don't exist"""
        try:
            # Create stocks table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) UNIQUE NOT NULL,
                    name VARCHAR(255),
                    sector VARCHAR(100),
                    industry VARCHAR(100),
                    market_cap BIGINT,
                    current_price DECIMAL(10,2),
                    avg_volume BIGINT,
                    last_updated TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create options_data table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS options_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    expiration_date DATE,
                    strike_price DECIMAL(10,2),
                    option_type VARCHAR(4),
                    bid DECIMAL(10,2),
                    ask DECIMAL(10,2),
                    volume INTEGER,
                    open_interest INTEGER,
                    implied_volatility DECIMAL(5,2),
                    last_updated TIMESTAMP DEFAULT NOW(),
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            """)

            # Create watchlist table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    added_date TIMESTAMP DEFAULT NOW(),
                    notes TEXT,
                    target_price DECIMAL(10,2),
                    stop_loss DECIMAL(10,2),
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            """)

            # Create trade_history table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    trade_type VARCHAR(20),
                    quantity INTEGER,
                    price DECIMAL(10,2),
                    premium DECIMAL(10,2),
                    trade_date TIMESTAMP DEFAULT NOW(),
                    expiration_date DATE,
                    strike_price DECIMAL(10,2),
                    status VARCHAR(20),
                    profit_loss DECIMAL(10,2)
                )
            """)

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error creating tables: {e}")
            self.conn.rollback()
            return False

    def add_stock(self, symbol: str, fetch_data: bool = True) -> bool:
        """Add a stock to the database"""
        try:
            symbol = symbol.upper()

            if fetch_data:
                # Check if symbol is delisted
                if is_symbol_delisted(symbol):
                    print(f"Symbol {symbol} is known to be delisted - skipping")
                    return False

                # Fetch stock data using safe wrapper
                info = safe_get_info(symbol, suppress_warnings=False)

                if not info:
                    print(f"Could not fetch data for {symbol} - may be delisted")
                    return False

                name = info.get('longName', symbol)
                sector = info.get('sector', 'Unknown')
                industry = info.get('industry', 'Unknown')
                market_cap = info.get('marketCap', 0)
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                avg_volume = info.get('averageVolume', 0)

            else:
                name = symbol
                sector = 'Unknown'
                industry = 'Unknown'
                market_cap = 0
                current_price = 0
                avg_volume = 0

            # Insert or update stock
            # Note: Database uses 'ticker' and 'price' column names
            self.cursor.execute("""
                INSERT INTO stocks (ticker, name, sector, industry, market_cap, price, avg_volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    market_cap = EXCLUDED.market_cap,
                    price = EXCLUDED.price,
                    avg_volume = EXCLUDED.avg_volume,
                    last_updated = NOW()
            """, (symbol.upper(), name, sector, industry, market_cap, current_price, avg_volume))

            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error adding stock {symbol}: {e}")
            self.conn.rollback()
            return False

    def bulk_add_stocks(self, symbols: List[str]) -> int:
        """Add multiple stocks to the database"""
        added = 0
        for symbol in symbols:
            if self.add_stock(symbol):
                added += 1
                print(f"Added {symbol} to database")
        return added

    def get_all_stocks(self) -> List[Dict]:
        """Get all stocks from database"""
        try:
            # Map database columns to expected schema
            # Database uses 'symbol' column
            # Database uses 'price' but code expects 'current_price'
            self.cursor.execute("""
                SELECT
                    id,
                    symbol,
                    company_name as name,
                    sector,
                    industry,
                    market_cap,
                    COALESCE(market_cap, 0) as current_price,
                    COALESCE(average_volume, 0) as avg_volume,
                    updated_at as last_updated
                FROM stocks
                ORDER BY symbol
            """)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error fetching stocks: {e}")
            return []

    def get_stocks_by_criteria(self,
                               max_price: Optional[float] = None,
                               min_volume: Optional[int] = None,
                               sector: Optional[str] = None) -> List[Dict]:
        """Get stocks matching specific criteria"""
        try:
            # Map database columns to expected schema
            query = """
                SELECT
                    id,
                    symbol,
                    company_name as name,
                    sector,
                    industry,
                    market_cap,
                    COALESCE(market_cap, 0) as current_price,
                    COALESCE(average_volume, 0) as avg_volume,
                    updated_at as last_updated
                FROM stocks
                WHERE 1=1
            """
            params = []

            if max_price:
                query += " AND price <= %s"
                params.append(max_price)

            if min_volume:
                query += " AND avg_volume >= %s"
                params.append(min_volume)

            if sector:
                query += " AND sector = %s"
                params.append(sector)

            query += " ORDER BY ticker"

            self.cursor.execute(query, params)
            return self.cursor.fetchall()

        except Exception as e:
            print(f"Error fetching filtered stocks: {e}")
            return []

    def update_stock_prices(self) -> int:
        """Update all stock prices in database"""
        updated = 0
        stocks = self.get_all_stocks()

        for stock in stocks:
            try:
                symbol = stock['symbol']

                # Skip delisted symbols
                if is_symbol_delisted(symbol):
                    print(f"Skipping delisted symbol {symbol}")
                    continue

                # Use safe wrapper to get current price
                current_price = safe_get_current_price(symbol, suppress_warnings=True)

                if current_price and current_price > 0:
                    self.cursor.execute("""
                        UPDATE stocks
                        SET price = %s, last_updated = NOW()
                        WHERE ticker = %s
                    """, (current_price, symbol))
                    updated += 1
                else:
                    print(f"Could not get valid price for {symbol} - may be delisted")

            except Exception as e:
                print(f"Error updating {stock['symbol']}: {e}")

        self.conn.commit()
        return updated

    def save_option_data(self, symbol: str, expiration: str, option_data: Dict) -> bool:
        """Save option data to database"""
        try:
            self.cursor.execute("""
                INSERT INTO options_data
                (symbol, expiration_date, strike_price, option_type, bid, ask, volume, open_interest, implied_volatility)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                expiration,
                option_data.get('strike', 0),
                option_data.get('type', 'PUT'),
                option_data.get('bid', 0),
                option_data.get('ask', 0),
                option_data.get('volume', 0),
                option_data.get('openInterest', 0),
                option_data.get('impliedVolatility', 0) * 100 if option_data.get('impliedVolatility') else 0
            ))
            self.conn.commit()
            return True

        except Exception as e:
            print(f"Error saving option data: {e}")
            self.conn.rollback()
            return False

    def get_stocks_with_best_premiums(self, max_price: float = 50, days_to_expiry: int = 30) -> List[Dict]:
        """Get stocks with best option premiums from database"""
        stocks = self.get_stocks_by_criteria(max_price=max_price)
        results = []

        for stock in stocks:
            try:
                symbol = stock['symbol']
                current_price = stock['current_price']

                if current_price <= 0:
                    continue

                # Skip delisted symbols
                if is_symbol_delisted(symbol):
                    continue

                # Use safe wrappers for yfinance calls
                expirations = safe_get_options_expirations(symbol, suppress_warnings=True)

                if not expirations:
                    continue

                # Find expiration closest to target days
                target_date = datetime.now() + timedelta(days=days_to_expiry)
                best_expiry = None
                min_diff = float('inf')

                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                    diff = abs((exp_date - target_date).days)
                    if diff < min_diff:
                        min_diff = diff
                        best_expiry = exp

                if best_expiry:
                    opt_chain = safe_get_option_chain(symbol, best_expiry, suppress_warnings=True)
                    if not opt_chain:
                        continue
                    puts = opt_chain.puts

                    # Find 5% OTM put
                    target_strike = current_price * 0.95
                    otm_puts = puts[puts['strike'] <= target_strike]

                    if not otm_puts.empty:
                        best_put = otm_puts.iloc[-1]

                        strike = best_put['strike']
                        bid = best_put['bid']
                        ask = best_put['ask']
                        premium = (bid + ask) / 2 if bid > 0 and ask > 0 else bid

                        if premium > 0:
                            return_pct = (premium * 100 / (strike * 100)) * 100

                            results.append({
                                'symbol': symbol,
                                'name': stock['name'],
                                'sector': stock['sector'],
                                'current_price': current_price,
                                'strike': strike,
                                'premium': premium,
                                'return_pct': return_pct,
                                'expiration': best_expiry,
                                'days_to_expiry': min_diff
                            })

            except Exception as e:
                print(f"Error processing {symbol}: {e}")

        # Sort by return percentage
        results.sort(key=lambda x: x['return_pct'], reverse=True)
        return results[:20]  # Return top 20

    def initialize_default_stocks(self) -> None:
        """Initialize database - no default stocks"""
        # No default stocks - user must add their own
        return 0