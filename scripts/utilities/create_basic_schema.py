"""
Create basic database schema without TimescaleDB
"""
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv(override=True)

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    database='magnus',
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD')
)

cursor = conn.cursor()

print("Creating basic Magnus database schema...")
print("=" * 70)

# Enable UUID extension
print("\n[1/6] Enabling UUID extension...")
cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
conn.commit()
print("[OK] UUID extension enabled")

# Create users table
print("\n[2/6] Creating users table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        first_name VARCHAR(100),
        last_name VARCHAR(100),
        is_active BOOLEAN DEFAULT true,
        risk_tolerance VARCHAR(20) DEFAULT 'moderate',
        max_portfolio_risk DECIMAL(5,4) DEFAULT 0.02,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
""")
conn.commit()
print("[OK] Users table created")

# Create stocks table
print("\n[3/6] Creating stocks table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        symbol VARCHAR(10) NOT NULL UNIQUE,
        company_name VARCHAR(255),
        sector VARCHAR(100),
        industry VARCHAR(100),
        market_cap BIGINT,
        is_active BOOLEAN DEFAULT true,
        is_optionable BOOLEAN DEFAULT false,
        average_volume BIGINT,
        beta DECIMAL(6,3),
        dividend_yield DECIMAL(5,4),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
    CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks(sector);
""")
conn.commit()
print("[OK] Stocks table created")

# Create watchlists table
print("\n[4/6] Creating watchlists table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS watchlists (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        is_default BOOLEAN DEFAULT false,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_watchlists_user_id ON watchlists(user_id);
""")
conn.commit()
print("[OK] Watchlists table created")

# Create stock_prices table
print("\n[5/6] Creating stock_prices table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        open DECIMAL(12,4),
        high DECIMAL(12,4),
        low DECIMAL(12,4),
        close DECIMAL(12,4),
        volume BIGINT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_stock_prices_stock_timestamp
        ON stock_prices(stock_id, timestamp DESC);
""")
conn.commit()
print("[OK] Stock prices table created")

# Create options_data table
print("\n[6/6] Creating options_data table...")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS options_data (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
        expiration_date DATE NOT NULL,
        strike_price DECIMAL(12,4) NOT NULL,
        option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
        bid DECIMAL(12,4),
        ask DECIMAL(12,4),
        last_price DECIMAL(12,4),
        volume INTEGER,
        open_interest INTEGER,
        implied_volatility DECIMAL(8,6),
        delta DECIMAL(8,6),
        gamma DECIMAL(8,6),
        theta DECIMAL(8,6),
        vega DECIMAL(8,6),
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_options_data_stock_exp
        ON options_data(stock_id, expiration_date);
    CREATE INDEX IF NOT EXISTS idx_options_data_type_strike
        ON options_data(option_type, strike_price);
""")
conn.commit()
print("[OK] Options data table created")

# Check all tables
print("\n" + "=" * 70)
print("Verifying created tables...")
cursor.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
""")
tables = cursor.fetchall()

print(f"\n[OK] Database has {len(tables)} tables:")
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
    count = cursor.fetchone()[0]
    print(f"  - {table[0]}: {count} rows")

cursor.close()
conn.close()

print("\n" + "=" * 70)
print("Basic schema created successfully!")
print("=" * 70)
print("\nDatabase is ready for use!")
print("Run 'python check_postgres.py' to verify.")
