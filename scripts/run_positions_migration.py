"""
Run the cached positions migration.
Creates tables for caching Robinhood positions in the database.
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.infrastructure.database import get_database


MIGRATION_SQL = """
-- Create cached_stock_positions table
CREATE TABLE IF NOT EXISTS cached_stock_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18, 8) NOT NULL,
    avg_buy_price DECIMAL(18, 8) NOT NULL,
    current_price DECIMAL(18, 8),
    cost_basis DECIMAL(18, 2),
    current_value DECIMAL(18, 2),
    pl DECIMAL(18, 2),
    pl_pct DECIMAL(10, 4),
    synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uk_cached_stock_symbol UNIQUE (symbol)
);

-- Create cached_option_positions table
CREATE TABLE IF NOT EXISTS cached_option_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    option_id VARCHAR(100),
    strategy VARCHAR(20),
    position_type VARCHAR(10),
    option_type VARCHAR(10),
    strike DECIMAL(18, 2) NOT NULL,
    expiration DATE NOT NULL,
    dte INTEGER,
    quantity DECIMAL(18, 8) NOT NULL,
    avg_price DECIMAL(18, 2),
    current_price DECIMAL(18, 2),
    total_premium DECIMAL(18, 2),
    current_value DECIMAL(18, 2),
    pl DECIMAL(18, 2),
    pl_pct DECIMAL(10, 4),
    breakeven DECIMAL(18, 2),
    delta DECIMAL(10, 4),
    theta DECIMAL(10, 4),
    gamma DECIMAL(10, 6),
    vega DECIMAL(10, 4),
    iv DECIMAL(10, 4),
    synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uk_cached_option_unique UNIQUE (symbol, strike, expiration, option_type)
);

-- Create portfolio summary cache
CREATE TABLE IF NOT EXISTS cached_portfolio_summary (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total_equity DECIMAL(18, 2),
    core_equity DECIMAL(18, 2),
    buying_power DECIMAL(18, 2),
    portfolio_cash DECIMAL(18, 2),
    uncleared_deposits DECIMAL(18, 2),
    unsettled_funds DECIMAL(18, 2),
    options_collateral DECIMAL(18, 2),
    total_stock_positions INTEGER,
    total_option_positions INTEGER,
    synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

-- Sync log for tracking sync history
CREATE TABLE IF NOT EXISTS positions_sync_log (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(20) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,
    stocks_synced INTEGER DEFAULT 0,
    options_synced INTEGER DEFAULT 0,
    error_message TEXT,
    duration_ms INTEGER
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_cached_stock_synced ON cached_stock_positions(synced_at);
CREATE INDEX IF NOT EXISTS idx_cached_option_synced ON cached_option_positions(synced_at);
CREATE INDEX IF NOT EXISTS idx_cached_option_symbol ON cached_option_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_cached_option_expiration ON cached_option_positions(expiration);
CREATE INDEX IF NOT EXISTS idx_sync_log_started ON positions_sync_log(started_at DESC);
"""


async def run_migration():
    print("Connecting to database...")
    db = await get_database()

    print("Running cached positions migration...")

    # Split by statement (simple split on semicolons)
    statements = [s.strip() for s in MIGRATION_SQL.split(';') if s.strip()]

    for i, statement in enumerate(statements, 1):
        if not statement:
            continue
        try:
            await db.execute(statement)
            # Extract table/index name for logging
            if 'CREATE TABLE' in statement:
                name = statement.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
                print(f"  [{i}/{len(statements)}] Created table: {name}")
            elif 'CREATE INDEX' in statement:
                name = statement.split('CREATE INDEX IF NOT EXISTS')[1].split(' ON')[0].strip()
                print(f"  [{i}/{len(statements)}] Created index: {name}")
            else:
                print(f"  [{i}/{len(statements)}] Executed statement")
        except Exception as e:
            if 'already exists' in str(e).lower():
                print(f"  [{i}/{len(statements)}] Already exists (skipped)")
            else:
                print(f"  [{i}/{len(statements)}] Error: {e}")

    print("\nMigration complete!")

    # Verify tables exist
    tables = await db.fetch("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name LIKE 'cached_%' OR table_name = 'positions_sync_log'
        ORDER BY table_name
    """)
    print(f"\nVerified tables: {[r['table_name'] for r in tables]}")


if __name__ == "__main__":
    asyncio.run(run_migration())
