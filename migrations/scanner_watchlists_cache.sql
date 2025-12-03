-- Scanner Watchlists Cache Table
-- Stores all watchlists in database for fast API responses
-- Synced periodically by scripts/sync_watchlists.py

CREATE TABLE IF NOT EXISTS scanner_watchlists (
    id SERIAL PRIMARY KEY,
    watchlist_id VARCHAR(100) NOT NULL UNIQUE,  -- e.g., 'predefined_popular', 'tv_123', 'sector_technology'
    source VARCHAR(50) NOT NULL,                 -- predefined, database, tradingview, robinhood
    name VARCHAR(255) NOT NULL,
    symbols TEXT[] NOT NULL,                     -- Array of symbols
    symbol_count INTEGER GENERATED ALWAYS AS (array_length(symbols, 1)) STORED,
    category VARCHAR(100),                       -- optional grouping: 'popular', 'sector', 'portfolio', etc.
    sort_order INTEGER DEFAULT 1000,             -- for custom ordering
    is_active BOOLEAN DEFAULT true,
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_scanner_watchlists_source ON scanner_watchlists(source);
CREATE INDEX IF NOT EXISTS idx_scanner_watchlists_active ON scanner_watchlists(is_active);
CREATE INDEX IF NOT EXISTS idx_scanner_watchlists_category ON scanner_watchlists(category);
CREATE INDEX IF NOT EXISTS idx_scanner_watchlists_sort ON scanner_watchlists(sort_order, name);

-- Sync history for tracking
CREATE TABLE IF NOT EXISTS scanner_watchlists_sync_log (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(50) NOT NULL,              -- 'full', 'incremental', 'source_specific'
    source VARCHAR(50),                          -- which source was synced (null for full)
    watchlists_synced INTEGER DEFAULT 0,
    total_symbols INTEGER DEFAULT 0,
    duration_seconds NUMERIC(10,2),
    status VARCHAR(20) DEFAULT 'success',        -- success, partial, failed
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_scanner_watchlists_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-updating timestamp
DROP TRIGGER IF EXISTS scanner_watchlists_updated_at ON scanner_watchlists;
CREATE TRIGGER scanner_watchlists_updated_at
    BEFORE UPDATE ON scanner_watchlists
    FOR EACH ROW
    EXECUTE FUNCTION update_scanner_watchlists_timestamp();

-- Comments
COMMENT ON TABLE scanner_watchlists IS 'Cached watchlists for Premium Scanner - synced periodically for fast API responses';
COMMENT ON COLUMN scanner_watchlists.watchlist_id IS 'Unique identifier: source_name format (e.g., predefined_popular, tv_123456)';
COMMENT ON COLUMN scanner_watchlists.source IS 'Data source: predefined, database, tradingview, robinhood';
COMMENT ON COLUMN scanner_watchlists.symbols IS 'Array of stock symbols in this watchlist';
COMMENT ON COLUMN scanner_watchlists.sort_order IS 'Lower numbers appear first in dropdown';
