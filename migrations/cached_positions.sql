-- =============================================================================
-- Cached Positions Table for Non-Blocking UI
-- =============================================================================
-- Purpose: Store Robinhood positions locally so the UI never blocks on API calls
-- Sync: Background service syncs every 30 minutes
-- Created: 2025-12-05

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
    strategy VARCHAR(20),  -- CSP, CC, Long Put, Long Call
    position_type VARCHAR(10),  -- long, short
    option_type VARCHAR(10),  -- put, call
    strike DECIMAL(18, 2) NOT NULL,
    expiration DATE NOT NULL,
    dte INTEGER,
    quantity DECIMAL(18, 8) NOT NULL,
    avg_price DECIMAL(18, 2),  -- Per contract
    current_price DECIMAL(18, 2),  -- Per contract
    total_premium DECIMAL(18, 2),
    current_value DECIMAL(18, 2),
    pl DECIMAL(18, 2),
    pl_pct DECIMAL(10, 4),
    breakeven DECIMAL(18, 2),
    -- Greeks
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
    sync_type VARCHAR(20) NOT NULL,  -- full, incremental, manual
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL,  -- running, success, failed
    stocks_synced INTEGER DEFAULT 0,
    options_synced INTEGER DEFAULT 0,
    error_message TEXT,
    duration_ms INTEGER
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_cached_stock_synced ON cached_stock_positions(synced_at);
CREATE INDEX IF NOT EXISTS idx_cached_option_synced ON cached_option_positions(synced_at);
CREATE INDEX IF NOT EXISTS idx_cached_option_symbol ON cached_option_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_cached_option_expiration ON cached_option_positions(expiration);
CREATE INDEX IF NOT EXISTS idx_sync_log_started ON positions_sync_log(started_at DESC);

-- Function to get cache freshness
CREATE OR REPLACE FUNCTION get_positions_cache_age()
RETURNS TABLE (
    stocks_age_seconds INTEGER,
    options_age_seconds INTEGER,
    summary_age_seconds INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        EXTRACT(EPOCH FROM (NOW() - (SELECT MAX(synced_at) FROM cached_stock_positions)))::INTEGER AS stocks_age_seconds,
        EXTRACT(EPOCH FROM (NOW() - (SELECT MAX(synced_at) FROM cached_option_positions)))::INTEGER AS options_age_seconds,
        EXTRACT(EPOCH FROM (NOW() - (SELECT synced_at FROM cached_portfolio_summary WHERE id = 1)))::INTEGER AS summary_age_seconds;
END;
$$ LANGUAGE plpgsql;

-- View for easy access to full cached positions
CREATE OR REPLACE VIEW v_cached_positions AS
SELECT
    'stock' AS position_class,
    symbol,
    quantity,
    avg_buy_price,
    current_price,
    cost_basis,
    current_value,
    pl,
    pl_pct,
    NULL::VARCHAR AS strategy,
    NULL::VARCHAR AS option_type,
    NULL::DECIMAL AS strike,
    NULL::DATE AS expiration,
    NULL::INTEGER AS dte,
    NULL::DECIMAL AS delta,
    NULL::DECIMAL AS theta,
    NULL::DECIMAL AS gamma,
    NULL::DECIMAL AS vega,
    NULL::DECIMAL AS iv,
    synced_at
FROM cached_stock_positions
UNION ALL
SELECT
    'option' AS position_class,
    symbol,
    quantity,
    avg_price AS avg_buy_price,
    current_price,
    total_premium AS cost_basis,
    current_value,
    pl,
    pl_pct,
    strategy,
    option_type,
    strike,
    expiration,
    dte,
    delta,
    theta,
    gamma,
    vega,
    iv,
    synced_at
FROM cached_option_positions;

COMMENT ON TABLE cached_stock_positions IS 'Cached stock positions from Robinhood, synced every 30 minutes';
COMMENT ON TABLE cached_option_positions IS 'Cached option positions from Robinhood, synced every 30 minutes';
COMMENT ON TABLE cached_portfolio_summary IS 'Cached portfolio summary (equity, buying power, etc.)';
COMMENT ON TABLE positions_sync_log IS 'Log of position sync operations for monitoring';
