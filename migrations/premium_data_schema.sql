-- Premium Data Storage Schema
-- Stores scanned option premiums for quick access and periodic updates

-- Drop existing tables if needed
DROP TABLE IF EXISTS premium_scan_history CASCADE;
DROP TABLE IF EXISTS premium_opportunities CASCADE;

-- Table to store scan history
CREATE TABLE IF NOT EXISTS premium_scan_history (
    id SERIAL PRIMARY KEY,
    scan_id VARCHAR(50) UNIQUE NOT NULL,
    symbols TEXT[],
    symbol_count INTEGER,
    dte INTEGER,
    max_price DECIMAL(10,2),
    min_premium_pct DECIMAL(5,2),
    result_count INTEGER,
    results JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table to store individual premium opportunities (latest scan per symbol/strike/expiry)
CREATE TABLE IF NOT EXISTS premium_opportunities (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    option_type VARCHAR(4) NOT NULL,  -- PUT or CALL
    strike DECIMAL(10,2) NOT NULL,
    expiration DATE NOT NULL,
    dte INTEGER NOT NULL,

    -- Stock data at time of scan
    stock_price DECIMAL(10,2),

    -- Option data
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    mid DECIMAL(10,4),
    premium DECIMAL(10,4),
    premium_pct DECIMAL(6,3),          -- Premium as % of stock price
    annualized_return DECIMAL(8,3),    -- Annualized return %
    monthly_return DECIMAL(8,3),       -- Monthly return %

    -- Greeks
    delta DECIMAL(8,5),
    gamma DECIMAL(8,5),
    theta DECIMAL(10,5),
    vega DECIMAL(10,5),
    rho DECIMAL(10,5),

    -- Volatility
    implied_volatility DECIMAL(8,4),

    -- Volume/Interest
    volume INTEGER,
    open_interest INTEGER,

    -- Risk metrics
    break_even DECIMAL(10,2),
    max_profit DECIMAL(10,2),
    max_loss DECIMAL(10,2),
    pop DECIMAL(6,3),                  -- Probability of profit

    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scan_id VARCHAR(50),

    -- Unique constraint on symbol/strike/expiration/type
    CONSTRAINT unique_option UNIQUE (symbol, option_type, strike, expiration)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_symbol ON premium_opportunities(symbol);
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_expiration ON premium_opportunities(expiration);
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_dte ON premium_opportunities(dte);
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_premium_pct ON premium_opportunities(premium_pct DESC);
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_annualized ON premium_opportunities(annualized_return DESC);
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_last_updated ON premium_opportunities(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_premium_opportunities_type ON premium_opportunities(option_type);

-- Index for scan history
CREATE INDEX IF NOT EXISTS idx_scan_history_created ON premium_scan_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scan_history_scan_id ON premium_scan_history(scan_id);

-- Add comment
COMMENT ON TABLE premium_opportunities IS 'Stores latest premium opportunities per option contract, updated periodically';
COMMENT ON TABLE premium_scan_history IS 'Stores history of premium scans with full results';
