-- ============================================================================
-- Table: stock_premiums
-- Store option premium data for wheel strategy analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_premiums (
    id SERIAL PRIMARY KEY,

    -- Option identification
    symbol VARCHAR(20) NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    expiration DATE NOT NULL,
    option_type VARCHAR(4) DEFAULT 'PUT',  -- PUT or CALL

    -- Stock data at scan time
    stock_price DECIMAL(10,2) NOT NULL,

    -- Option pricing
    bid DECIMAL(10,2),
    ask DECIMAL(10,2),
    premium DECIMAL(10,2) NOT NULL,  -- Premium for 1 contract (100 shares)
    premium_pct DECIMAL(6,3) NOT NULL,  -- Premium as % of strike

    -- Greeks & metrics
    iv DECIMAL(6,2),  -- Implied Volatility %
    delta DECIMAL(6,4),
    theta DECIMAL(8,6),

    -- DTE & returns
    dte INTEGER NOT NULL,  -- Days to expiration
    monthly_return DECIMAL(8,3),
    annual_return DECIMAL(8,3),

    -- Liquidity metrics
    volume INTEGER DEFAULT 0,
    open_interest INTEGER DEFAULT 0,
    liquidity_score INTEGER,

    -- Spread quality
    bid_ask_spread DECIMAL(6,3),
    spread_pct DECIMAL(6,2),
    spread_quality VARCHAR(10),  -- tight, moderate, wide

    -- Analysis fields
    otm_pct DECIMAL(6,3),  -- OTM percentage (negative = OTM, positive = ITM)
    collateral DECIMAL(12,2),  -- Required collateral

    -- Scan metadata
    scan_id VARCHAR(50),  -- Reference to premium_scan_history
    watchlist_source VARCHAR(100),  -- Which watchlist this came from

    -- Timestamps
    scanned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint: one row per option contract per scan
    CONSTRAINT uq_stock_premium_option UNIQUE (symbol, strike, expiration, option_type, scanned_at)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_stock_premiums_symbol ON stock_premiums(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_premiums_expiration ON stock_premiums(expiration);
CREATE INDEX IF NOT EXISTS idx_stock_premiums_dte ON stock_premiums(dte);
CREATE INDEX IF NOT EXISTS idx_stock_premiums_premium_pct ON stock_premiums(premium_pct DESC);
CREATE INDEX IF NOT EXISTS idx_stock_premiums_annual_return ON stock_premiums(annual_return DESC);
CREATE INDEX IF NOT EXISTS idx_stock_premiums_liquidity ON stock_premiums(liquidity_score DESC);
CREATE INDEX IF NOT EXISTS idx_stock_premiums_scanned ON stock_premiums(scanned_at DESC);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_stock_premiums_high_return ON stock_premiums(symbol, annual_return DESC) WHERE annual_return > 20;
CREATE INDEX IF NOT EXISTS idx_stock_premiums_quality ON stock_premiums(spread_quality, liquidity_score DESC);
