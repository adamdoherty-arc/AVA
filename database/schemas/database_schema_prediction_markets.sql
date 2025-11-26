-- Prediction Markets Database Schema
-- Stores event contract data from Kalshi API with AI scoring

CREATE TABLE IF NOT EXISTS prediction_markets (
    id SERIAL PRIMARY KEY,

    -- Market Identification
    ticker VARCHAR(50) UNIQUE NOT NULL,          -- Kalshi ticker (e.g., PRES-2024-DEM)
    title VARCHAR(500) NOT NULL,                 -- Human-readable market title
    category VARCHAR(50),                         -- Politics, Sports, Economics, etc.
    subcategory VARCHAR(50),                      -- Presidential, NFL, Fed Rates, etc.

    -- Pricing Data
    yes_price DECIMAL(10, 4),                    -- Current Yes price (0-1 range, e.g., 0.52 = 52%)
    no_price DECIMAL(10, 4),                     -- Current No price (0-1 range)
    yes_bid DECIMAL(10, 4),                      -- Best bid for Yes
    yes_ask DECIMAL(10, 4),                      -- Best ask for Yes
    no_bid DECIMAL(10, 4),                       -- Best bid for No
    no_ask DECIMAL(10, 4),                       -- Best ask for No

    -- Volume & Liquidity Metrics
    volume_24h INTEGER DEFAULT 0,                -- Contracts traded in last 24 hours
    open_interest INTEGER DEFAULT 0,             -- Total outstanding contracts
    bid_ask_spread DECIMAL(10, 4),              -- Spread in decimal (e.g., 0.01 = 1%)

    -- Market Timing
    open_date TIMESTAMP,                         -- When market opened for trading
    close_date TIMESTAMP,                        -- When trading closes
    resolution_date TIMESTAMP,                   -- When outcome is determined
    days_to_close INTEGER,                       -- Calculated: days until trading closes

    -- AI Scoring & Recommendations
    ai_score DECIMAL(5, 2),                     -- Overall rating 0-100
    ai_reasoning TEXT,                           -- Explanation of the score
    recommended_position VARCHAR(10),            -- 'Yes', 'No', or 'Skip'
    risk_level VARCHAR(20),                      -- 'Low', 'Medium', or 'High'
    expected_value DECIMAL(10, 4),              -- EV calculation

    -- Market Details
    description TEXT,                            -- Full market description
    rules TEXT,                                  -- Settlement rules
    market_status VARCHAR(20) DEFAULT 'active',  -- 'active', 'closed', 'settled'
    result VARCHAR(10),                          -- 'Yes', 'No', or NULL if not settled

    -- Metadata
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),

    -- Constraints
    CONSTRAINT price_range_yes CHECK (yes_price BETWEEN 0 AND 1),
    CONSTRAINT price_range_no CHECK (no_price BETWEEN 0 AND 1),
    CONSTRAINT spread_positive CHECK (bid_ask_spread >= 0),
    CONSTRAINT ai_score_range CHECK (ai_score BETWEEN 0 AND 100)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_prediction_ai_score ON prediction_markets(ai_score DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_category ON prediction_markets(category);
CREATE INDEX IF NOT EXISTS idx_prediction_close_date ON prediction_markets(close_date);
CREATE INDEX IF NOT EXISTS idx_prediction_status ON prediction_markets(market_status);
CREATE INDEX IF NOT EXISTS idx_prediction_updated ON prediction_markets(last_updated DESC);

-- View for top opportunities
CREATE OR REPLACE VIEW top_prediction_opportunities AS
SELECT
    ticker,
    title,
    category,
    yes_price,
    no_price,
    volume_24h,
    ai_score,
    recommended_position,
    risk_level,
    days_to_close,
    expected_value
FROM prediction_markets
WHERE market_status = 'active'
    AND ai_score >= 70
    AND days_to_close > 0
ORDER BY ai_score DESC
LIMIT 50;

-- Table for tracking AI score accuracy over time
CREATE TABLE IF NOT EXISTS prediction_market_accuracy (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(50) NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    ai_score DECIMAL(5, 2),
    recommended_position VARCHAR(10),
    predicted_probability DECIMAL(10, 4),
    actual_result VARCHAR(10),                   -- 'Yes' or 'No'
    correct_prediction BOOLEAN,
    settlement_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_accuracy_ticker ON prediction_market_accuracy(ticker);
CREATE INDEX IF NOT EXISTS idx_accuracy_date ON prediction_market_accuracy(prediction_date DESC);

COMMENT ON TABLE prediction_markets IS 'Stores Kalshi prediction market data with AI-powered opportunity ratings';
COMMENT ON COLUMN prediction_markets.ai_score IS 'AI-generated score 0-100 based on mispricing, liquidity, risk-reward, and catalysts';
COMMENT ON COLUMN prediction_markets.expected_value IS 'Probability-weighted expected return percentage';
