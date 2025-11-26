-- Earnings Sync Service Database Schema
-- Supports historical earnings data, upcoming earnings calendar, and beat/miss tracking

-- ============================================================================
-- EARNINGS HISTORY TABLE
-- Stores all historical earnings reports from Robinhood API
-- ============================================================================
CREATE TABLE IF NOT EXISTS earnings_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,

    -- Timing Information
    report_date DATE NOT NULL,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    earnings_time VARCHAR(10) CHECK (earnings_time IN ('bmo', 'amc', 'unspecified')),

    -- EPS Data
    eps_actual DECIMAL(10,4),
    eps_estimate DECIMAL(10,4),
    eps_surprise DECIMAL(10,4),  -- Actual - Estimate
    eps_surprise_percent DECIMAL(8,2),  -- ((Actual - Estimate) / |Estimate|) * 100

    -- Revenue Data (if available)
    revenue_actual BIGINT,
    revenue_estimate BIGINT,
    revenue_surprise BIGINT,
    revenue_surprise_percent DECIMAL(8,2),

    -- Beat/Miss Classification
    beat_miss VARCHAR(10) CHECK (beat_miss IN ('beat', 'miss', 'meet', 'unknown')),

    -- Conference Call Information
    call_datetime TIMESTAMP WITH TIME ZONE,
    call_broadcast_url TEXT,

    -- Metadata
    data_source VARCHAR(50) DEFAULT 'robinhood',
    raw_data JSONB,  -- Store complete Robinhood API response
    synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(symbol, report_date, fiscal_quarter, fiscal_year)
);

-- ============================================================================
-- EARNINGS EVENTS TABLE (Upcoming Earnings Calendar)
-- Stores upcoming earnings announcements and tracks pre/post market data
-- ============================================================================
CREATE TABLE IF NOT EXISTS earnings_events (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,

    -- Event Timing
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(10) CHECK (earnings_time IN ('bmo', 'amc', 'unspecified')),
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,

    -- Estimates
    eps_estimate DECIMAL(10,4),
    revenue_estimate BIGINT,
    whisper_number DECIMAL(10,4),  -- Unofficial analyst whispers

    -- Actual Results (filled after announcement)
    eps_actual DECIMAL(10,4),
    revenue_actual BIGINT,
    surprise_percent DECIMAL(8,2),

    -- Market Data (Pre-Earnings)
    pre_earnings_price DECIMAL(10,2),
    pre_earnings_iv DECIMAL(6,4),  -- Implied Volatility before earnings
    pre_earnings_volume BIGINT,

    -- Market Data (Post-Earnings)
    post_earnings_price DECIMAL(10,2),
    post_earnings_iv DECIMAL(6,4),
    post_earnings_volume BIGINT,
    price_move_percent DECIMAL(8,2),
    volume_ratio DECIMAL(8,2),  -- Post volume / Pre volume

    -- Options Activity
    options_volume INTEGER,
    put_call_ratio DECIMAL(6,4),
    unusual_options_activity BOOLEAN DEFAULT FALSE,

    -- Conference Call
    call_datetime TIMESTAMP WITH TIME ZONE,
    call_broadcast_url TEXT,

    -- Status Tracking
    is_confirmed BOOLEAN DEFAULT FALSE,
    has_occurred BOOLEAN DEFAULT FALSE,
    data_source VARCHAR(50) DEFAULT 'robinhood',
    raw_data JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(symbol, earnings_date, fiscal_quarter, fiscal_year)
);

-- ============================================================================
-- EARNINGS SYNC STATUS TABLE
-- Tracks sync operations for monitoring and debugging
-- ============================================================================
CREATE TABLE IF NOT EXISTS earnings_sync_status (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,

    -- Sync Information
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_sync_status VARCHAR(20) CHECK (last_sync_status IN ('success', 'failed', 'partial', 'no_data')),
    last_error_message TEXT,

    -- Data Statistics
    historical_quarters_found INTEGER DEFAULT 0,
    upcoming_events_found INTEGER DEFAULT 0,
    total_syncs INTEGER DEFAULT 0,
    failed_syncs INTEGER DEFAULT 0,

    -- Next Scheduled Sync
    next_sync_at TIMESTAMP WITH TIME ZONE,
    sync_frequency_hours INTEGER DEFAULT 24,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol)
);

-- ============================================================================
-- EARNINGS ALERTS TABLE
-- User-configured alerts for upcoming earnings
-- ============================================================================
CREATE TABLE IF NOT EXISTS earnings_alerts (
    id SERIAL PRIMARY KEY,
    user_id UUID,  -- Reference to users table if exists
    symbol VARCHAR(10) NOT NULL,

    -- Alert Configuration
    alert_type VARCHAR(30) CHECK (alert_type IN (
        'upcoming_earnings',
        'earnings_beat',
        'earnings_miss',
        'high_iv_pre_earnings',
        'unusual_options_activity'
    )),

    -- Trigger Conditions
    days_before_earnings INTEGER DEFAULT 1,
    min_surprise_percent DECIMAL(8,2),
    min_iv_threshold DECIMAL(6,4),

    -- Notification Settings
    is_active BOOLEAN DEFAULT TRUE,
    notification_methods TEXT[] DEFAULT ARRAY['email'],
    last_triggered_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Earnings History Indexes
CREATE INDEX IF NOT EXISTS idx_earnings_history_symbol ON earnings_history(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_history_date ON earnings_history(report_date DESC);
CREATE INDEX IF NOT EXISTS idx_earnings_history_symbol_date ON earnings_history(symbol, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_earnings_history_beat_miss ON earnings_history(beat_miss) WHERE beat_miss IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_earnings_history_quarter ON earnings_history(fiscal_year, fiscal_quarter);

-- Earnings Events Indexes
CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol ON earnings_events(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_events_date ON earnings_events(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_events_upcoming ON earnings_events(earnings_date)
    WHERE has_occurred = FALSE AND earnings_date >= CURRENT_DATE;
CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol_date ON earnings_events(symbol, earnings_date DESC);
CREATE INDEX IF NOT EXISTS idx_earnings_events_confirmed ON earnings_events(is_confirmed, earnings_date)
    WHERE is_confirmed = TRUE;

-- Sync Status Indexes
CREATE INDEX IF NOT EXISTS idx_sync_status_symbol ON earnings_sync_status(symbol);
CREATE INDEX IF NOT EXISTS idx_sync_status_next_sync ON earnings_sync_status(next_sync_at)
    WHERE next_sync_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sync_status_failed ON earnings_sync_status(last_sync_status)
    WHERE last_sync_status = 'failed';

-- Alerts Indexes
CREATE INDEX IF NOT EXISTS idx_earnings_alerts_symbol ON earnings_alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_alerts_active ON earnings_alerts(is_active, symbol)
    WHERE is_active = TRUE;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Upcoming Earnings Calendar (Next 30 Days)
CREATE OR REPLACE VIEW v_upcoming_earnings AS
SELECT
    ee.symbol,
    ee.earnings_date,
    ee.earnings_time,
    ee.eps_estimate,
    ee.revenue_estimate,
    ee.whisper_number,
    ee.pre_earnings_iv,
    ee.call_datetime,
    ee.is_confirmed,
    -- Historical Beat Rate
    (
        SELECT ROUND(
            100.0 * COUNT(*) FILTER (WHERE beat_miss = 'beat') / NULLIF(COUNT(*), 0), 2
        )
        FROM earnings_history eh
        WHERE eh.symbol = ee.symbol
        AND eh.report_date >= CURRENT_DATE - INTERVAL '2 years'
    ) as historical_beat_rate_pct,
    -- Last Quarter Performance
    (
        SELECT eps_surprise_percent
        FROM earnings_history eh
        WHERE eh.symbol = ee.symbol
        ORDER BY report_date DESC
        LIMIT 1
    ) as last_quarter_surprise_pct
FROM earnings_events ee
WHERE ee.has_occurred = FALSE
AND ee.earnings_date >= CURRENT_DATE
AND ee.earnings_date <= CURRENT_DATE + INTERVAL '30 days'
ORDER BY ee.earnings_date, ee.symbol;

-- View: Earnings Beat/Miss Statistics by Symbol
CREATE OR REPLACE VIEW v_earnings_beat_stats AS
SELECT
    symbol,
    COUNT(*) as total_reports,
    COUNT(*) FILTER (WHERE beat_miss = 'beat') as beats,
    COUNT(*) FILTER (WHERE beat_miss = 'miss') as misses,
    COUNT(*) FILTER (WHERE beat_miss = 'meet') as meets,
    ROUND(100.0 * COUNT(*) FILTER (WHERE beat_miss = 'beat') / NULLIF(COUNT(*), 0), 2) as beat_rate_pct,
    ROUND(AVG(eps_surprise_percent), 2) as avg_surprise_pct,
    ROUND(STDDEV(eps_surprise_percent), 2) as surprise_volatility,
    MAX(report_date) as last_earnings_date,
    MIN(report_date) as first_earnings_date
FROM earnings_history
WHERE report_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY symbol
ORDER BY beat_rate_pct DESC;

-- View: High Conviction Earnings Plays
CREATE OR REPLACE VIEW v_high_conviction_earnings AS
SELECT
    ee.symbol,
    ee.earnings_date,
    ee.earnings_time,
    ee.eps_estimate,
    ee.pre_earnings_iv,
    stats.beat_rate_pct,
    stats.avg_surprise_pct,
    stats.total_reports,
    CASE
        WHEN stats.beat_rate_pct >= 75 THEN 'Strong Beat History'
        WHEN stats.beat_rate_pct <= 25 THEN 'Strong Miss History'
        ELSE 'Mixed History'
    END as conviction_signal
FROM earnings_events ee
JOIN v_earnings_beat_stats stats ON ee.symbol = stats.symbol
WHERE ee.has_occurred = FALSE
AND ee.earnings_date >= CURRENT_DATE
AND ee.earnings_date <= CURRENT_DATE + INTERVAL '14 days'
AND stats.total_reports >= 4  -- At least 4 quarters of history
AND (stats.beat_rate_pct >= 75 OR stats.beat_rate_pct <= 25)  -- Strong pattern
ORDER BY ee.earnings_date, stats.beat_rate_pct DESC;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function: Calculate Beat Rate for a Symbol
CREATE OR REPLACE FUNCTION calculate_beat_rate(
    p_symbol VARCHAR(10),
    p_lookback_quarters INTEGER DEFAULT 8
) RETURNS DECIMAL(5,2) AS $$
DECLARE
    v_beat_rate DECIMAL(5,2);
BEGIN
    SELECT
        ROUND(100.0 * COUNT(*) FILTER (WHERE beat_miss = 'beat') / NULLIF(COUNT(*), 0), 2)
    INTO v_beat_rate
    FROM (
        SELECT beat_miss
        FROM earnings_history
        WHERE symbol = p_symbol
        AND beat_miss IS NOT NULL
        ORDER BY report_date DESC
        LIMIT p_lookback_quarters
    ) recent_quarters;

    RETURN COALESCE(v_beat_rate, 0);
END;
$$ LANGUAGE plpgsql;

-- Function: Get Next Earnings Date for Symbol
CREATE OR REPLACE FUNCTION get_next_earnings_date(p_symbol VARCHAR(10))
RETURNS DATE AS $$
DECLARE
    v_next_date DATE;
BEGIN
    SELECT earnings_date
    INTO v_next_date
    FROM earnings_events
    WHERE symbol = p_symbol
    AND has_occurred = FALSE
    AND earnings_date >= CURRENT_DATE
    ORDER BY earnings_date
    LIMIT 1;

    RETURN v_next_date;
END;
$$ LANGUAGE plpgsql;

-- Function: Update Sync Status
CREATE OR REPLACE FUNCTION update_sync_status(
    p_symbol VARCHAR(10),
    p_status VARCHAR(20),
    p_historical_count INTEGER DEFAULT 0,
    p_upcoming_count INTEGER DEFAULT 0,
    p_error_message TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO earnings_sync_status (
        symbol,
        last_sync_at,
        last_sync_status,
        last_error_message,
        historical_quarters_found,
        upcoming_events_found,
        total_syncs,
        failed_syncs,
        next_sync_at,
        updated_at
    ) VALUES (
        p_symbol,
        NOW(),
        p_status,
        p_error_message,
        p_historical_count,
        p_upcoming_count,
        1,
        CASE WHEN p_status = 'failed' THEN 1 ELSE 0 END,
        NOW() + INTERVAL '24 hours',
        NOW()
    )
    ON CONFLICT (symbol) DO UPDATE SET
        last_sync_at = NOW(),
        last_sync_status = p_status,
        last_error_message = p_error_message,
        historical_quarters_found = p_historical_count,
        upcoming_events_found = p_upcoming_count,
        total_syncs = earnings_sync_status.total_syncs + 1,
        failed_syncs = earnings_sync_status.failed_syncs +
            CASE WHEN p_status = 'failed' THEN 1 ELSE 0 END,
        next_sync_at = NOW() + INTERVAL '24 hours',
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Auto-update timestamps
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_earnings_history_timestamp
    BEFORE UPDATE ON earnings_history
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_earnings_events_timestamp
    BEFORE UPDATE ON earnings_events
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_earnings_sync_status_timestamp
    BEFORE UPDATE ON earnings_sync_status
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- ============================================================================
-- SAMPLE QUERIES (Commented)
-- ============================================================================

/*
-- Get upcoming earnings in next 7 days with high beat rates
SELECT * FROM v_high_conviction_earnings
WHERE earnings_date <= CURRENT_DATE + INTERVAL '7 days';

-- Get historical performance for a symbol
SELECT
    symbol,
    report_date,
    eps_actual,
    eps_estimate,
    eps_surprise_percent,
    beat_miss
FROM earnings_history
WHERE symbol = 'AAPL'
ORDER BY report_date DESC
LIMIT 8;

-- Get symbols that need sync (haven't synced in 24 hours)
SELECT symbol
FROM earnings_sync_status
WHERE last_sync_at < NOW() - INTERVAL '24 hours'
   OR last_sync_at IS NULL
ORDER BY last_sync_at NULLS FIRST;

-- Get beat rate for specific symbol
SELECT calculate_beat_rate('NVDA', 8);

-- Get all upcoming earnings this week grouped by day
SELECT
    earnings_date,
    COUNT(*) as num_earnings,
    STRING_AGG(symbol, ', ' ORDER BY symbol) as symbols
FROM earnings_events
WHERE earnings_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
AND has_occurred = FALSE
GROUP BY earnings_date
ORDER BY earnings_date;
*/
