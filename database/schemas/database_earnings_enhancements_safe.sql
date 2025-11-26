-- ============================================================================
-- Earnings Calendar Database Enhancements (Safe Version)
-- ============================================================================

BEGIN;

-- Add missing fields to earnings_events table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='expected_move_dollars') THEN
        ALTER TABLE earnings_events ADD COLUMN expected_move_dollars DECIMAL(10, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='expected_move_pct') THEN
        ALTER TABLE earnings_events ADD COLUMN expected_move_pct DECIMAL(10, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='price_move_dollars') THEN
        ALTER TABLE earnings_events ADD COLUMN price_move_dollars DECIMAL(10, 2);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='exceeded_expected_move') THEN
        ALTER TABLE earnings_events ADD COLUMN exceeded_expected_move BOOLEAN DEFAULT FALSE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='is_confirmed') THEN
        ALTER TABLE earnings_events ADD COLUMN is_confirmed BOOLEAN DEFAULT FALSE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='has_occurred') THEN
        ALTER TABLE earnings_events ADD COLUMN has_occurred BOOLEAN DEFAULT FALSE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='fiscal_quarter') THEN
        ALTER TABLE earnings_events ADD COLUMN fiscal_quarter INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='fiscal_year') THEN
        ALTER TABLE earnings_events ADD COLUMN fiscal_year INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='call_datetime') THEN
        ALTER TABLE earnings_events ADD COLUMN call_datetime TIMESTAMP WITH TIME ZONE;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='call_broadcast_url') THEN
        ALTER TABLE earnings_events ADD COLUMN call_broadcast_url TEXT;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='earnings_events' AND column_name='raw_data') THEN
        ALTER TABLE earnings_events ADD COLUMN raw_data JSONB;
    END IF;
END $$;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_earnings_events_date_future
ON earnings_events(earnings_date);

CREATE INDEX IF NOT EXISTS idx_earnings_events_quality
ON earnings_events(earnings_date, pre_earnings_iv);

-- Create earnings pattern analysis table
CREATE TABLE IF NOT EXISTS earnings_pattern_analysis (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    beat_rate_8q DECIMAL(5, 2),
    miss_rate_8q DECIMAL(5, 2),
    meet_rate_8q DECIMAL(5, 2),
    avg_surprise_pct_8q DECIMAL(10, 2),
    surprise_std_8q DECIMAL(10, 2),
    revenue_beat_rate DECIMAL(5, 2),
    exceed_expected_move_rate DECIMAL(5, 2),
    quality_score DECIMAL(5, 2),
    quarters_analyzed INTEGER,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pattern_quality
ON earnings_pattern_analysis(quality_score DESC);

-- Create IV tracking table (for time-series analysis)
CREATE TABLE IF NOT EXISTS earnings_iv_tracking (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    earnings_date DATE NOT NULL,
    snapshot_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    iv_rank DECIMAL(5, 2),
    iv_percentile DECIMAL(5, 2),
    atm_iv DECIMAL(10, 4),
    atm_call_price DECIMAL(10, 2),
    atm_put_price DECIMAL(10, 2),
    straddle_price DECIMAL(10, 2),
    days_to_earnings INTEGER,
    stock_price DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, earnings_date, snapshot_timestamp)
);

CREATE INDEX IF NOT EXISTS idx_iv_tracking_symbol_date
ON earnings_iv_tracking(symbol, earnings_date, snapshot_timestamp);

-- Create useful views

-- View: Upcoming high-quality earnings
CREATE OR REPLACE VIEW v_upcoming_quality_earnings AS
SELECT
    e.symbol,
    e.earnings_date,
    e.earnings_time,
    e.expected_move_pct,
    e.pre_earnings_iv,
    e.pre_earnings_price,
    p.beat_rate_8q,
    p.avg_surprise_pct_8q,
    p.quality_score,
    s.company_name,
    s.sector,
    s.market_cap
FROM earnings_events e
LEFT JOIN earnings_pattern_analysis p ON e.symbol = p.symbol
LEFT JOIN stocks s ON e.symbol = s.symbol
WHERE e.has_occurred = FALSE
ORDER BY e.earnings_date, p.quality_score DESC NULLS LAST;

-- View: Historical earnings with results
CREATE OR REPLACE VIEW v_earnings_results AS
SELECT
    e.symbol,
    e.earnings_date,
    e.eps_actual,
    e.eps_estimate,
    e.eps_actual - e.eps_estimate as eps_surprise,
    CASE
        WHEN e.eps_estimate != 0 THEN
            ((e.eps_actual - e.eps_estimate) / ABS(e.eps_estimate)) * 100
        ELSE NULL
    END as eps_surprise_pct,
    e.revenue_actual,
    e.revenue_estimate,
    e.expected_move_pct,
    e.price_move_percent as actual_move_pct,
    e.exceeded_expected_move,
    e.pre_earnings_iv,
    e.post_earnings_iv,
    CASE
        WHEN e.pre_earnings_iv IS NOT NULL AND e.post_earnings_iv IS NOT NULL THEN
            e.pre_earnings_iv - e.post_earnings_iv
        ELSE NULL
    END as iv_crush,
    e.volume_ratio,
    CASE
        WHEN e.eps_actual > e.eps_estimate THEN 'beat'
        WHEN e.eps_actual < e.eps_estimate THEN 'miss'
        WHEN e.eps_actual = e.eps_estimate THEN 'meet'
        ELSE 'pending'
    END as result,
    s.company_name,
    s.sector
FROM earnings_events e
LEFT JOIN stocks s ON e.symbol = s.symbol
WHERE e.eps_actual IS NOT NULL
ORDER BY e.earnings_date DESC;

-- View: IV expansion tracking
CREATE OR REPLACE VIEW v_iv_expansion AS
SELECT
    symbol,
    earnings_date,
    MAX(atm_iv) as peak_iv,
    MIN(atm_iv) as min_iv,
    MAX(atm_iv) - MIN(atm_iv) as iv_expansion,
    AVG(atm_iv) as avg_iv,
    COUNT(*) as snapshots,
    MIN(snapshot_timestamp) as first_snapshot,
    MAX(snapshot_timestamp) as last_snapshot
FROM earnings_iv_tracking
GROUP BY symbol, earnings_date
ORDER BY earnings_date DESC;

-- Useful functions

-- Function: Calculate beat rate for a symbol
CREATE OR REPLACE FUNCTION calculate_beat_rate(
    p_symbol VARCHAR(20),
    p_lookback_quarters INTEGER DEFAULT 8
)
RETURNS DECIMAL(5, 2) AS $$
DECLARE
    v_beat_rate DECIMAL(5, 2);
BEGIN
    SELECT
        (COUNT(*) FILTER (WHERE beat_miss = 'beat')::DECIMAL /
         NULLIF(COUNT(*), 0)) * 100
    INTO v_beat_rate
    FROM (
        SELECT beat_miss
        FROM earnings_history
        WHERE symbol = p_symbol
          AND beat_miss IS NOT NULL
        ORDER BY report_date DESC
        LIMIT p_lookback_quarters
    ) recent_earnings;

    RETURN COALESCE(v_beat_rate, 0);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: Get earnings quality score
CREATE OR REPLACE FUNCTION get_quality_score(p_symbol VARCHAR(20))
RETURNS DECIMAL(5, 2) AS $$
DECLARE
    v_quality_score DECIMAL(5, 2);
BEGIN
    SELECT quality_score
    INTO v_quality_score
    FROM earnings_pattern_analysis
    WHERE symbol = p_symbol;

    RETURN COALESCE(v_quality_score, 0);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: Calculate expected move from straddle
CREATE OR REPLACE FUNCTION calculate_expected_move(
    p_call_price DECIMAL(10, 2),
    p_put_price DECIMAL(10, 2),
    p_stock_price DECIMAL(10, 2)
)
RETURNS TABLE(
    expected_move_dollars DECIMAL(10, 2),
    expected_move_pct DECIMAL(10, 2),
    straddle_price DECIMAL(10, 2)
) AS $$
DECLARE
    v_straddle DECIMAL(10, 2);
    v_em_dollars DECIMAL(10, 2);
    v_em_pct DECIMAL(10, 2);
BEGIN
    v_straddle := p_call_price + p_put_price;
    v_em_dollars := v_straddle * 0.85;
    v_em_pct := (v_em_dollars / NULLIF(p_stock_price, 0)) * 100;

    RETURN QUERY SELECT v_em_dollars, v_em_pct, v_straddle;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMIT;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Earnings calendar enhancements installed successfully!';
END $$;
