-- ============================================================================
-- Earnings Calendar Database Enhancements
-- ============================================================================
-- Run this to add missing fields and optimize your earnings tables
-- ============================================================================

-- Add missing fields to earnings_events table
ALTER TABLE earnings_events
ADD COLUMN IF NOT EXISTS expected_move_dollars DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS expected_move_pct DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS price_move_dollars DECIMAL(10, 2),
ADD COLUMN IF NOT EXISTS exceeded_expected_move BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS is_confirmed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS has_occurred BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS fiscal_quarter INTEGER,
ADD COLUMN IF NOT EXISTS fiscal_year INTEGER,
ADD COLUMN IF NOT EXISTS call_datetime TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS call_broadcast_url TEXT,
ADD COLUMN IF NOT EXISTS raw_data JSONB;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_earnings_events_upcoming
ON earnings_events(earnings_date)
WHERE earnings_date >= CURRENT_DATE;

CREATE INDEX IF NOT EXISTS idx_earnings_events_quality
ON earnings_events(earnings_date, pre_earnings_iv)
WHERE earnings_date >= CURRENT_DATE
  AND pre_earnings_iv IS NOT NULL;

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
ON earnings_pattern_analysis(quality_score DESC)
WHERE quality_score IS NOT NULL;

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
    p.beat_rate_8q,
    p.avg_surprise_pct_8q,
    p.quality_score,
    s.company_name,
    s.sector,
    s.market_cap,
    s.last_price as current_price
FROM earnings_events e
LEFT JOIN earnings_pattern_analysis p ON e.symbol = p.symbol
LEFT JOIN stocks s ON e.symbol = s.symbol
WHERE e.earnings_date >= CURRENT_DATE
  AND e.earnings_date <= CURRENT_DATE + INTERVAL '30 days'
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
    ABS(e.price_move_percent) > e.expected_move_pct as exceeded_expected,
    e.pre_earnings_iv,
    e.post_earnings_iv,
    e.pre_earnings_iv - e.post_earnings_iv as iv_crush,
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
$$ LANGUAGE plpgsql;

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
$$ LANGUAGE plpgsql;

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
$$ LANGUAGE plpgsql;

-- Materialized view for performance (refresh daily)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_earnings_stats AS
SELECT
    symbol,
    COUNT(*) as total_earnings,
    COUNT(*) FILTER (WHERE beat_miss = 'beat') as beats,
    COUNT(*) FILTER (WHERE beat_miss = 'miss') as misses,
    COUNT(*) FILTER (WHERE beat_miss = 'meet') as meets,
    AVG(eps_surprise_percent) as avg_surprise_pct,
    STDDEV(eps_surprise_percent) as surprise_std,
    AVG(ABS(price_move_percent)) as avg_abs_move,
    COUNT(*) FILTER (WHERE exceeded_expected_move) as exceeded_count,
    (COUNT(*) FILTER (WHERE exceeded_expected_move)::DECIMAL /
     NULLIF(COUNT(*) FILTER (WHERE expected_move_pct IS NOT NULL), 0)) * 100 as exceed_rate
FROM earnings_events
WHERE eps_actual IS NOT NULL
  AND earnings_date >= CURRENT_DATE - INTERVAL '3 years'
GROUP BY symbol;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_earnings_stats_symbol
ON mv_earnings_stats(symbol);

-- Refresh command (run daily via scheduled job)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY mv_earnings_stats;

-- Grant permissions (adjust as needed)
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO your_user;

-- Comments for documentation
COMMENT ON TABLE earnings_pattern_analysis IS 'Cached earnings pattern metrics for quick dashboard queries';
COMMENT ON TABLE earnings_iv_tracking IS 'Time-series IV data before earnings for expansion analysis';
COMMENT ON VIEW v_upcoming_quality_earnings IS 'Upcoming earnings with quality scores for trading opportunities';
COMMENT ON VIEW v_earnings_results IS 'Historical earnings with beat/miss and price movement analysis';
COMMENT ON FUNCTION calculate_beat_rate IS 'Calculate percentage of earnings beats over last N quarters';
COMMENT ON FUNCTION get_quality_score IS 'Get cached quality score for a symbol (0-100 scale)';

-- Sample queries to test

-- Get top quality earnings in next 7 days
/*
SELECT *
FROM v_upcoming_quality_earnings
WHERE earnings_date <= CURRENT_DATE + INTERVAL '7 days'
  AND quality_score > 70
ORDER BY quality_score DESC, earnings_date;
*/

-- Get symbols with highest beat rates
/*
SELECT
    symbol,
    calculate_beat_rate(symbol, 12) as beat_rate_12q,
    get_quality_score(symbol) as quality_score
FROM stocks
WHERE is_active = TRUE
ORDER BY calculate_beat_rate(symbol, 12) DESC
LIMIT 20;
*/

-- Analyze IV expansion patterns
/*
SELECT
    symbol,
    earnings_date,
    iv_expansion,
    peak_iv,
    snapshots
FROM v_iv_expansion
WHERE earnings_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY iv_expansion DESC
LIMIT 20;
*/

-- Find stocks that consistently exceed expected move
/*
SELECT
    symbol,
    total_earnings,
    exceed_rate,
    avg_abs_move,
    avg_surprise_pct
FROM mv_earnings_stats
WHERE total_earnings >= 8
  AND exceed_rate > 40
ORDER BY exceed_rate DESC;
*/

-- Analysis: Beat rate vs actual performance
/*
SELECT
    p.symbol,
    p.beat_rate_8q,
    p.quality_score,
    AVG(e.price_move_percent) as avg_earnings_move,
    COUNT(*) as earnings_count
FROM earnings_pattern_analysis p
JOIN earnings_events e ON p.symbol = e.symbol
WHERE e.earnings_date >= CURRENT_DATE - INTERVAL '2 years'
  AND e.price_move_percent IS NOT NULL
GROUP BY p.symbol, p.beat_rate_8q, p.quality_score
HAVING COUNT(*) >= 4
ORDER BY p.quality_score DESC;
*/

COMMIT;

-- ============================================================================
-- Post-installation verification
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Earnings calendar enhancements installed successfully!';
    RAISE NOTICE '';
    RAISE NOTICE 'New tables created:';
    RAISE NOTICE '  - earnings_pattern_analysis';
    RAISE NOTICE '  - earnings_iv_tracking';
    RAISE NOTICE '';
    RAISE NOTICE 'New views created:';
    RAISE NOTICE '  - v_upcoming_quality_earnings';
    RAISE NOTICE '  - v_earnings_results';
    RAISE NOTICE '  - v_iv_expansion';
    RAISE NOTICE '';
    RAISE NOTICE 'New functions created:';
    RAISE NOTICE '  - calculate_beat_rate()';
    RAISE NOTICE '  - get_quality_score()';
    RAISE NOTICE '  - calculate_expected_move()';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '  1. Run daily_earnings_automation.py to populate data';
    RAISE NOTICE '  2. Check v_upcoming_quality_earnings for opportunities';
    RAISE NOTICE '  3. Refresh mv_earnings_stats daily for best performance';
END $$;
