-- ============================================================================
-- AVA World-Class Advisor Enhancement
-- Migration: 004_world_class_advisor.sql
-- Version: 4.0.0
-- Date: 2025-11-28
-- Description: Adds proactive monitoring, goal tracking, recommendation learning,
--              and intelligent alert system for passive income advisory
-- ============================================================================

BEGIN;

-- ============================================================================
-- PART 1: Alert System Tables
-- ============================================================================

-- Alert priority levels
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_priority') THEN
        CREATE TYPE alert_priority AS ENUM ('urgent', 'important', 'informational');
    END IF;
END $$;

-- Alert categories
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_category') THEN
        CREATE TYPE alert_category AS ENUM (
            'assignment_risk',
            'earnings_proximity',
            'opportunity_csp',
            'opportunity_cc',
            'iv_spike',
            'xtrades_new',
            'margin_warning',
            'theta_decay',
            'expiration_reminder',
            'goal_progress',
            'report_ready'
        );
    END IF;
END $$;

-- Alert delivery channels
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'alert_channel') THEN
        CREATE TYPE alert_channel AS ENUM ('telegram', 'discord', 'email', 'push', 'in_app');
    END IF;
END $$;

-- Main alerts table
CREATE TABLE IF NOT EXISTS ava_alerts (
    id SERIAL PRIMARY KEY,

    -- Classification
    category alert_category NOT NULL,
    priority alert_priority NOT NULL,

    -- Content
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,  -- Flexible data for each alert type

    -- Related entities
    symbol VARCHAR(20),                   -- Related stock symbol (if applicable)
    position_id UUID,                     -- Related position (if applicable)

    -- Deduplication
    fingerprint VARCHAR(64) UNIQUE,       -- Hash of key alert properties

    -- Lifecycle
    is_active BOOLEAN DEFAULT TRUE,
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for ava_alerts
CREATE INDEX IF NOT EXISTS idx_ava_alerts_category ON ava_alerts(category, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_alerts_priority ON ava_alerts(priority, is_active);
CREATE INDEX IF NOT EXISTS idx_ava_alerts_active ON ava_alerts(is_active, created_at DESC) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_ava_alerts_symbol ON ava_alerts(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ava_alerts_fingerprint ON ava_alerts(fingerprint);
CREATE INDEX IF NOT EXISTS idx_ava_alerts_metadata ON ava_alerts USING GIN(metadata);

COMMENT ON TABLE ava_alerts IS 'Proactive alerts for position risks, opportunities, and reports';
COMMENT ON COLUMN ava_alerts.fingerprint IS 'SHA256 hash for deduplication (prevents duplicate alerts)';
COMMENT ON COLUMN ava_alerts.metadata IS 'Flexible JSON for alert-specific data (strike, expiry, score, etc.)';


-- User alert preferences per category
CREATE TABLE IF NOT EXISTS ava_alert_preferences (
    id SERIAL PRIMARY KEY,

    -- User identification
    user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
    platform VARCHAR(50) NOT NULL DEFAULT 'web',

    -- Category settings
    category alert_category NOT NULL,
    enabled BOOLEAN DEFAULT TRUE,
    priority_threshold alert_priority DEFAULT 'informational',

    -- Channels (array of enabled channels)
    channels TEXT[] DEFAULT ARRAY['telegram']::TEXT[],

    -- Quiet hours (don't disturb during these times)
    quiet_hours_enabled BOOLEAN DEFAULT FALSE,
    quiet_hours_start TIME,               -- e.g., 22:00
    quiet_hours_end TIME,                 -- e.g., 07:00

    -- Throttling
    max_per_hour INTEGER DEFAULT 10,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, platform, category)
);

-- Indexes for ava_alert_preferences
CREATE INDEX IF NOT EXISTS idx_ava_alert_prefs_user ON ava_alert_preferences(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_alert_prefs_category ON ava_alert_preferences(category);

COMMENT ON TABLE ava_alert_preferences IS 'User preferences for alert categories and delivery channels';


-- Alert delivery tracking
CREATE TABLE IF NOT EXISTS ava_alert_deliveries (
    id SERIAL PRIMARY KEY,
    alert_id INTEGER REFERENCES ava_alerts(id) ON DELETE CASCADE,

    -- Delivery info
    channel alert_channel NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, sent, failed, throttled

    -- Tracking
    sent_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- External references
    external_message_id VARCHAR(100),      -- Telegram message_id, etc.

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for ava_alert_deliveries
CREATE INDEX IF NOT EXISTS idx_ava_deliveries_alert ON ava_alert_deliveries(alert_id);
CREATE INDEX IF NOT EXISTS idx_ava_deliveries_status ON ava_alert_deliveries(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_deliveries_channel ON ava_alert_deliveries(channel, status);

COMMENT ON TABLE ava_alert_deliveries IS 'Tracks alert delivery status per channel with retry support';


-- Rate limiting per channel
CREATE TABLE IF NOT EXISTS ava_alert_rate_limits (
    id SERIAL PRIMARY KEY,

    -- Rate limit scope
    user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
    channel alert_channel NOT NULL,

    -- Window tracking
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_duration_minutes INTEGER DEFAULT 60,
    max_alerts INTEGER DEFAULT 10,
    alerts_sent INTEGER DEFAULT 0,

    UNIQUE(user_id, channel, window_start)
);

-- Index for rate limit lookups
CREATE INDEX IF NOT EXISTS idx_ava_rate_limits_lookup ON ava_alert_rate_limits(user_id, channel, window_start DESC);

COMMENT ON TABLE ava_alert_rate_limits IS 'Rate limiting state to prevent alert spam';


-- ============================================================================
-- PART 2: Goal Tracking Tables
-- ============================================================================

-- User income/return goals
CREATE TABLE IF NOT EXISTS ava_user_goals (
    id SERIAL PRIMARY KEY,

    -- User identification
    user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
    platform VARCHAR(50) NOT NULL DEFAULT 'web',

    -- Goal definition
    goal_type VARCHAR(50) NOT NULL,        -- 'monthly_income', 'annual_return', 'risk_budget', 'win_rate'
    goal_name VARCHAR(200) NOT NULL,
    target_value DECIMAL(15,2) NOT NULL,   -- e.g., 2500.00 for $2,500/month
    target_unit VARCHAR(50) NOT NULL,      -- 'USD', 'percent', 'trades', 'ratio'

    -- Time frame
    period_type VARCHAR(20) NOT NULL,      -- 'daily', 'weekly', 'monthly', 'quarterly', 'annual'
    start_date DATE NOT NULL DEFAULT CURRENT_DATE,
    end_date DATE,                         -- NULL = ongoing

    -- Progress tracking
    current_value DECIMAL(15,2) DEFAULT 0,
    progress_pct DECIMAL(5,2) DEFAULT 0,
    last_updated_at TIMESTAMP WITH TIME ZONE,

    -- Strategy constraints (optional)
    allowed_strategies TEXT[],             -- ['CSP', 'CC', 'wheel']
    max_position_size DECIMAL(10,2),       -- Max $ per position
    max_total_exposure DECIMAL(10,2),      -- Max total $ at risk

    -- Status
    status VARCHAR(20) DEFAULT 'active',   -- 'active', 'paused', 'completed', 'abandoned'

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, platform, goal_type, goal_name),
    CONSTRAINT chk_progress CHECK (progress_pct >= 0 AND progress_pct <= 200)  -- Allow up to 200% (exceeded)
);

-- Indexes for ava_user_goals
CREATE INDEX IF NOT EXISTS idx_ava_goals_user ON ava_user_goals(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_goals_active ON ava_user_goals(status, user_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_ava_goals_type ON ava_user_goals(goal_type);
CREATE INDEX IF NOT EXISTS idx_ava_goals_period ON ava_user_goals(period_type, start_date);

COMMENT ON TABLE ava_user_goals IS 'User income and return goals with progress tracking';
COMMENT ON COLUMN ava_user_goals.target_value IS 'Target value (e.g., 2500 for $2,500/month income)';
COMMENT ON COLUMN ava_user_goals.progress_pct IS 'Progress percentage (can exceed 100% if goal is surpassed)';


-- Goal progress history (snapshots)
CREATE TABLE IF NOT EXISTS ava_goal_progress_history (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES ava_user_goals(id) ON DELETE CASCADE,

    -- Progress snapshot
    snapshot_date DATE NOT NULL,
    period_value DECIMAL(15,2) NOT NULL,        -- Value achieved this period
    cumulative_value DECIMAL(15,2) NOT NULL,    -- Cumulative value
    progress_pct DECIMAL(5,2) NOT NULL,

    -- Contributing activity
    trades_count INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    premium_collected DECIMAL(15,2) DEFAULT 0,
    total_pnl DECIMAL(15,2) DEFAULT 0,

    -- Analysis
    notes TEXT,
    ai_analysis TEXT,                           -- AI-generated progress analysis

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(goal_id, snapshot_date)
);

-- Indexes for ava_goal_progress_history
CREATE INDEX IF NOT EXISTS idx_ava_goal_history_goal ON ava_goal_progress_history(goal_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_ava_goal_history_date ON ava_goal_progress_history(snapshot_date DESC);

COMMENT ON TABLE ava_goal_progress_history IS 'Historical snapshots of goal progress for trend analysis';


-- ============================================================================
-- PART 3: Recommendation Tracking & Learning Tables
-- ============================================================================

-- Track all chatbot recommendations
CREATE TABLE IF NOT EXISTS ava_chat_recommendations (
    id SERIAL PRIMARY KEY,

    -- Context
    user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
    platform VARCHAR(50) NOT NULL DEFAULT 'web',
    conversation_id VARCHAR(100),

    -- Recommendation details
    recommendation_type VARCHAR(50) NOT NULL,  -- 'trade', 'strategy', 'adjustment', 'exit', 'hold'
    symbol VARCHAR(20),
    strategy VARCHAR(100),
    recommendation_text TEXT NOT NULL,
    confidence_score DECIMAL(3,2),             -- 0.00-1.00
    reasoning TEXT,

    -- Context snapshot (what data was used)
    context_snapshot JSONB,                    -- Portfolio state, market conditions at time of rec
    rag_sources_used TEXT[],                   -- RAG document IDs used
    agents_used TEXT[],                        -- Which agents contributed

    -- Outcome tracking
    user_action VARCHAR(50),                   -- 'accepted', 'rejected', 'modified', 'ignored'
    user_action_at TIMESTAMP WITH TIME ZONE,
    trade_id UUID,                             -- Link to actual trade if executed

    -- Actual outcome
    actual_outcome VARCHAR(50),                -- 'win', 'loss', 'breakeven', 'pending', 'expired'
    actual_pnl DECIMAL(15,2),
    outcome_recorded_at TIMESTAMP WITH TIME ZONE,

    -- Learning feedback
    recommendation_correct BOOLEAN,            -- Did the recommendation turn out to be correct?
    feedback_score INTEGER,                    -- User rating 1-5
    feedback_text TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for ava_chat_recommendations
CREATE INDEX IF NOT EXISTS idx_ava_recs_user ON ava_chat_recommendations(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_recs_symbol ON ava_chat_recommendations(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ava_recs_outcome ON ava_chat_recommendations(actual_outcome) WHERE actual_outcome IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ava_recs_correct ON ava_chat_recommendations(recommendation_correct) WHERE recommendation_correct IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ava_recs_created ON ava_chat_recommendations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_recs_context ON ava_chat_recommendations USING GIN(context_snapshot);

COMMENT ON TABLE ava_chat_recommendations IS 'Tracks chatbot recommendations with outcomes for learning';
COMMENT ON COLUMN ava_chat_recommendations.context_snapshot IS 'JSON snapshot of portfolio and market state when recommendation was made';
COMMENT ON COLUMN ava_chat_recommendations.recommendation_correct IS 'Whether the recommendation was ultimately correct (for learning)';


-- Learned patterns from user trading history
CREATE TABLE IF NOT EXISTS ava_learning_patterns (
    id SERIAL PRIMARY KEY,

    -- User identification
    user_id VARCHAR(100) NOT NULL DEFAULT 'default_user',
    platform VARCHAR(50) NOT NULL DEFAULT 'web',

    -- Pattern identification
    pattern_type VARCHAR(50) NOT NULL,         -- 'winning_setup', 'losing_setup', 'preference', 'timing', 'sizing'
    pattern_name VARCHAR(200) NOT NULL,
    pattern_description TEXT,

    -- Pattern conditions (what defines this pattern)
    pattern_conditions JSONB NOT NULL,         -- e.g., {"iv_rank": ">70", "dte": "30-45", "delta": "0.20-0.30"}

    -- Evidence
    sample_trades JSONB,                       -- Trade IDs that matched this pattern
    sample_count INTEGER DEFAULT 0,

    -- Performance metrics
    win_rate DECIMAL(5,2),
    avg_pnl DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    avg_holding_days DECIMAL(5,1),

    -- Confidence
    confidence_score DECIMAL(3,2) DEFAULT 0.50,
    last_validated_at TIMESTAMP WITH TIME ZONE,
    validation_count INTEGER DEFAULT 0,

    -- Application
    active BOOLEAN DEFAULT TRUE,
    weight_multiplier DECIMAL(3,2) DEFAULT 1.0,  -- For recommendation scoring

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, platform, pattern_type, pattern_name)
);

-- Indexes for ava_learning_patterns
CREATE INDEX IF NOT EXISTS idx_ava_patterns_user ON ava_learning_patterns(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_patterns_type ON ava_learning_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_ava_patterns_active ON ava_learning_patterns(active, confidence_score DESC) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_ava_patterns_conditions ON ava_learning_patterns USING GIN(pattern_conditions);

COMMENT ON TABLE ava_learning_patterns IS 'Learned patterns from user trading history for personalization';
COMMENT ON COLUMN ava_learning_patterns.pattern_conditions IS 'JSON conditions that define this pattern (IV, DTE, delta, etc.)';
COMMENT ON COLUMN ava_learning_patterns.weight_multiplier IS 'Multiplier applied to recommendations matching this pattern';


-- ============================================================================
-- PART 4: Monitoring State Tables
-- ============================================================================

-- Track automatic scan results
CREATE TABLE IF NOT EXISTS ava_opportunity_scans (
    id SERIAL PRIMARY KEY,

    -- Scan metadata
    scan_type VARCHAR(50) NOT NULL,            -- 'auto_premium', 'iv_spike', 'calendar_spread'
    watchlist_used VARCHAR(100),
    symbols_scanned INTEGER DEFAULT 0,

    -- Results summary
    opportunities_found INTEGER DEFAULT 0,
    alerts_generated INTEGER DEFAULT 0,

    -- Top opportunities (up to 10)
    top_opportunities JSONB DEFAULT '[]'::jsonb,

    -- Execution stats
    scan_duration_ms INTEGER,
    errors TEXT[],

    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Index for scan history
CREATE INDEX IF NOT EXISTS idx_ava_scans_type ON ava_opportunity_scans(scan_type, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_scans_started ON ava_opportunity_scans(started_at DESC);

COMMENT ON TABLE ava_opportunity_scans IS 'History of automatic opportunity scans';


-- IV history for spike detection
CREATE TABLE IF NOT EXISTS ava_iv_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,

    -- IV metrics
    iv_30 DECIMAL(6,2),                        -- 30-day IV
    iv_60 DECIMAL(6,2),                        -- 60-day IV
    iv_rank DECIMAL(5,2),                      -- IV rank (0-100)
    iv_percentile DECIMAL(5,2),                -- IV percentile (0-100)

    -- Context
    stock_price DECIMAL(10,2),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(symbol, date)
);

-- Indexes for IV history
CREATE INDEX IF NOT EXISTS idx_ava_iv_symbol ON ava_iv_history(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_ava_iv_date ON ava_iv_history(date DESC);
CREATE INDEX IF NOT EXISTS idx_ava_iv_rank ON ava_iv_history(iv_rank DESC) WHERE iv_rank IS NOT NULL;

COMMENT ON TABLE ava_iv_history IS 'Historical IV data for spike detection and trend analysis';


-- Generated reports storage
CREATE TABLE IF NOT EXISTS ava_generated_reports (
    id SERIAL PRIMARY KEY,

    -- Report type
    report_type VARCHAR(50) NOT NULL,          -- 'morning_briefing', 'weekly_summary', 'monthly_report', 'expiration_review'
    report_date DATE NOT NULL,

    -- Content
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,                     -- Markdown formatted report
    summary TEXT,                              -- Short summary for notifications

    -- Metadata
    metrics JSONB DEFAULT '{}'::jsonb,         -- Key metrics included in report

    -- Delivery status
    telegram_sent BOOLEAN DEFAULT FALSE,
    email_sent BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(report_type, report_date)
);

-- Index for report retrieval
CREATE INDEX IF NOT EXISTS idx_ava_reports_type ON ava_generated_reports(report_type, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_ava_reports_date ON ava_generated_reports(report_date DESC);

COMMENT ON TABLE ava_generated_reports IS 'Storage for generated reports (morning briefings, summaries, etc.)';


-- ============================================================================
-- PART 5: Functions and Triggers
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_ava_advisor_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_alert_prefs_updated_at') THEN
        CREATE TRIGGER ava_alert_prefs_updated_at
            BEFORE UPDATE ON ava_alert_preferences
            FOR EACH ROW
            EXECUTE FUNCTION update_ava_advisor_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_goals_updated_at') THEN
        CREATE TRIGGER ava_goals_updated_at
            BEFORE UPDATE ON ava_user_goals
            FOR EACH ROW
            EXECUTE FUNCTION update_ava_advisor_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_patterns_updated_at') THEN
        CREATE TRIGGER ava_patterns_updated_at
            BEFORE UPDATE ON ava_learning_patterns
            FOR EACH ROW
            EXECUTE FUNCTION update_ava_advisor_updated_at();
    END IF;
END $$;


-- Function: Check rate limit and increment counter
CREATE OR REPLACE FUNCTION check_and_increment_rate_limit(
    p_user_id VARCHAR,
    p_channel alert_channel,
    p_max_per_hour INTEGER DEFAULT 10
)
RETURNS BOOLEAN AS $$
DECLARE
    current_window TIMESTAMP WITH TIME ZONE;
    current_count INTEGER;
BEGIN
    -- Get current hour window
    current_window := date_trunc('hour', NOW());

    -- Try to get or create rate limit record
    INSERT INTO ava_alert_rate_limits (user_id, channel, window_start, max_alerts, alerts_sent)
    VALUES (p_user_id, p_channel, current_window, p_max_per_hour, 1)
    ON CONFLICT (user_id, channel, window_start)
    DO UPDATE SET alerts_sent = ava_alert_rate_limits.alerts_sent + 1
    RETURNING alerts_sent INTO current_count;

    -- Check if under limit
    RETURN current_count <= p_max_per_hour;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION check_and_increment_rate_limit IS 'Check rate limit and increment counter, returns TRUE if allowed';


-- Function: Generate alert fingerprint
CREATE OR REPLACE FUNCTION generate_alert_fingerprint(
    p_category alert_category,
    p_symbol VARCHAR,
    p_metadata JSONB
)
RETURNS VARCHAR AS $$
DECLARE
    fingerprint_key TEXT;
BEGIN
    -- Build fingerprint based on category
    CASE p_category
        WHEN 'assignment_risk' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              COALESCE(p_metadata->>'strike', '') || ':' ||
                              COALESCE(p_metadata->>'expiration', '');
        WHEN 'opportunity_csp', 'opportunity_cc' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              COALESCE(p_metadata->>'strike', '') || ':' ||
                              date_trunc('hour', NOW())::TEXT;
        WHEN 'xtrades_new' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_metadata->>'trade_id', '');
        WHEN 'earnings_proximity' THEN
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              COALESCE(p_metadata->>'earnings_date', '');
        ELSE
            fingerprint_key := p_category || ':' || COALESCE(p_symbol, '') || ':' ||
                              date_trunc('hour', NOW())::TEXT;
    END CASE;

    -- Return MD5 hash (32 chars)
    RETURN MD5(fingerprint_key);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION generate_alert_fingerprint IS 'Generate unique fingerprint for alert deduplication';


-- Function: Calculate goal progress from trades
CREATE OR REPLACE FUNCTION calculate_goal_progress(
    p_goal_id INTEGER
)
RETURNS TABLE (
    current_value DECIMAL(15,2),
    progress_pct DECIMAL(5,2),
    trades_count INTEGER,
    winning_trades INTEGER
) AS $$
DECLARE
    v_goal ava_user_goals%ROWTYPE;
    v_start_date DATE;
    v_end_date DATE;
    v_total_premium DECIMAL(15,2);
    v_trade_count INTEGER;
    v_win_count INTEGER;
BEGIN
    -- Get goal details
    SELECT * INTO v_goal FROM ava_user_goals WHERE id = p_goal_id;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Calculate date range based on period type
    CASE v_goal.period_type
        WHEN 'monthly' THEN
            v_start_date := date_trunc('month', CURRENT_DATE)::DATE;
            v_end_date := (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month' - INTERVAL '1 day')::DATE;
        WHEN 'weekly' THEN
            v_start_date := date_trunc('week', CURRENT_DATE)::DATE;
            v_end_date := (date_trunc('week', CURRENT_DATE) + INTERVAL '6 days')::DATE;
        WHEN 'annual' THEN
            v_start_date := date_trunc('year', CURRENT_DATE)::DATE;
            v_end_date := (date_trunc('year', CURRENT_DATE) + INTERVAL '1 year' - INTERVAL '1 day')::DATE;
        ELSE
            v_start_date := v_goal.start_date;
            v_end_date := COALESCE(v_goal.end_date, CURRENT_DATE);
    END CASE;

    -- Calculate from trade_journal if it exists (otherwise return 0)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trade_journal') THEN
        SELECT
            COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0),
            COUNT(*),
            COUNT(*) FILTER (WHERE pnl > 0)
        INTO v_total_premium, v_trade_count, v_win_count
        FROM trade_journal
        WHERE closed_date >= v_start_date
          AND closed_date <= v_end_date;
    ELSE
        v_total_premium := 0;
        v_trade_count := 0;
        v_win_count := 0;
    END IF;

    -- Return results
    current_value := v_total_premium;
    progress_pct := CASE WHEN v_goal.target_value > 0
                        THEN LEAST((v_total_premium / v_goal.target_value * 100), 200)
                        ELSE 0 END;
    trades_count := v_trade_count;
    winning_trades := v_win_count;

    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION calculate_goal_progress IS 'Calculate goal progress from trade journal data';


-- Function: Get recommendation accuracy stats
CREATE OR REPLACE FUNCTION get_recommendation_accuracy(
    p_user_id VARCHAR DEFAULT 'default_user',
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_recommendations INTEGER,
    recommendations_with_outcome INTEGER,
    correct_recommendations INTEGER,
    accuracy_pct DECIMAL(5,2),
    avg_confidence DECIMAL(3,2),
    avg_pnl DECIMAL(15,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_recommendations,
        COUNT(*) FILTER (WHERE actual_outcome IS NOT NULL)::INTEGER as recommendations_with_outcome,
        COUNT(*) FILTER (WHERE recommendation_correct = TRUE)::INTEGER as correct_recommendations,
        CASE
            WHEN COUNT(*) FILTER (WHERE recommendation_correct IS NOT NULL) > 0
            THEN (COUNT(*) FILTER (WHERE recommendation_correct = TRUE)::DECIMAL /
                  COUNT(*) FILTER (WHERE recommendation_correct IS NOT NULL) * 100)
            ELSE 0
        END as accuracy_pct,
        AVG(confidence_score)::DECIMAL(3,2) as avg_confidence,
        AVG(actual_pnl)::DECIMAL(15,2) as avg_pnl
    FROM ava_chat_recommendations
    WHERE user_id = p_user_id
      AND created_at >= NOW() - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_recommendation_accuracy IS 'Get recommendation accuracy statistics for a user';


-- ============================================================================
-- PART 6: Views
-- ============================================================================

-- View: Active alerts summary
CREATE OR REPLACE VIEW v_ava_active_alerts AS
SELECT
    category,
    priority,
    COUNT(*) as count,
    MIN(created_at) as oldest,
    MAX(created_at) as newest
FROM ava_alerts
WHERE is_active = TRUE
  AND (expires_at IS NULL OR expires_at > NOW())
GROUP BY category, priority
ORDER BY priority, category;

COMMENT ON VIEW v_ava_active_alerts IS 'Summary of active alerts by category and priority';


-- View: Goal dashboard
CREATE OR REPLACE VIEW v_ava_goal_dashboard AS
SELECT
    g.id,
    g.goal_name,
    g.goal_type,
    g.target_value,
    g.target_unit,
    g.current_value,
    g.progress_pct,
    g.period_type,
    g.status,
    CASE
        WHEN g.progress_pct >= 100 THEN 'exceeded'
        WHEN g.progress_pct >= 75 THEN 'on_track'
        WHEN g.progress_pct >= 50 THEN 'moderate'
        ELSE 'behind'
    END as progress_status,
    g.updated_at as last_updated
FROM ava_user_goals g
WHERE g.status = 'active'
ORDER BY g.progress_pct DESC;

COMMENT ON VIEW v_ava_goal_dashboard IS 'Dashboard view of active goals with progress status';


-- View: Learning patterns summary
CREATE OR REPLACE VIEW v_ava_pattern_summary AS
SELECT
    user_id,
    pattern_type,
    COUNT(*) as pattern_count,
    AVG(win_rate) as avg_win_rate,
    AVG(confidence_score) as avg_confidence,
    SUM(sample_count) as total_samples
FROM ava_learning_patterns
WHERE active = TRUE
GROUP BY user_id, pattern_type
ORDER BY avg_win_rate DESC;

COMMENT ON VIEW v_ava_pattern_summary IS 'Summary of learned patterns by type';


-- ============================================================================
-- PART 7: Default Data
-- ============================================================================

-- Insert default alert preferences
INSERT INTO ava_alert_preferences (user_id, platform, category, enabled, priority_threshold, channels, max_per_hour)
VALUES
    ('default_user', 'web', 'assignment_risk', TRUE, 'urgent', ARRAY['telegram', 'email'], 20),
    ('default_user', 'web', 'earnings_proximity', TRUE, 'important', ARRAY['telegram'], 10),
    ('default_user', 'web', 'opportunity_csp', TRUE, 'important', ARRAY['telegram'], 10),
    ('default_user', 'web', 'opportunity_cc', TRUE, 'important', ARRAY['telegram'], 10),
    ('default_user', 'web', 'iv_spike', TRUE, 'informational', ARRAY['telegram'], 5),
    ('default_user', 'web', 'xtrades_new', TRUE, 'important', ARRAY['telegram'], 10),
    ('default_user', 'web', 'margin_warning', TRUE, 'urgent', ARRAY['telegram', 'email'], 20),
    ('default_user', 'web', 'theta_decay', TRUE, 'informational', ARRAY['telegram'], 5),
    ('default_user', 'web', 'expiration_reminder', TRUE, 'important', ARRAY['telegram'], 10),
    ('default_user', 'web', 'goal_progress', TRUE, 'informational', ARRAY['telegram'], 5),
    ('default_user', 'web', 'report_ready', TRUE, 'informational', ARRAY['telegram', 'email'], 5)
ON CONFLICT (user_id, platform, category) DO NOTHING;


-- Insert default $2,500/month income goal
INSERT INTO ava_user_goals (user_id, platform, goal_type, goal_name, target_value, target_unit, period_type, status)
VALUES
    ('default_user', 'web', 'monthly_income', 'Monthly Premium Income', 2500.00, 'USD', 'monthly', 'active')
ON CONFLICT (user_id, platform, goal_type, goal_name) DO NOTHING;


-- ============================================================================
-- Migration Completion
-- ============================================================================

-- Record migration
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'schema_migrations'
    ) THEN
        INSERT INTO schema_migrations (version, name, applied_at)
        VALUES (4, 'world_class_advisor', NOW())
        ON CONFLICT (version) DO NOTHING;
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Verify tables were created
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE 'ava_%'
ORDER BY tablename;

-- Verify enums were created
SELECT typname, enumlabel
FROM pg_type t
JOIN pg_enum e ON t.oid = e.enumtypid
WHERE typname IN ('alert_priority', 'alert_category', 'alert_channel')
ORDER BY typname, enumsortorder;

-- ============================================================================
-- Success Message
-- ============================================================================

SELECT 'AVA World-Class Advisor migration completed successfully!' as status,
       'Tables: ava_alerts, ava_alert_preferences, ava_alert_deliveries, ava_alert_rate_limits' as alert_tables,
       'Tables: ava_user_goals, ava_goal_progress_history' as goal_tables,
       'Tables: ava_chat_recommendations, ava_learning_patterns' as learning_tables,
       'Tables: ava_opportunity_scans, ava_iv_history, ava_generated_reports' as monitoring_tables;
