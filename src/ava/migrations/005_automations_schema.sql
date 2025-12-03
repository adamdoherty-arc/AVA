-- ============================================================================
-- Automations Management Schema
-- ============================================================================
-- Provides enable/disable control, execution tracking, and state management
-- for all scheduled tasks and background automations.
--
-- Created: 2025-11-28
-- Author: AVA Trading Platform
-- ============================================================================

-- Table 1: automations
-- Master registry of all automations (Celery tasks + migrated scripts)
CREATE TABLE IF NOT EXISTS automations (
    id SERIAL PRIMARY KEY,

    -- Identification
    name VARCHAR(100) NOT NULL UNIQUE,              -- 'sync-kalshi-markets'
    display_name VARCHAR(200) NOT NULL,             -- 'Sync Kalshi Markets'
    automation_type VARCHAR(50) NOT NULL,           -- 'celery_beat', 'celery_task'

    -- Celery task reference
    celery_task_name VARCHAR(255),                  -- 'src.services.tasks.sync_kalshi_markets'

    -- Schedule configuration
    schedule_type VARCHAR(50),                      -- 'crontab', 'interval'
    schedule_config JSONB,                          -- {"minute": "*/5", "hour": "*"}
    schedule_display VARCHAR(100),                  -- 'Every 5 minutes'

    -- Queue and routing
    queue VARCHAR(100) DEFAULT 'default',           -- 'market_data', 'predictions'

    -- Categorization
    category VARCHAR(100) NOT NULL,                 -- 'market_data', 'notifications', 'maintenance'
    description TEXT,

    -- State management
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    enabled_updated_at TIMESTAMP WITH TIME ZONE,
    enabled_updated_by VARCHAR(100),

    -- Execution settings
    timeout_seconds INTEGER DEFAULT 300,
    max_retries INTEGER DEFAULT 3,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT chk_automation_type CHECK (
        automation_type IN ('celery_beat', 'celery_task')
    )
);

-- Table 2: automation_executions
-- Detailed execution history for each run
CREATE TABLE IF NOT EXISTS automation_executions (
    id BIGSERIAL PRIMARY KEY,
    automation_id INTEGER NOT NULL REFERENCES automations(id) ON DELETE CASCADE,

    -- Celery task info
    celery_task_id VARCHAR(255),                    -- UUID from Celery

    -- Execution timing
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds NUMERIC(10, 3),

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'running',

    -- Results
    result JSONB,                                   -- Task return value
    error_message TEXT,
    error_traceback TEXT,

    -- Metrics
    records_processed INTEGER,

    -- Context
    triggered_by VARCHAR(100) DEFAULT 'scheduler',  -- 'scheduler', 'api', 'manual'
    worker_hostname VARCHAR(255),

    CONSTRAINT chk_execution_status CHECK (
        status IN ('pending', 'running', 'success', 'failed', 'revoked', 'timeout', 'skipped')
    )
);

-- Table 3: automation_state_log
-- Audit log for enable/disable actions
CREATE TABLE IF NOT EXISTS automation_state_log (
    id SERIAL PRIMARY KEY,
    automation_id INTEGER NOT NULL REFERENCES automations(id) ON DELETE CASCADE,

    previous_state BOOLEAN,
    new_state BOOLEAN NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    changed_by VARCHAR(100),
    reason TEXT,

    -- Track any running tasks that were affected
    affected_task_ids TEXT[]
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_automations_name ON automations(name);
CREATE INDEX IF NOT EXISTS idx_automations_category ON automations(category);
CREATE INDEX IF NOT EXISTS idx_automations_enabled ON automations(is_enabled);
CREATE INDEX IF NOT EXISTS idx_automations_type ON automations(automation_type);

CREATE INDEX IF NOT EXISTS idx_executions_automation_id ON automation_executions(automation_id);
CREATE INDEX IF NOT EXISTS idx_executions_started_at ON automation_executions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_executions_status ON automation_executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_celery_task_id ON automation_executions(celery_task_id);

-- Composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_executions_automation_status_time
    ON automation_executions(automation_id, status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_state_log_automation
    ON automation_state_log(automation_id, changed_at DESC);

-- ============================================================================
-- View: v_automation_status
-- Combines automation definition with latest execution info
-- ============================================================================

CREATE OR REPLACE VIEW v_automation_status AS
SELECT
    a.*,
    latest.id AS last_execution_id,
    latest.status AS last_run_status,
    latest.started_at AS last_run_at,
    latest.completed_at AS last_completed_at,
    latest.duration_seconds AS last_duration_seconds,
    latest.error_message AS last_error,
    latest.records_processed AS last_records_processed,

    -- Calculate next run (placeholder - computed by service)
    NULL::TIMESTAMP WITH TIME ZONE AS next_run_at,

    -- Execution stats (last 24 hours)
    stats.total_runs_24h,
    stats.successful_runs_24h,
    stats.failed_runs_24h,
    CASE
        WHEN stats.total_runs_24h > 0
        THEN ROUND((stats.successful_runs_24h::NUMERIC / stats.total_runs_24h) * 100, 1)
        ELSE NULL
    END AS success_rate_24h,
    stats.avg_duration_24h

FROM automations a

-- Latest execution
LEFT JOIN LATERAL (
    SELECT * FROM automation_executions
    WHERE automation_id = a.id
    ORDER BY started_at DESC
    LIMIT 1
) latest ON true

-- 24h stats
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) AS total_runs_24h,
        COUNT(*) FILTER (WHERE status = 'success') AS successful_runs_24h,
        COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs_24h,
        ROUND(AVG(duration_seconds)::NUMERIC, 2) AS avg_duration_24h
    FROM automation_executions
    WHERE automation_id = a.id
    AND started_at > NOW() - INTERVAL '24 hours'
) stats ON true;

-- ============================================================================
-- Seed Data: Initial Automations from celery_app.py
-- ============================================================================

INSERT INTO automations (name, display_name, automation_type, celery_task_name, schedule_type, schedule_config, schedule_display, queue, category, description, is_enabled)
VALUES
    -- Market Data Tasks
    ('sync-kalshi-markets', 'Sync Kalshi Markets', 'celery_beat',
     'src.services.tasks.sync_kalshi_markets', 'crontab',
     '{"minute": "*/5"}', 'Every 5 minutes',
     'market_data', 'market_data',
     'Sync Kalshi prediction markets for sports betting', TRUE),

    ('update-stock-prices', 'Update Stock Prices', 'celery_beat',
     'src.services.tasks.update_stock_prices', 'crontab',
     '{"minute": "*/1", "hour": "9-16", "day_of_week": "mon-fri"}', 'Every minute (market hours)',
     'market_data', 'market_data',
     'Update stock prices for watchlist during market hours', TRUE),

    ('sync-discord-messages', 'Sync Discord Messages', 'celery_beat',
     'src.services.tasks.sync_discord_messages', 'crontab',
     '{"minute": "*/5"}', 'Every 5 minutes',
     'market_data', 'market_data',
     'Sync Discord messages with premium alert prioritization', TRUE),

    ('update-earnings-calendar', 'Update Earnings Calendar', 'celery_beat',
     'src.services.tasks.update_earnings_calendar', 'crontab',
     '{"hour": "6", "minute": "0"}', 'Daily at 6 AM',
     'market_data', 'market_data',
     'Update earnings calendar for next 30 days', TRUE),

    -- Predictions Tasks
    ('generate-predictions', 'Generate AI Predictions', 'celery_beat',
     'src.services.tasks.generate_predictions', 'crontab',
     '{"minute": "*/15"}', 'Every 15 minutes',
     'predictions', 'predictions',
     'Generate AI predictions for upcoming sports games', TRUE),

    -- Notifications Tasks
    ('send-hourly-alerts', 'Send Hourly Alerts', 'celery_beat',
     'src.services.tasks.send_alerts', 'crontab',
     '{"minute": "0"}', 'Every hour',
     'notifications', 'notifications',
     'Send scheduled alerts for high-confidence predictions', TRUE),

    -- Maintenance Tasks
    ('cleanup-old-data', 'Cleanup Old Data', 'celery_beat',
     'src.services.tasks.cleanup_old_data', 'crontab',
     '{"hour": "2", "minute": "0"}', 'Daily at 2 AM',
     'maintenance', 'maintenance',
     'Delete old Discord messages, predictions, and cache entries (90 day retention)', TRUE),

    ('warm-caches', 'Warm Caches', 'celery_beat',
     'src.services.tasks.warm_caches', 'crontab',
     '{"minute": "*/30"}', 'Every 30 minutes',
     'maintenance', 'maintenance',
     'Pre-warm frequently accessed caches for Kalshi and NFL data', TRUE),

    -- RAG Tasks
    ('sync-xtrades-to-rag', 'Sync XTrades to RAG', 'celery_beat',
     'src.services.tasks.sync_xtrades_to_rag', 'crontab',
     '{"hour": "1", "minute": "0"}', 'Daily at 1 AM',
     'maintenance', 'rag',
     'Sync XTrades messages to RAG knowledge base', TRUE),

    ('sync-discord-to-rag', 'Sync Discord to RAG', 'celery_beat',
     'src.services.tasks.sync_discord_to_rag', 'crontab',
     '{"hour": "2", "minute": "0"}', 'Daily at 2 AM',
     'maintenance', 'rag',
     'Sync Discord messages to RAG knowledge base (7-day rolling window)', TRUE)

ON CONFLICT (name) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    automation_type = EXCLUDED.automation_type,
    celery_task_name = EXCLUDED.celery_task_name,
    schedule_type = EXCLUDED.schedule_type,
    schedule_config = EXCLUDED.schedule_config,
    schedule_display = EXCLUDED.schedule_display,
    queue = EXCLUDED.queue,
    category = EXCLUDED.category,
    description = EXCLUDED.description,
    updated_at = NOW();

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE automations IS 'Master registry of all automated tasks managed by the Developer Console';
COMMENT ON TABLE automation_executions IS 'Execution history and performance tracking for automations';
COMMENT ON TABLE automation_state_log IS 'Audit trail for enable/disable actions';
COMMENT ON VIEW v_automation_status IS 'Combined view of automation definitions with latest execution status and 24h statistics';
