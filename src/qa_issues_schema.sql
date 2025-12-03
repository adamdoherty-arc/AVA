-- ============================================================================
-- QA Issues Tracking System - Database Schema
-- ============================================================================
-- Purpose: Track QA check results, issues found, fixes applied, and system health
-- Database: magnus (PostgreSQL)
-- Created: 2025-11-26
-- ============================================================================
--
-- Features:
-- - QA run tracking with health scores
-- - Individual check results per run
-- - Issue tracking with severity and status workflow
-- - Fix tracking and resolution history
-- - Health score trending
-- - Hot spot analysis (files with frequent issues)
-- ============================================================================

-- ============================================================================
-- Table 1: qa_runs
-- ============================================================================
-- Track each QA cycle execution
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,  -- UUID or timestamp-based ID
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,

    -- Health metrics
    health_score NUMERIC(5,2),  -- 0.00 to 100.00
    total_checks INTEGER DEFAULT 0,
    passed_checks INTEGER DEFAULT 0,
    failed_checks INTEGER DEFAULT 0,
    warned_checks INTEGER DEFAULT 0,
    skipped_checks INTEGER DEFAULT 0,

    -- Issue counts
    critical_issues INTEGER DEFAULT 0,
    high_issues INTEGER DEFAULT 0,
    medium_issues INTEGER DEFAULT 0,
    low_issues INTEGER DEFAULT 0,

    -- Fix tracking
    auto_fixes_attempted INTEGER DEFAULT 0,
    auto_fixes_succeeded INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(50) DEFAULT 'running',  -- 'running', 'completed', 'failed', 'interrupted'
    error_message TEXT,

    -- Metadata
    triggered_by VARCHAR(100) DEFAULT 'scheduler',  -- 'scheduler', 'manual', 'hook'

    CONSTRAINT chk_qa_run_status CHECK (status IN ('running', 'completed', 'failed', 'interrupted'))
);

-- Table comments
COMMENT ON TABLE qa_runs IS 'Track each QA cycle execution with health metrics and issue counts';
COMMENT ON COLUMN qa_runs.run_id IS 'Unique identifier for the QA run (UUID or timestamp-based)';
COMMENT ON COLUMN qa_runs.health_score IS 'Overall health score from 0.00 to 100.00';
COMMENT ON COLUMN qa_runs.triggered_by IS 'What triggered this run: scheduler, manual, or hook';

-- ============================================================================
-- Table 2: qa_check_results
-- ============================================================================
-- Individual check results from each QA run
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_check_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES qa_runs(id) ON DELETE CASCADE,

    -- Check identification
    module_name VARCHAR(200) NOT NULL,  -- e.g., 'api_endpoints', 'import_health'
    check_name VARCHAR(200) NOT NULL,   -- e.g., 'server_available', 'circular_imports'

    -- Result
    status VARCHAR(50) NOT NULL,  -- 'passed', 'failed', 'warned', 'skipped', 'error'
    message TEXT,
    details JSONB,  -- Structured details about the check result

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    duration_ms INTEGER,

    -- Fix info
    auto_fixable BOOLEAN DEFAULT FALSE,
    fix_attempted BOOLEAN DEFAULT FALSE,
    fix_succeeded BOOLEAN DEFAULT FALSE,
    fix_message TEXT,

    CONSTRAINT chk_result_status CHECK (status IN ('passed', 'failed', 'warned', 'skipped', 'error'))
);

-- Table comments
COMMENT ON TABLE qa_check_results IS 'Individual check results from each QA cycle';
COMMENT ON COLUMN qa_check_results.module_name IS 'QA check module (e.g., api_endpoints, import_health)';
COMMENT ON COLUMN qa_check_results.check_name IS 'Specific check within module (e.g., server_available)';
COMMENT ON COLUMN qa_check_results.details IS 'JSON object with structured check details';

-- ============================================================================
-- Table 3: qa_issues
-- ============================================================================
-- Track individual issues found across QA runs
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_issues (
    id SERIAL PRIMARY KEY,
    issue_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA256 hash for deduplication

    -- Issue identification
    module_name VARCHAR(200) NOT NULL,
    check_name VARCHAR(200) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,

    -- Classification
    severity VARCHAR(20) NOT NULL,  -- 'critical', 'high', 'medium', 'low'
    category VARCHAR(100),  -- 'api', 'import', 'security', 'performance', etc.

    -- Status tracking
    status VARCHAR(50) DEFAULT 'open',  -- 'open', 'fixing', 'fixed', 'ignored', 'wont_fix'

    -- Occurrence tracking
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    first_seen_run_id INTEGER REFERENCES qa_runs(id) ON DELETE SET NULL,
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_run_id INTEGER REFERENCES qa_runs(id) ON DELETE SET NULL,
    occurrence_count INTEGER DEFAULT 1,

    -- Resolution info
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),  -- 'auto_fix', 'manual', 'agent_name'
    resolution_notes TEXT,

    -- Files affected
    files_affected TEXT[],
    primary_file VARCHAR(500),

    -- Metadata
    details JSONB,  -- Additional structured data
    tags TEXT[],

    CONSTRAINT chk_issue_severity CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    CONSTRAINT chk_issue_status CHECK (status IN ('open', 'fixing', 'fixed', 'ignored', 'wont_fix'))
);

-- Table comments
COMMENT ON TABLE qa_issues IS 'Track individual issues found across QA runs with deduplication';
COMMENT ON COLUMN qa_issues.issue_hash IS 'SHA256 hash of module+check+title for deduplication';
COMMENT ON COLUMN qa_issues.occurrence_count IS 'Number of times this issue has been seen';
COMMENT ON COLUMN qa_issues.status IS 'Current status: open, fixing, fixed, ignored, wont_fix';

-- ============================================================================
-- Table 4: qa_issue_occurrences
-- ============================================================================
-- Link table tracking when issues occur in which runs
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_issue_occurrences (
    id SERIAL PRIMARY KEY,
    issue_id INTEGER NOT NULL REFERENCES qa_issues(id) ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES qa_runs(id) ON DELETE CASCADE,
    check_result_id INTEGER REFERENCES qa_check_results(id) ON DELETE SET NULL,
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(issue_id, run_id)
);

-- Table comments
COMMENT ON TABLE qa_issue_occurrences IS 'Link table tracking issue occurrences per run';

-- ============================================================================
-- Table 5: qa_fixes
-- ============================================================================
-- Track all fix attempts (auto and manual)
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_fixes (
    id SERIAL PRIMARY KEY,
    issue_id INTEGER NOT NULL REFERENCES qa_issues(id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES qa_runs(id) ON DELETE SET NULL,

    -- Fix details
    fix_type VARCHAR(50) NOT NULL,  -- 'auto', 'manual', 'agent'
    fixer_name VARCHAR(100),  -- Agent name or 'auto_fix' or 'user'

    -- Timing
    attempted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Result
    success BOOLEAN NOT NULL,
    message TEXT,
    error_details TEXT,

    -- Changes made
    files_modified TEXT[],
    lines_added INTEGER DEFAULT 0,
    lines_removed INTEGER DEFAULT 0,
    git_commit_hash VARCHAR(40),

    -- Details
    details JSONB,

    CONSTRAINT chk_fix_type CHECK (fix_type IN ('auto', 'manual', 'agent'))
);

-- Table comments
COMMENT ON TABLE qa_fixes IS 'Track all fix attempts for QA issues';
COMMENT ON COLUMN qa_fixes.fix_type IS 'Type of fix: auto, manual, or agent';

-- ============================================================================
-- Table 6: qa_health_history
-- ============================================================================
-- Track health score over time for trending
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_health_history (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    run_id INTEGER REFERENCES qa_runs(id) ON DELETE SET NULL,

    -- Health metrics
    health_score NUMERIC(5,2) NOT NULL,

    -- Component scores (for breakdown)
    api_health NUMERIC(5,2),
    import_health NUMERIC(5,2),
    security_health NUMERIC(5,2),
    performance_health NUMERIC(5,2),
    code_quality_health NUMERIC(5,2),

    -- Issue counts at this point
    open_critical INTEGER DEFAULT 0,
    open_high INTEGER DEFAULT 0,
    open_medium INTEGER DEFAULT 0,
    open_low INTEGER DEFAULT 0,
    total_open_issues INTEGER DEFAULT 0
);

-- Table comments
COMMENT ON TABLE qa_health_history IS 'Track health score trends over time';

-- ============================================================================
-- Table 7: qa_hot_spots
-- ============================================================================
-- Track files that frequently have issues
-- ============================================================================

CREATE TABLE IF NOT EXISTS qa_hot_spots (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(500) UNIQUE NOT NULL,

    -- Issue tracking
    total_issues INTEGER DEFAULT 0,
    critical_issues INTEGER DEFAULT 0,
    high_issues INTEGER DEFAULT 0,
    medium_issues INTEGER DEFAULT 0,
    low_issues INTEGER DEFAULT 0,

    -- Severity score (weighted)
    severity_score INTEGER DEFAULT 0,  -- critical*10 + high*5 + medium*2 + low*1

    -- Patterns
    common_patterns TEXT[],  -- Array of common issue patterns
    categories TEXT[],  -- Array of issue categories

    -- Timing
    first_issue_at TIMESTAMP WITH TIME ZONE,
    last_issue_at TIMESTAMP WITH TIME ZONE,

    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table comments
COMMENT ON TABLE qa_hot_spots IS 'Track files that frequently have issues';
COMMENT ON COLUMN qa_hot_spots.severity_score IS 'Weighted score: critical*10 + high*5 + medium*2 + low*1';

-- ============================================================================
-- INDEXES - Optimized for common query patterns
-- ============================================================================

-- QA runs indexes
CREATE INDEX IF NOT EXISTS idx_qa_runs_started_at ON qa_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_runs_status ON qa_runs(status);
CREATE INDEX IF NOT EXISTS idx_qa_runs_health_score ON qa_runs(health_score);

-- Check results indexes
CREATE INDEX IF NOT EXISTS idx_qa_check_results_run_id ON qa_check_results(run_id);
CREATE INDEX IF NOT EXISTS idx_qa_check_results_module ON qa_check_results(module_name);
CREATE INDEX IF NOT EXISTS idx_qa_check_results_status ON qa_check_results(status);
CREATE INDEX IF NOT EXISTS idx_qa_check_results_auto_fixable ON qa_check_results(auto_fixable) WHERE auto_fixable = TRUE;

-- Issues indexes
CREATE INDEX IF NOT EXISTS idx_qa_issues_status ON qa_issues(status) WHERE status NOT IN ('fixed', 'ignored', 'wont_fix');
CREATE INDEX IF NOT EXISTS idx_qa_issues_severity ON qa_issues(severity);
CREATE INDEX IF NOT EXISTS idx_qa_issues_module ON qa_issues(module_name);
CREATE INDEX IF NOT EXISTS idx_qa_issues_category ON qa_issues(category);
CREATE INDEX IF NOT EXISTS idx_qa_issues_last_seen ON qa_issues(last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_issues_first_seen ON qa_issues(first_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_issues_occurrence ON qa_issues(occurrence_count DESC);

-- Issue occurrences indexes
CREATE INDEX IF NOT EXISTS idx_qa_issue_occurrences_issue ON qa_issue_occurrences(issue_id);
CREATE INDEX IF NOT EXISTS idx_qa_issue_occurrences_run ON qa_issue_occurrences(run_id);

-- Fixes indexes
CREATE INDEX IF NOT EXISTS idx_qa_fixes_issue ON qa_fixes(issue_id);
CREATE INDEX IF NOT EXISTS idx_qa_fixes_success ON qa_fixes(success);
CREATE INDEX IF NOT EXISTS idx_qa_fixes_attempted ON qa_fixes(attempted_at DESC);

-- Health history indexes
CREATE INDEX IF NOT EXISTS idx_qa_health_history_recorded ON qa_health_history(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_qa_health_history_score ON qa_health_history(health_score);

-- Hot spots indexes
CREATE INDEX IF NOT EXISTS idx_qa_hot_spots_severity ON qa_hot_spots(severity_score DESC);
CREATE INDEX IF NOT EXISTS idx_qa_hot_spots_total ON qa_hot_spots(total_issues DESC);

-- ============================================================================
-- VIEWS - Common query patterns for dashboard
-- ============================================================================

-- View: Current open issues summary
CREATE OR REPLACE VIEW v_qa_open_issues AS
SELECT
    i.id,
    i.issue_hash,
    i.module_name,
    i.check_name,
    i.title,
    i.severity,
    i.category,
    i.status,
    i.first_seen_at,
    i.last_seen_at,
    i.occurrence_count,
    i.primary_file,
    ARRAY_LENGTH(i.files_affected, 1) AS file_count,
    (SELECT COUNT(*) FROM qa_fixes f WHERE f.issue_id = i.id AND f.success = FALSE) AS failed_fix_attempts
FROM qa_issues i
WHERE i.status IN ('open', 'fixing')
ORDER BY
    CASE i.severity
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    i.occurrence_count DESC;

-- View: Recent QA runs with summary
CREATE OR REPLACE VIEW v_qa_recent_runs AS
SELECT
    r.id,
    r.run_id,
    r.started_at,
    r.completed_at,
    r.duration_seconds,
    r.status,
    r.health_score,
    r.total_checks,
    r.passed_checks,
    r.failed_checks,
    r.warned_checks,
    (r.critical_issues + r.high_issues) AS high_severity_issues,
    r.auto_fixes_succeeded,
    r.triggered_by,
    CASE
        WHEN r.health_score >= 90 THEN 'excellent'
        WHEN r.health_score >= 75 THEN 'good'
        WHEN r.health_score >= 50 THEN 'fair'
        WHEN r.health_score >= 25 THEN 'poor'
        ELSE 'critical'
    END AS health_status
FROM qa_runs r
ORDER BY r.started_at DESC;

-- View: Issue trends by category
CREATE OR REPLACE VIEW v_qa_issue_trends AS
SELECT
    category,
    COUNT(*) FILTER (WHERE status IN ('open', 'fixing')) AS open_count,
    COUNT(*) FILTER (WHERE status = 'fixed') AS fixed_count,
    COUNT(*) FILTER (WHERE status IN ('ignored', 'wont_fix')) AS dismissed_count,
    COUNT(*) AS total_count,
    AVG(occurrence_count)::INTEGER AS avg_occurrences,
    MAX(last_seen_at) AS last_active
FROM qa_issues
GROUP BY category
ORDER BY open_count DESC;

-- View: Top hot spots
CREATE OR REPLACE VIEW v_qa_top_hot_spots AS
SELECT
    h.file_path,
    h.total_issues,
    h.severity_score,
    h.critical_issues,
    h.high_issues,
    h.medium_issues,
    h.low_issues,
    h.common_patterns,
    h.last_issue_at,
    EXTRACT(DAY FROM NOW() - h.last_issue_at) AS days_since_last_issue
FROM qa_hot_spots h
WHERE h.total_issues > 0
ORDER BY h.severity_score DESC, h.total_issues DESC
LIMIT 20;

-- View: Health score trend (last 7 days)
CREATE OR REPLACE VIEW v_qa_health_trend AS
SELECT
    DATE_TRUNC('hour', recorded_at) AS hour,
    AVG(health_score)::NUMERIC(5,2) AS avg_health,
    MIN(health_score) AS min_health,
    MAX(health_score) AS max_health,
    AVG(total_open_issues)::INTEGER AS avg_open_issues
FROM qa_health_history
WHERE recorded_at > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', recorded_at)
ORDER BY hour DESC;

-- ============================================================================
-- FUNCTIONS - Helper functions
-- ============================================================================

-- Function: Calculate issue hash for deduplication
CREATE OR REPLACE FUNCTION calculate_issue_hash(
    p_module_name VARCHAR,
    p_check_name VARCHAR,
    p_title VARCHAR
)
RETURNS VARCHAR AS $$
BEGIN
    RETURN encode(
        sha256(
            (p_module_name || '::' || p_check_name || '::' || p_title)::bytea
        ),
        'hex'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: Update hot spot for a file
CREATE OR REPLACE FUNCTION update_hot_spot(
    p_file_path VARCHAR,
    p_severity VARCHAR,
    p_pattern VARCHAR DEFAULT NULL,
    p_category VARCHAR DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO qa_hot_spots (file_path, total_issues, first_issue_at, last_issue_at)
    VALUES (p_file_path, 1, NOW(), NOW())
    ON CONFLICT (file_path) DO UPDATE SET
        total_issues = qa_hot_spots.total_issues + 1,
        critical_issues = qa_hot_spots.critical_issues + CASE WHEN p_severity = 'critical' THEN 1 ELSE 0 END,
        high_issues = qa_hot_spots.high_issues + CASE WHEN p_severity = 'high' THEN 1 ELSE 0 END,
        medium_issues = qa_hot_spots.medium_issues + CASE WHEN p_severity = 'medium' THEN 1 ELSE 0 END,
        low_issues = qa_hot_spots.low_issues + CASE WHEN p_severity = 'low' THEN 1 ELSE 0 END,
        severity_score = qa_hot_spots.severity_score + CASE p_severity
            WHEN 'critical' THEN 10
            WHEN 'high' THEN 5
            WHEN 'medium' THEN 2
            WHEN 'low' THEN 1
            ELSE 0
        END,
        common_patterns = CASE
            WHEN p_pattern IS NOT NULL AND NOT (p_pattern = ANY(qa_hot_spots.common_patterns))
            THEN array_append(qa_hot_spots.common_patterns, p_pattern)
            ELSE qa_hot_spots.common_patterns
        END,
        categories = CASE
            WHEN p_category IS NOT NULL AND NOT (p_category = ANY(qa_hot_spots.categories))
            THEN array_append(qa_hot_spots.categories, p_category)
            ELSE qa_hot_spots.categories
        END,
        last_issue_at = NOW(),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Function: Get or create issue (with deduplication)
CREATE OR REPLACE FUNCTION upsert_qa_issue(
    p_module_name VARCHAR,
    p_check_name VARCHAR,
    p_title VARCHAR,
    p_description TEXT,
    p_severity VARCHAR,
    p_category VARCHAR,
    p_run_id INTEGER,
    p_files_affected TEXT[] DEFAULT NULL,
    p_primary_file VARCHAR DEFAULT NULL,
    p_details JSONB DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_issue_hash VARCHAR;
    v_issue_id INTEGER;
BEGIN
    -- Calculate hash
    v_issue_hash := calculate_issue_hash(p_module_name, p_check_name, p_title);

    -- Try to insert or update
    INSERT INTO qa_issues (
        issue_hash, module_name, check_name, title, description,
        severity, category, first_seen_run_id, last_seen_run_id,
        files_affected, primary_file, details
    ) VALUES (
        v_issue_hash, p_module_name, p_check_name, p_title, p_description,
        p_severity, p_category, p_run_id, p_run_id,
        p_files_affected, p_primary_file, p_details
    )
    ON CONFLICT (issue_hash) DO UPDATE SET
        last_seen_at = NOW(),
        last_seen_run_id = p_run_id,
        occurrence_count = qa_issues.occurrence_count + 1,
        -- Re-open if was fixed but reappeared
        status = CASE
            WHEN qa_issues.status = 'fixed' THEN 'open'
            ELSE qa_issues.status
        END
    RETURNING id INTO v_issue_id;

    RETURN v_issue_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS - Automatic updates
-- ============================================================================

-- Trigger: Update health history after run completion
CREATE OR REPLACE FUNCTION update_health_history_on_run_complete()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND OLD.status = 'running' THEN
        INSERT INTO qa_health_history (
            run_id,
            health_score,
            open_critical,
            open_high,
            open_medium,
            open_low,
            total_open_issues
        )
        SELECT
            NEW.id,
            NEW.health_score,
            COUNT(*) FILTER (WHERE severity = 'critical' AND status IN ('open', 'fixing')),
            COUNT(*) FILTER (WHERE severity = 'high' AND status IN ('open', 'fixing')),
            COUNT(*) FILTER (WHERE severity = 'medium' AND status IN ('open', 'fixing')),
            COUNT(*) FILTER (WHERE severity = 'low' AND status IN ('open', 'fixing')),
            COUNT(*) FILTER (WHERE status IN ('open', 'fixing'))
        FROM qa_issues;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_health_history
    AFTER UPDATE ON qa_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_health_history_on_run_complete();

-- ============================================================================
-- SCHEMA VERIFICATION QUERY
-- ============================================================================

SELECT
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) AS column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
AND table_name LIKE 'qa_%'
ORDER BY table_name;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
