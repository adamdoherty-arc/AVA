-- ============================================================================
-- AVA Technical Specifications System
-- Migration: 006_technical_specs_system.sql
-- Version: 6.0.0
-- Date: 2025-11-28
-- Description: Comprehensive schema for storing detailed technical specifications
--              of all AVA platform features, enabling AI-powered discovery,
--              semantic search, dependency tracking, and efficiency analysis.
-- NOTE: pgvector is optional - embedding column uses FLOAT[] fallback if unavailable
-- ============================================================================

BEGIN;

-- Try to enable pgvector, but don't fail if unavailable
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector extension not available - using float array fallback for embeddings';
END $$;

-- ============================================================================
-- PART 1: Core Enumeration Types
-- ============================================================================

-- Feature Categories (based on actual codebase structure)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'spec_category') THEN
        CREATE TYPE spec_category AS ENUM (
            'core',                    -- Core infrastructure
            'agents_trading',          -- Trading agents
            'agents_analysis',         -- Analysis agents
            'agents_sports',           -- Sports betting agents
            'agents_monitoring',       -- Monitoring agents
            'agents_research',         -- Research agents
            'agents_management',       -- Management agents
            'agents_code',             -- Code/development agents
            'backend_services',        -- Backend services
            'backend_routers',         -- API routers
            'frontend_pages',          -- Frontend pages
            'frontend_components',     -- Frontend components
            'integrations',            -- External integrations
            'database',                -- Database schemas/migrations
            'infrastructure'           -- Infrastructure components
        );
    END IF;
END $$;

-- Dependency relationship types
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'dependency_type') THEN
        CREATE TYPE dependency_type AS ENUM (
            'imports',                 -- Code imports
            'calls',                   -- Function/API calls
            'data_flow',               -- Data flows between components
            'inherits',                -- Class inheritance
            'implements',              -- Interface implementation
            'uses_service',            -- Uses a backend service
            'uses_database',           -- Uses database table
            'requires_config',         -- Requires configuration
            'triggers',                -- Event triggers
            'composes'                 -- Component composition
        );
    END IF;
END $$;

-- Issue severity levels
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'issue_severity') THEN
        CREATE TYPE issue_severity AS ENUM (
            'critical',
            'high',
            'medium',
            'low',
            'info'
        );
    END IF;
END $$;

-- Enhancement priority
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'enhancement_priority') THEN
        CREATE TYPE enhancement_priority AS ENUM (
            'p0_critical',
            'p1_high',
            'p2_medium',
            'p3_low',
            'p4_backlog'
        );
    END IF;
END $$;


-- ============================================================================
-- PART 2: Main Specifications Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_feature_specs (
    id SERIAL PRIMARY KEY,

    -- Identification
    feature_id VARCHAR(100) UNIQUE NOT NULL,    -- e.g., 'portfolio_agent', 'premium_scanner_page'
    feature_name VARCHAR(255) NOT NULL,         -- Human-readable name
    category spec_category NOT NULL,
    subcategory VARCHAR(100),                   -- Optional subcategory

    -- Core Description
    purpose TEXT NOT NULL,                      -- What the feature does
    description TEXT,                           -- Detailed description
    key_responsibilities TEXT[],                -- Array of main responsibilities

    -- Version Tracking
    version VARCHAR(20) DEFAULT '1.0.0',
    is_current BOOLEAN DEFAULT TRUE,            -- Latest version flag

    -- Status
    status VARCHAR(50) DEFAULT 'active',        -- 'active', 'deprecated', 'planned', 'experimental'
    maturity_level VARCHAR(50) DEFAULT 'stable', -- 'prototype', 'alpha', 'beta', 'stable', 'mature'

    -- Technical Details (JSONB for flexibility)
    technical_details JSONB DEFAULT '{}'::jsonb,

    -- Vector Embedding for Semantic Search (1536 dims for OpenAI ada-002)
    -- Uses FLOAT[] as fallback if pgvector not available
    embedding FLOAT[],

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analyzed_at TIMESTAMP WITH TIME ZONE,       -- Last AI analysis timestamp

    CONSTRAINT chk_feature_status CHECK (status IN ('active', 'deprecated', 'planned', 'experimental', 'removed'))
);

-- Indexes for ava_feature_specs
CREATE INDEX IF NOT EXISTS idx_feature_specs_category ON ava_feature_specs(category);
CREATE INDEX IF NOT EXISTS idx_feature_specs_subcategory ON ava_feature_specs(category, subcategory);
CREATE INDEX IF NOT EXISTS idx_feature_specs_status ON ava_feature_specs(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_feature_specs_feature_id ON ava_feature_specs(feature_id);
CREATE INDEX IF NOT EXISTS idx_feature_specs_current ON ava_feature_specs(is_current) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_feature_specs_technical ON ava_feature_specs USING GIN(technical_details);

-- Vector similarity search index (requires pgvector)
-- Will be created when pgvector is available:
-- CREATE INDEX IF NOT EXISTS idx_feature_specs_embedding ON ava_feature_specs
--     USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);

-- GIN index on embedding for basic array operations
CREATE INDEX IF NOT EXISTS idx_feature_specs_embedding ON ava_feature_specs USING GIN(embedding);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_feature_specs_fts ON ava_feature_specs
    USING GIN(to_tsvector('english', COALESCE(feature_name, '') || ' ' || COALESCE(purpose, '') || ' ' || COALESCE(description, '')));

COMMENT ON TABLE ava_feature_specs IS 'Master table for all AVA platform feature specifications';
COMMENT ON COLUMN ava_feature_specs.feature_id IS 'Unique identifier matching code naming conventions';
COMMENT ON COLUMN ava_feature_specs.embedding IS 'Vector embedding for semantic similarity search (1536 dimensions)';


-- ============================================================================
-- PART 3: Source Files Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_source_files (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- File Information
    file_path VARCHAR(500) NOT NULL,           -- Relative path from project root
    file_type VARCHAR(50) NOT NULL,            -- 'python', 'typescript', 'sql', 'json', etc.
    is_primary BOOLEAN DEFAULT FALSE,          -- Primary implementation file

    -- Line Number Tracking
    start_line INTEGER,                        -- Start line of relevant code
    end_line INTEGER,                          -- End line of relevant code

    -- File Metadata
    file_purpose VARCHAR(500),                 -- What this file does for the feature
    key_exports TEXT[],                        -- Classes, functions, constants exported

    -- Analysis Data
    loc INTEGER,                               -- Lines of code
    complexity_score INTEGER,                  -- Cyclomatic complexity (if analyzed)
    last_modified TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, file_path)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_source_files_spec ON ava_spec_source_files(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_source_files_path ON ava_spec_source_files(file_path);
CREATE INDEX IF NOT EXISTS idx_spec_source_files_type ON ava_spec_source_files(file_type);
CREATE INDEX IF NOT EXISTS idx_spec_source_files_primary ON ava_spec_source_files(spec_id, is_primary) WHERE is_primary = TRUE;

COMMENT ON TABLE ava_spec_source_files IS 'Source files associated with each feature specification';


-- ============================================================================
-- PART 4: Dependencies Graph
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_dependencies (
    id SERIAL PRIMARY KEY,

    -- Dependency Relationship
    source_spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,
    target_spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Relationship Type
    dependency_type dependency_type NOT NULL,

    -- Details
    description TEXT,                          -- Description of the relationship
    is_critical BOOLEAN DEFAULT FALSE,         -- Critical dependency (breaks if removed)
    strength INTEGER DEFAULT 5,                -- 1-10 strength of coupling

    -- Analysis
    bidirectional BOOLEAN DEFAULT FALSE,       -- Two-way dependency

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(source_spec_id, target_spec_id, dependency_type),
    CONSTRAINT chk_no_self_dependency CHECK (source_spec_id != target_spec_id),
    CONSTRAINT chk_strength CHECK (strength >= 1 AND strength <= 10)
);

-- Indexes for graph traversal
CREATE INDEX IF NOT EXISTS idx_spec_deps_source ON ava_spec_dependencies(source_spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_deps_target ON ava_spec_dependencies(target_spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_deps_type ON ava_spec_dependencies(dependency_type);
CREATE INDEX IF NOT EXISTS idx_spec_deps_critical ON ava_spec_dependencies(is_critical) WHERE is_critical = TRUE;

COMMENT ON TABLE ava_spec_dependencies IS 'Dependency graph between features for impact analysis';


-- ============================================================================
-- PART 5: API Endpoints
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_api_endpoints (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Endpoint Details
    method VARCHAR(10) NOT NULL,               -- GET, POST, PUT, DELETE, PATCH, WS
    path VARCHAR(500) NOT NULL,                -- /api/v1/portfolio
    router_name VARCHAR(100),                  -- FastAPI router name

    -- Documentation
    summary VARCHAR(500),
    description TEXT,

    -- Request/Response
    request_model VARCHAR(200),                -- Pydantic model name
    response_model VARCHAR(200),               -- Pydantic model name
    query_params JSONB DEFAULT '[]'::jsonb,    -- Query parameters
    path_params JSONB DEFAULT '[]'::jsonb,     -- Path parameters

    -- Authentication
    requires_auth BOOLEAN DEFAULT TRUE,
    required_permissions TEXT[],

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    deprecated_at TIMESTAMP WITH TIME ZONE,

    -- Performance
    avg_response_time_ms INTEGER,
    p95_response_time_ms INTEGER,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, method, path)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_endpoints_spec ON ava_spec_api_endpoints(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_endpoints_path ON ava_spec_api_endpoints(path);
CREATE INDEX IF NOT EXISTS idx_spec_endpoints_method ON ava_spec_api_endpoints(method, path);
CREATE INDEX IF NOT EXISTS idx_spec_endpoints_router ON ava_spec_api_endpoints(router_name);

COMMENT ON TABLE ava_spec_api_endpoints IS 'API endpoints exposed by each feature';


-- ============================================================================
-- PART 6: Database Tables Used
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_database_tables (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Table Information
    table_name VARCHAR(200) NOT NULL,
    schema_name VARCHAR(100) DEFAULT 'public',

    -- Usage Type
    usage_type VARCHAR(50) NOT NULL,           -- 'read', 'write', 'read_write', 'owns'
    is_owner BOOLEAN DEFAULT FALSE,            -- Feature owns this table

    -- Details
    columns_used TEXT[],                       -- Specific columns accessed
    access_patterns TEXT[],                    -- Common access patterns

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, table_name, schema_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_db_tables_spec ON ava_spec_database_tables(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_db_tables_table ON ava_spec_database_tables(table_name);
CREATE INDEX IF NOT EXISTS idx_spec_db_tables_owner ON ava_spec_database_tables(is_owner) WHERE is_owner = TRUE;

COMMENT ON TABLE ava_spec_database_tables IS 'Database tables used by each feature';


-- ============================================================================
-- PART 7: Efficiency Ratings (8 Dimensions)
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_efficiency_ratings (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Overall Rating
    overall_rating DECIMAL(3,1) NOT NULL,      -- 1.0 to 10.0

    -- Dimensional Ratings (7 dimensions)
    code_completeness DECIMAL(3,1),
    test_coverage DECIMAL(3,1),
    performance DECIMAL(3,1),
    error_handling DECIMAL(3,1),
    documentation DECIMAL(3,1),
    maintainability DECIMAL(3,1),
    dependencies DECIMAL(3,1),

    -- Priority Level (computed)
    priority_level VARCHAR(20),                -- 'critical', 'high', 'medium', 'low'

    -- Detailed Metrics
    metrics JSONB DEFAULT '{}'::jsonb,         -- Detailed metric breakdown

    -- Analysis
    analysis_summary TEXT,                     -- AI-generated summary
    strengths TEXT[],                          -- List of strengths
    weaknesses TEXT[],                         -- List of weaknesses
    quick_wins TEXT[],                         -- Easy improvements

    -- Assessment Info
    assessed_by VARCHAR(100) DEFAULT 'ai',     -- 'ai', 'manual', 'hybrid'
    assessment_version VARCHAR(20),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id),
    CONSTRAINT chk_rating_range CHECK (
        overall_rating >= 1.0 AND overall_rating <= 10.0 AND
        (code_completeness IS NULL OR (code_completeness >= 1.0 AND code_completeness <= 10.0)) AND
        (test_coverage IS NULL OR (test_coverage >= 1.0 AND test_coverage <= 10.0)) AND
        (performance IS NULL OR (performance >= 1.0 AND performance <= 10.0)) AND
        (error_handling IS NULL OR (error_handling >= 1.0 AND error_handling <= 10.0)) AND
        (documentation IS NULL OR (documentation >= 1.0 AND documentation <= 10.0)) AND
        (maintainability IS NULL OR (maintainability >= 1.0 AND maintainability <= 10.0)) AND
        (dependencies IS NULL OR (dependencies >= 1.0 AND dependencies <= 10.0))
    )
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_efficiency_spec ON ava_spec_efficiency_ratings(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_efficiency_overall ON ava_spec_efficiency_ratings(overall_rating DESC);
CREATE INDEX IF NOT EXISTS idx_spec_efficiency_low ON ava_spec_efficiency_ratings(overall_rating) WHERE overall_rating < 7.0;
CREATE INDEX IF NOT EXISTS idx_spec_efficiency_priority ON ava_spec_efficiency_ratings(priority_level);
CREATE INDEX IF NOT EXISTS idx_spec_efficiency_metrics ON ava_spec_efficiency_ratings USING GIN(metrics);

COMMENT ON TABLE ava_spec_efficiency_ratings IS 'Efficiency ratings and assessments for each feature (7 dimensions)';


-- ============================================================================
-- PART 8: Known Issues
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_known_issues (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Issue Details
    issue_title VARCHAR(500) NOT NULL,
    issue_description TEXT NOT NULL,
    severity issue_severity NOT NULL,

    -- Classification
    issue_type VARCHAR(100),                   -- 'bug', 'tech_debt', 'security', 'performance', 'design'
    affected_files TEXT[],                     -- Files affected

    -- Resolution
    status VARCHAR(50) DEFAULT 'open',         -- 'open', 'in_progress', 'resolved', 'wont_fix'
    resolution TEXT,
    resolved_at TIMESTAMP WITH TIME ZONE,

    -- External References
    github_issue_url VARCHAR(500),
    related_commits TEXT[],

    -- Timestamps
    reported_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_issues_spec ON ava_spec_known_issues(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_issues_severity ON ava_spec_known_issues(severity);
CREATE INDEX IF NOT EXISTS idx_spec_issues_status ON ava_spec_known_issues(status) WHERE status != 'resolved';
CREATE INDEX IF NOT EXISTS idx_spec_issues_type ON ava_spec_known_issues(issue_type);

COMMENT ON TABLE ava_spec_known_issues IS 'Known issues and bugs for each feature';


-- ============================================================================
-- PART 9: Enhancement Opportunities
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_enhancements (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Enhancement Details
    enhancement_title VARCHAR(500) NOT NULL,
    enhancement_description TEXT NOT NULL,
    priority enhancement_priority NOT NULL,

    -- Impact Assessment
    estimated_effort VARCHAR(50),              -- 'trivial', 'small', 'medium', 'large', 'epic'
    expected_impact VARCHAR(50),               -- 'low', 'medium', 'high', 'transformative'
    affected_areas TEXT[],                     -- Areas that would be affected

    -- Implementation
    implementation_notes TEXT,
    prerequisite_enhancements INTEGER[],       -- IDs of prerequisites

    -- Status
    status VARCHAR(50) DEFAULT 'proposed',     -- 'proposed', 'approved', 'in_progress', 'completed', 'rejected'
    completed_at TIMESTAMP WITH TIME ZONE,

    -- AI Analysis
    ai_confidence DECIMAL(3,2),                -- AI confidence in this enhancement
    reasoning TEXT,

    -- Timestamps
    proposed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_enhancements_spec ON ava_spec_enhancements(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_enhancements_priority ON ava_spec_enhancements(priority);
CREATE INDEX IF NOT EXISTS idx_spec_enhancements_status ON ava_spec_enhancements(status);
CREATE INDEX IF NOT EXISTS idx_spec_enhancements_open ON ava_spec_enhancements(priority, status)
    WHERE status IN ('proposed', 'approved');

COMMENT ON TABLE ava_spec_enhancements IS 'Enhancement opportunities identified for each feature';


-- ============================================================================
-- PART 10: Integration Points
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_integrations (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Integration Details
    integration_name VARCHAR(200) NOT NULL,    -- 'Robinhood API', 'Redis Cache', 'OpenAI'
    integration_type VARCHAR(100) NOT NULL,    -- 'api', 'database', 'cache', 'message_queue', 'file_system'

    -- Configuration
    config_requirements JSONB DEFAULT '[]'::jsonb,  -- Required configuration
    env_variables TEXT[],                      -- Environment variables needed

    -- Connection Details
    endpoint_pattern VARCHAR(500),             -- URL pattern or connection string pattern
    auth_type VARCHAR(100),                    -- 'api_key', 'oauth', 'basic', 'none'

    -- Error Handling
    retry_strategy VARCHAR(200),
    fallback_behavior TEXT,
    circuit_breaker_enabled BOOLEAN DEFAULT FALSE,

    -- Health
    health_check_endpoint VARCHAR(500),
    is_critical BOOLEAN DEFAULT FALSE,         -- Feature fails without this

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, integration_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_integrations_spec ON ava_spec_integrations(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_integrations_name ON ava_spec_integrations(integration_name);
CREATE INDEX IF NOT EXISTS idx_spec_integrations_type ON ava_spec_integrations(integration_type);
CREATE INDEX IF NOT EXISTS idx_spec_integrations_critical ON ava_spec_integrations(is_critical) WHERE is_critical = TRUE;

COMMENT ON TABLE ava_spec_integrations IS 'External integration points for each feature';


-- ============================================================================
-- PART 11: Performance Metrics
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_performance_metrics (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Metric Identification
    metric_name VARCHAR(200) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,         -- 'latency', 'throughput', 'memory', 'cpu', 'error_rate'

    -- Target Values
    target_value DECIMAL(15,4),
    target_unit VARCHAR(50),                   -- 'ms', 'req/s', 'MB', '%'

    -- Current Measurements
    current_value DECIMAL(15,4),
    current_p50 DECIMAL(15,4),
    current_p95 DECIMAL(15,4),
    current_p99 DECIMAL(15,4),

    -- Status
    meets_target BOOLEAN,

    -- Historical (JSONB for flexibility)
    historical_data JSONB DEFAULT '[]'::jsonb,

    -- Timestamps
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, metric_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_perf_spec ON ava_spec_performance_metrics(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_perf_type ON ava_spec_performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_spec_perf_misses ON ava_spec_performance_metrics(meets_target) WHERE meets_target = FALSE;

COMMENT ON TABLE ava_spec_performance_metrics IS 'Performance metrics and SLOs for each feature';


-- ============================================================================
-- PART 12: Error Handling Specifications
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_error_handling (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Error Type
    error_type VARCHAR(200) NOT NULL,          -- 'ValidationError', 'APITimeoutError', etc.
    error_code VARCHAR(50),                    -- HTTP code or custom error code

    -- Handling
    handling_strategy TEXT NOT NULL,           -- How the error is handled
    user_message_template TEXT,                -- Message shown to users

    -- Recovery
    is_recoverable BOOLEAN DEFAULT TRUE,
    retry_enabled BOOLEAN DEFAULT FALSE,
    max_retries INTEGER,

    -- Logging
    log_level VARCHAR(20) DEFAULT 'ERROR',
    alert_threshold INTEGER,                   -- Alert after N occurrences

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_errors_spec ON ava_spec_error_handling(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_errors_type ON ava_spec_error_handling(error_type);

COMMENT ON TABLE ava_spec_error_handling IS 'Error handling specifications for each feature';


-- ============================================================================
-- PART 13: Version History
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_version_history (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Version Info
    version VARCHAR(20) NOT NULL,
    previous_version VARCHAR(20),

    -- Change Details
    change_type VARCHAR(50) NOT NULL,          -- 'major', 'minor', 'patch', 'breaking'
    change_summary TEXT NOT NULL,
    change_details JSONB DEFAULT '{}'::jsonb,

    -- Snapshot of key data at this version
    spec_snapshot JSONB NOT NULL,              -- Full spec at this version

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_history_spec ON ava_spec_version_history(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_history_version ON ava_spec_version_history(version);
CREATE INDEX IF NOT EXISTS idx_spec_history_created ON ava_spec_version_history(created_at DESC);

COMMENT ON TABLE ava_spec_version_history IS 'Version history for tracking spec changes over time';


-- ============================================================================
-- PART 14: Search Tags
-- ============================================================================

CREATE TABLE IF NOT EXISTS ava_spec_tags (
    id SERIAL PRIMARY KEY,
    spec_id INTEGER NOT NULL REFERENCES ava_feature_specs(id) ON DELETE CASCADE,

    -- Tag Information
    tag VARCHAR(100) NOT NULL,
    tag_category VARCHAR(100),                 -- 'technology', 'domain', 'integration', 'status'

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(spec_id, tag)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_spec_tags_spec ON ava_spec_tags(spec_id);
CREATE INDEX IF NOT EXISTS idx_spec_tags_tag ON ava_spec_tags(tag);
CREATE INDEX IF NOT EXISTS idx_spec_tags_category ON ava_spec_tags(tag_category, tag);

COMMENT ON TABLE ava_spec_tags IS 'Tags for categorizing and searching features';


-- ============================================================================
-- PART 15: Functions
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_spec_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- Function: Search specs by semantic similarity
-- Note: This function uses FLOAT[] for embeddings. For proper vector search,
-- install pgvector and change embedding column to VECTOR(1536)
CREATE OR REPLACE FUNCTION search_specs_by_embedding(
    p_query_embedding FLOAT[],
    p_limit INTEGER DEFAULT 10,
    p_similarity_threshold DECIMAL DEFAULT 0.7,
    p_category spec_category DEFAULT NULL
)
RETURNS TABLE (
    spec_id INTEGER,
    feature_id VARCHAR,
    feature_name VARCHAR,
    category spec_category,
    purpose TEXT,
    similarity DECIMAL
) AS $$
BEGIN
    -- With FLOAT[], we use a simpler approach (fallback until pgvector available)
    -- This returns all matching features ordered by name (proper similarity requires pgvector)
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        fs.purpose,
        1.0::DECIMAL as similarity  -- Placeholder until pgvector
    FROM ava_feature_specs fs
    WHERE fs.is_current = TRUE
      AND fs.status = 'active'
      AND (p_category IS NULL OR fs.category = p_category)
    ORDER BY fs.feature_name
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_specs_by_embedding IS 'Semantic search for feature specs (requires pgvector for true similarity)';


-- Function: Get all features touching a specific integration
CREATE OR REPLACE FUNCTION get_features_by_integration(
    p_integration_name VARCHAR
)
RETURNS TABLE (
    spec_id INTEGER,
    feature_id VARCHAR,
    feature_name VARCHAR,
    category spec_category,
    integration_type VARCHAR,
    is_critical BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        si.integration_type,
        si.is_critical
    FROM ava_feature_specs fs
    JOIN ava_spec_integrations si ON fs.id = si.spec_id
    WHERE si.integration_name ILIKE '%' || p_integration_name || '%'
      AND fs.is_current = TRUE
      AND fs.status = 'active'
    ORDER BY si.is_critical DESC, fs.feature_name;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_features_by_integration IS 'Find all features using a specific integration (e.g., Robinhood, Redis)';


-- Function: Get features with low efficiency
CREATE OR REPLACE FUNCTION get_low_efficiency_features(
    p_threshold DECIMAL DEFAULT 7.0,
    p_category spec_category DEFAULT NULL
)
RETURNS TABLE (
    spec_id INTEGER,
    feature_id VARCHAR,
    feature_name VARCHAR,
    category spec_category,
    overall_rating DECIMAL,
    priority_level VARCHAR,
    weaknesses TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        er.overall_rating,
        er.priority_level,
        er.weaknesses
    FROM ava_feature_specs fs
    JOIN ava_spec_efficiency_ratings er ON fs.id = er.spec_id
    WHERE fs.is_current = TRUE
      AND fs.status = 'active'
      AND (p_category IS NULL OR fs.category = p_category)
      AND er.overall_rating < p_threshold
    ORDER BY er.overall_rating ASC;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_low_efficiency_features IS 'Find features with efficiency below threshold';


-- Function: Get dependency chain (recursive)
CREATE OR REPLACE FUNCTION get_dependency_chain(
    p_spec_id INTEGER,
    p_direction VARCHAR DEFAULT 'downstream',  -- 'downstream' (what depends on me) or 'upstream' (what I depend on)
    p_max_depth INTEGER DEFAULT 3
)
RETURNS TABLE (
    spec_id INTEGER,
    feature_id VARCHAR,
    feature_name VARCHAR,
    dependency_type dependency_type,
    depth INTEGER,
    path INTEGER[]
) AS $$
BEGIN
    IF p_direction = 'downstream' THEN
        RETURN QUERY
        WITH RECURSIVE dependency_chain AS (
            -- Base case: direct dependencies
            SELECT
                sd.target_spec_id as spec_id,
                sd.dependency_type,
                1 as depth,
                ARRAY[p_spec_id, sd.target_spec_id] as path
            FROM ava_spec_dependencies sd
            WHERE sd.source_spec_id = p_spec_id

            UNION ALL

            -- Recursive case
            SELECT
                sd.target_spec_id,
                sd.dependency_type,
                dc.depth + 1,
                dc.path || sd.target_spec_id
            FROM ava_spec_dependencies sd
            JOIN dependency_chain dc ON sd.source_spec_id = dc.spec_id
            WHERE dc.depth < p_max_depth
              AND NOT sd.target_spec_id = ANY(dc.path)
        )
        SELECT
            dc.spec_id,
            fs.feature_id,
            fs.feature_name,
            dc.dependency_type,
            dc.depth,
            dc.path
        FROM dependency_chain dc
        JOIN ava_feature_specs fs ON dc.spec_id = fs.id
        WHERE fs.is_current = TRUE
        ORDER BY dc.depth, fs.feature_name;
    ELSE
        RETURN QUERY
        WITH RECURSIVE dependency_chain AS (
            -- Base case: what I depend on
            SELECT
                sd.source_spec_id as spec_id,
                sd.dependency_type,
                1 as depth,
                ARRAY[p_spec_id, sd.source_spec_id] as path
            FROM ava_spec_dependencies sd
            WHERE sd.target_spec_id = p_spec_id

            UNION ALL

            -- Recursive case
            SELECT
                sd.source_spec_id,
                sd.dependency_type,
                dc.depth + 1,
                dc.path || sd.source_spec_id
            FROM ava_spec_dependencies sd
            JOIN dependency_chain dc ON sd.target_spec_id = dc.spec_id
            WHERE dc.depth < p_max_depth
              AND NOT sd.source_spec_id = ANY(dc.path)
        )
        SELECT
            dc.spec_id,
            fs.feature_id,
            fs.feature_name,
            dc.dependency_type,
            dc.depth,
            dc.path
        FROM dependency_chain dc
        JOIN ava_feature_specs fs ON dc.spec_id = fs.id
        WHERE fs.is_current = TRUE
        ORDER BY dc.depth, fs.feature_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_dependency_chain IS 'Get upstream or downstream dependency chain with depth control';


-- Function: Get features by database table
CREATE OR REPLACE FUNCTION get_features_by_table(
    p_table_name VARCHAR
)
RETURNS TABLE (
    spec_id INTEGER,
    feature_id VARCHAR,
    feature_name VARCHAR,
    category spec_category,
    usage_type VARCHAR,
    is_owner BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        fs.id,
        fs.feature_id,
        fs.feature_name,
        fs.category,
        dt.usage_type,
        dt.is_owner
    FROM ava_feature_specs fs
    JOIN ava_spec_database_tables dt ON fs.id = dt.spec_id
    WHERE dt.table_name ILIKE '%' || p_table_name || '%'
      AND fs.is_current = TRUE
      AND fs.status = 'active'
    ORDER BY dt.is_owner DESC, fs.feature_name;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_features_by_table IS 'Find all features using a specific database table';


-- ============================================================================
-- PART 16: Views
-- ============================================================================

-- View: Feature overview with efficiency
CREATE OR REPLACE VIEW v_ava_feature_overview AS
SELECT
    fs.id,
    fs.feature_id,
    fs.feature_name,
    fs.category,
    fs.subcategory,
    fs.status,
    fs.maturity_level,
    fs.purpose,
    er.overall_rating,
    er.code_completeness,
    er.test_coverage,
    er.performance as perf_rating,
    er.error_handling,
    er.documentation,
    er.priority_level,
    COALESCE(issue_counts.open_issues, 0) as open_issues,
    COALESCE(issue_counts.critical_issues, 0) as critical_issues,
    COALESCE(enhancement_counts.pending_enhancements, 0) as pending_enhancements,
    COALESCE(dep_counts.dependencies, 0) as dependency_count,
    COALESCE(dep_counts.dependents, 0) as dependent_count,
    fs.updated_at,
    fs.analyzed_at
FROM ava_feature_specs fs
LEFT JOIN ava_spec_efficiency_ratings er ON fs.id = er.spec_id
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE status != 'resolved') as open_issues,
        COUNT(*) FILTER (WHERE severity = 'critical' AND status != 'resolved') as critical_issues
    FROM ava_spec_known_issues
    WHERE spec_id = fs.id
) issue_counts ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*) as pending_enhancements
    FROM ava_spec_enhancements
    WHERE spec_id = fs.id AND status IN ('proposed', 'approved')
) enhancement_counts ON true
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE source_spec_id = fs.id) as dependencies,
        COUNT(*) FILTER (WHERE target_spec_id = fs.id) as dependents
    FROM ava_spec_dependencies
    WHERE source_spec_id = fs.id OR target_spec_id = fs.id
) dep_counts ON true
WHERE fs.is_current = TRUE;

COMMENT ON VIEW v_ava_feature_overview IS 'Comprehensive feature overview with ratings and counts';


-- View: Integration usage summary
CREATE OR REPLACE VIEW v_ava_integration_usage AS
SELECT
    si.integration_name,
    si.integration_type,
    COUNT(DISTINCT fs.id) as feature_count,
    COUNT(*) FILTER (WHERE si.is_critical) as critical_usage_count,
    ARRAY_AGG(DISTINCT fs.category) as categories_using,
    ARRAY_AGG(DISTINCT fs.feature_name ORDER BY fs.feature_name) as features_using
FROM ava_spec_integrations si
JOIN ava_feature_specs fs ON si.spec_id = fs.id
WHERE fs.is_current = TRUE AND fs.status = 'active'
GROUP BY si.integration_name, si.integration_type
ORDER BY feature_count DESC;

COMMENT ON VIEW v_ava_integration_usage IS 'Summary of integration usage across features';


-- View: Low efficiency features requiring attention
CREATE OR REPLACE VIEW v_ava_features_needing_attention AS
SELECT
    fs.feature_id,
    fs.feature_name,
    fs.category,
    er.overall_rating,
    er.priority_level,
    er.weaknesses,
    COALESCE(ic.critical_count, 0) as critical_issues,
    COALESCE(ic.high_count, 0) as high_issues,
    ARRAY(
        SELECT enhancement_title
        FROM ava_spec_enhancements e
        WHERE e.spec_id = fs.id
          AND e.priority IN ('p0_critical', 'p1_high')
          AND e.status IN ('proposed', 'approved')
        LIMIT 3
    ) as top_enhancements
FROM ava_feature_specs fs
JOIN ava_spec_efficiency_ratings er ON fs.id = er.spec_id
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE severity = 'critical') as critical_count,
        COUNT(*) FILTER (WHERE severity = 'high') as high_count
    FROM ava_spec_known_issues
    WHERE spec_id = fs.id AND status != 'resolved'
) ic ON true
WHERE fs.is_current = TRUE
  AND fs.status = 'active'
  AND (er.overall_rating < 7.0 OR ic.critical_count > 0 OR ic.high_count > 0)
ORDER BY er.overall_rating ASC, ic.critical_count DESC;

COMMENT ON VIEW v_ava_features_needing_attention IS 'Features with low efficiency or critical issues';


-- View: Category summary
CREATE OR REPLACE VIEW v_ava_category_summary AS
SELECT
    fs.category,
    COUNT(*) as feature_count,
    ROUND(AVG(er.overall_rating)::numeric, 2) as avg_efficiency,
    ROUND(MIN(er.overall_rating)::numeric, 2) as min_efficiency,
    ROUND(MAX(er.overall_rating)::numeric, 2) as max_efficiency,
    SUM(COALESCE(ic.open_issues, 0)) as total_open_issues,
    SUM(COALESCE(ec.pending, 0)) as total_pending_enhancements
FROM ava_feature_specs fs
LEFT JOIN ava_spec_efficiency_ratings er ON fs.id = er.spec_id
LEFT JOIN LATERAL (
    SELECT COUNT(*) as open_issues
    FROM ava_spec_known_issues
    WHERE spec_id = fs.id AND status != 'resolved'
) ic ON true
LEFT JOIN LATERAL (
    SELECT COUNT(*) as pending
    FROM ava_spec_enhancements
    WHERE spec_id = fs.id AND status IN ('proposed', 'approved')
) ec ON true
WHERE fs.is_current = TRUE AND fs.status = 'active'
GROUP BY fs.category
ORDER BY feature_count DESC;

COMMENT ON VIEW v_ava_category_summary IS 'Summary statistics by category';


-- ============================================================================
-- PART 17: Triggers
-- ============================================================================

-- Apply update triggers
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_feature_specs_updated_at') THEN
        CREATE TRIGGER ava_feature_specs_updated_at
            BEFORE UPDATE ON ava_feature_specs
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_source_files_updated_at') THEN
        CREATE TRIGGER ava_spec_source_files_updated_at
            BEFORE UPDATE ON ava_spec_source_files
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_dependencies_updated_at') THEN
        CREATE TRIGGER ava_spec_dependencies_updated_at
            BEFORE UPDATE ON ava_spec_dependencies
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_efficiency_updated_at') THEN
        CREATE TRIGGER ava_spec_efficiency_updated_at
            BEFORE UPDATE ON ava_spec_efficiency_ratings
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_issues_updated_at') THEN
        CREATE TRIGGER ava_spec_issues_updated_at
            BEFORE UPDATE ON ava_spec_known_issues
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_enhancements_updated_at') THEN
        CREATE TRIGGER ava_spec_enhancements_updated_at
            BEFORE UPDATE ON ava_spec_enhancements
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_integrations_updated_at') THEN
        CREATE TRIGGER ava_spec_integrations_updated_at
            BEFORE UPDATE ON ava_spec_integrations
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'ava_spec_perf_metrics_updated_at') THEN
        CREATE TRIGGER ava_spec_perf_metrics_updated_at
            BEFORE UPDATE ON ava_spec_performance_metrics
            FOR EACH ROW
            EXECUTE FUNCTION update_spec_updated_at();
    END IF;
END $$;


-- ============================================================================
-- PART 18: Migration Completion
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'schema_migrations'
    ) THEN
        INSERT INTO schema_migrations (version, name, applied_at)
        VALUES (6, 'technical_specs_system', NOW())
        ON CONFLICT (version) DO NOTHING;
    END IF;
END $$;

COMMIT;


-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Verify tables were created
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE tablename LIKE 'ava_spec%' OR tablename = 'ava_feature_specs'
ORDER BY tablename;

-- Verify enums were created
SELECT typname, enumlabel
FROM pg_type t
JOIN pg_enum e ON t.oid = e.enumtypid
WHERE typname IN ('spec_category', 'dependency_type', 'issue_severity', 'enhancement_priority')
ORDER BY typname, enumsortorder;


-- ============================================================================
-- SUCCESS MESSAGE
-- ============================================================================

SELECT 'AVA Technical Specifications System migration completed successfully!' as status,
       'Tables: ava_feature_specs, ava_spec_source_files, ava_spec_dependencies, ava_spec_api_endpoints, ava_spec_database_tables, ava_spec_efficiency_ratings, ava_spec_known_issues, ava_spec_enhancements, ava_spec_integrations, ava_spec_performance_metrics, ava_spec_error_handling, ava_spec_version_history, ava_spec_tags' as tables_created,
       'Functions: search_specs_by_embedding, get_features_by_integration, get_low_efficiency_features, get_dependency_chain, get_features_by_table' as functions_created,
       'Views: v_ava_feature_overview, v_ava_integration_usage, v_ava_features_needing_attention, v_ava_category_summary' as views_created;
