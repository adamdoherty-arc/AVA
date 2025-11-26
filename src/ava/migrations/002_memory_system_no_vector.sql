-- ============================================================================
-- AVA Memory System - Multi-Level Memory Enhancement (No pgvector)
-- Migration: 002_memory_system_no_vector.sql
-- Version: 2.1.0
-- Date: 2025-11-23
-- Description: Memory system without pgvector dependency
-- ============================================================================

BEGIN;

-- ============================================================================
-- Table: ava_user_memory
-- Purpose: Store multi-level user memory (preferences, facts, entities)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ava_user_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    category VARCHAR(100),
    key VARCHAR(200) NOT NULL,
    value JSONB NOT NULL,
    confidence_score DECIMAL(3,2) DEFAULT 1.0,
    importance INTEGER DEFAULT 5,
    source VARCHAR(100),
    context TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, platform, memory_type, key),
    CONSTRAINT chk_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_importance CHECK (importance >= 1 AND importance <= 10)
);

CREATE INDEX IF NOT EXISTS idx_ava_user_memory_user ON ava_user_memory(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_type ON ava_user_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_category ON ava_user_memory(category);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_key ON ava_user_memory(user_id, platform, key);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_importance ON ava_user_memory(importance DESC) WHERE importance >= 8;
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_expires ON ava_user_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_accessed ON ava_user_memory(last_accessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_value_gin ON ava_user_memory USING GIN(value);

-- ============================================================================
-- Table: ava_conversation_summaries
-- Purpose: Store conversation summaries (embeddings stored as JSON)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ava_conversation_summaries (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    session_id VARCHAR(100),
    conversation_start TIMESTAMP WITH TIME ZONE NOT NULL,
    conversation_end TIMESTAMP WITH TIME ZONE NOT NULL,
    message_count INTEGER NOT NULL,
    summary TEXT NOT NULL,
    key_topics TEXT[],
    entities_mentioned TEXT[],
    sentiment VARCHAR(20),
    embedding JSONB,  -- Store embedding as JSON array
    original_tokens INTEGER,
    summary_tokens INTEGER,
    compression_ratio DECIMAL(5,2),
    tokens_saved INTEGER,
    model_used VARCHAR(100),
    quality_score DECIMAL(3,2),
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT chk_message_count CHECK (message_count > 0),
    CONSTRAINT chk_tokens CHECK (original_tokens > 0 AND summary_tokens > 0),
    CONSTRAINT chk_quality CHECK (quality_score >= 0.0 AND quality_score <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_user ON ava_conversation_summaries(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_session ON ava_conversation_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_time ON ava_conversation_summaries(conversation_start DESC);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_created ON ava_conversation_summaries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_topics ON ava_conversation_summaries USING GIN(key_topics);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_entities ON ava_conversation_summaries USING GIN(entities_mentioned);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_summary_search ON ava_conversation_summaries USING GIN(to_tsvector('english', summary));

-- ============================================================================
-- Table: ava_entity_memory
-- Purpose: Track specific entities (tickers, strategies) and interactions
-- ============================================================================
CREATE TABLE IF NOT EXISTS ava_entity_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    entity_name VARCHAR(200),
    mention_count INTEGER DEFAULT 1,
    first_mentioned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_mentioned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    contexts TEXT[],
    overall_sentiment VARCHAR(20) DEFAULT 'neutral',
    interest_score INTEGER DEFAULT 5,
    related_entities JSONB DEFAULT '[]'::jsonb,
    user_notes TEXT,
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, platform, entity_type, entity_id),
    CONSTRAINT chk_interest_score CHECK (interest_score >= 1 AND interest_score <= 10)
);

CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_user ON ava_entity_memory(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_type ON ava_entity_memory(entity_type);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_entity ON ava_entity_memory(entity_id);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_interest ON ava_entity_memory(interest_score DESC);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_mentions ON ava_entity_memory(mention_count DESC);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_last_mentioned ON ava_entity_memory(last_mentioned_at DESC);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_contexts ON ava_entity_memory USING GIN(contexts);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_tags ON ava_entity_memory USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_related ON ava_entity_memory USING GIN(related_entities);

-- ============================================================================
-- Functions and Triggers
-- ============================================================================

CREATE OR REPLACE FUNCTION update_ava_memory_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ava_user_memory_updated_at
    BEFORE UPDATE ON ava_user_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_ava_memory_updated_at();

CREATE TRIGGER ava_entity_memory_updated_at
    BEFORE UPDATE ON ava_entity_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_ava_memory_updated_at();

CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    rows_deleted INTEGER;
BEGIN
    DELETE FROM ava_user_memory
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    GET DIAGNOSTICS rows_deleted = ROW_COUNT;
    RETURN rows_deleted;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_user_memory_summary(
    p_user_id VARCHAR,
    p_platform VARCHAR
)
RETURNS TABLE (
    memory_type VARCHAR,
    count BIGINT,
    avg_importance NUMERIC,
    most_recent TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        um.memory_type,
        COUNT(*)::BIGINT,
        AVG(um.importance)::NUMERIC,
        MAX(um.updated_at)
    FROM ava_user_memory um
    WHERE um.user_id = p_user_id
      AND um.platform = p_platform
      AND (um.expires_at IS NULL OR um.expires_at > NOW())
    GROUP BY um.memory_type
    ORDER BY COUNT(*) DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Views
-- ============================================================================

CREATE OR REPLACE VIEW v_ava_user_memory_health AS
SELECT
    user_id,
    platform,
    COUNT(*) as total_memories,
    COUNT(*) FILTER (WHERE memory_type = 'preference') as preferences,
    COUNT(*) FILTER (WHERE memory_type = 'fact') as facts,
    COUNT(*) FILTER (WHERE memory_type = 'entity') as entities,
    AVG(importance) as avg_importance,
    AVG(confidence_score) as avg_confidence,
    MAX(updated_at) as last_updated
FROM ava_user_memory
WHERE expires_at IS NULL OR expires_at > NOW()
GROUP BY user_id, platform;

CREATE OR REPLACE VIEW v_ava_conversation_stats AS
SELECT
    user_id,
    platform,
    COUNT(*) as total_summaries,
    SUM(message_count) as total_messages,
    AVG(compression_ratio) as avg_compression_ratio,
    SUM(tokens_saved) as total_tokens_saved,
    AVG(quality_score) as avg_quality_score,
    MAX(conversation_end) as last_conversation
FROM ava_conversation_summaries
GROUP BY user_id, platform;

CREATE OR REPLACE VIEW v_ava_top_entities AS
SELECT
    user_id,
    platform,
    entity_type,
    entity_id,
    entity_name,
    mention_count,
    interest_score,
    overall_sentiment,
    last_mentioned_at
FROM ava_entity_memory
WHERE mention_count >= 3
ORDER BY mention_count DESC, interest_score DESC;

-- ============================================================================
-- Migration tracking
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'schema_migrations'
    ) THEN
        INSERT INTO schema_migrations (version, name, applied_at)
        VALUES (2, 'memory_system', NOW())
        ON CONFLICT (version) DO NOTHING;
    END IF;
END $$;

COMMIT;

SELECT 'AVA Memory System migration completed!' as status,
       'Tables: ava_user_memory, ava_conversation_summaries, ava_entity_memory' as info;
