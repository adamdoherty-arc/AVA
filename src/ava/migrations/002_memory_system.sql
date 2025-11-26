-- ============================================================================
-- AVA Memory System - Multi-Level Memory Enhancement
-- Migration: 002_memory_system.sql
-- Version: 2.1.0
-- Date: 2025-11-23
-- Description: Adds multi-level memory system for persistent user preferences,
--              conversation history, entity tracking, and context compression
-- ============================================================================

BEGIN;

-- ============================================================================
-- Enable pgvector extension for embeddings (if not already enabled)
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Table: ava_user_memory
-- Purpose: Store multi-level user memory (preferences, facts, entities)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ava_user_memory (
    id SERIAL PRIMARY KEY,

    -- User identification (cross-platform)
    user_id VARCHAR(100) NOT NULL,  -- Format: "telegram:123456" or "discord:789012"
    platform VARCHAR(50) NOT NULL,   -- 'telegram', 'discord', 'web'

    -- Memory categorization
    memory_type VARCHAR(50) NOT NULL,  -- 'preference', 'fact', 'entity', 'trading_style'
    category VARCHAR(100),              -- 'risk_tolerance', 'favorite_ticker', 'strategy', etc.

    -- Memory content
    key VARCHAR(200) NOT NULL,          -- Specific memory key (e.g., 'preferred_strategy')
    value JSONB NOT NULL,               -- Memory value (structured data)
    confidence_score DECIMAL(3,2) DEFAULT 1.0,  -- 0.0-1.0, how confident we are

    -- Metadata
    source VARCHAR(100),                -- 'explicit', 'inferred', 'observed'
    context TEXT,                       -- Original context where this was learned
    importance INTEGER DEFAULT 5,       -- 1-10 scale

    -- Lifecycle
    access_count INTEGER DEFAULT 0,     -- How many times accessed
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE, -- NULL = permanent

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(user_id, platform, memory_type, key),
    CONSTRAINT chk_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    CONSTRAINT chk_importance CHECK (importance >= 1 AND importance <= 10)
);

-- Indexes for ava_user_memory
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_user ON ava_user_memory(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_type ON ava_user_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_category ON ava_user_memory(category);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_key ON ava_user_memory(user_id, platform, key);
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_importance ON ava_user_memory(importance DESC) WHERE importance >= 8;
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_expires ON ava_user_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_accessed ON ava_user_memory(last_accessed_at DESC);

-- GIN index for JSONB value search
CREATE INDEX IF NOT EXISTS idx_ava_user_memory_value_gin ON ava_user_memory USING GIN(value);

COMMENT ON TABLE ava_user_memory IS 'Multi-level memory system for user preferences, facts, and entities';
COMMENT ON COLUMN ava_user_memory.user_id IS 'Cross-platform user identifier (e.g., telegram:123456)';
COMMENT ON COLUMN ava_user_memory.memory_type IS 'Type of memory: preference, fact, entity, trading_style';
COMMENT ON COLUMN ava_user_memory.confidence_score IS 'Confidence in this memory (0.0-1.0)';
COMMENT ON COLUMN ava_user_memory.importance IS 'Importance score (1-10) for prioritization';


-- ============================================================================
-- Table: ava_conversation_summaries
-- Purpose: Store conversation summaries with vector embeddings for RAG
-- ============================================================================
CREATE TABLE IF NOT EXISTS ava_conversation_summaries (
    id SERIAL PRIMARY KEY,

    -- User identification
    user_id VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL,

    -- Conversation metadata
    session_id VARCHAR(100),            -- Optional session grouping
    conversation_start TIMESTAMP WITH TIME ZONE NOT NULL,
    conversation_end TIMESTAMP WITH TIME ZONE NOT NULL,
    message_count INTEGER NOT NULL,

    -- Summary content
    summary TEXT NOT NULL,              -- Human-readable summary
    key_topics TEXT[],                  -- Array of key topics discussed
    entities_mentioned TEXT[],          -- Entities (tickers, strategies) mentioned
    sentiment VARCHAR(20),              -- 'positive', 'negative', 'neutral', 'mixed'

    -- Vector embedding for semantic search
    embedding VECTOR(1536),             -- OpenAI text-embedding-ada-002 dimensions

    -- Context compression metrics
    original_tokens INTEGER,            -- Original conversation token count
    summary_tokens INTEGER,             -- Summary token count
    compression_ratio DECIMAL(5,2),     -- Compression achieved
    tokens_saved INTEGER,               -- Tokens saved by summarization

    -- Metadata
    model_used VARCHAR(100),            -- Model used for summarization
    quality_score DECIMAL(3,2),         -- 0.0-1.0, quality of summary

    -- Lifecycle
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT chk_message_count CHECK (message_count > 0),
    CONSTRAINT chk_tokens CHECK (original_tokens > 0 AND summary_tokens > 0),
    CONSTRAINT chk_quality CHECK (quality_score >= 0.0 AND quality_score <= 1.0)
);

-- Indexes for ava_conversation_summaries
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_user ON ava_conversation_summaries(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_session ON ava_conversation_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_time ON ava_conversation_summaries(conversation_start DESC);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_created ON ava_conversation_summaries(created_at DESC);

-- GIN indexes for array searches
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_topics ON ava_conversation_summaries USING GIN(key_topics);
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_entities ON ava_conversation_summaries USING GIN(entities_mentioned);

-- Vector similarity search index (using HNSW for fast approximate nearest neighbor)
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_embedding ON ava_conversation_summaries
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search on summary
CREATE INDEX IF NOT EXISTS idx_ava_conv_summaries_summary_search ON ava_conversation_summaries
    USING GIN(to_tsvector('english', summary));

COMMENT ON TABLE ava_conversation_summaries IS 'Conversation summaries with vector embeddings for semantic search and RAG';
COMMENT ON COLUMN ava_conversation_summaries.embedding IS 'Vector embedding for semantic similarity search (1536 dimensions)';
COMMENT ON COLUMN ava_conversation_summaries.compression_ratio IS 'Compression ratio achieved (original_tokens / summary_tokens)';
COMMENT ON COLUMN ava_conversation_summaries.tokens_saved IS 'Tokens saved through summarization for cost optimization';


-- ============================================================================
-- Table: ava_entity_memory
-- Purpose: Track specific entities (tickers, strategies) and user interactions
-- ============================================================================
CREATE TABLE IF NOT EXISTS ava_entity_memory (
    id SERIAL PRIMARY KEY,

    -- User identification
    user_id VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL,

    -- Entity identification
    entity_type VARCHAR(50) NOT NULL,   -- 'ticker', 'strategy', 'sector', 'person'
    entity_id VARCHAR(100) NOT NULL,    -- 'AAPL', 'wheel_strategy', 'tech', etc.
    entity_name VARCHAR(200),           -- Human-readable name

    -- Interaction tracking
    mention_count INTEGER DEFAULT 1,    -- Number of times mentioned
    first_mentioned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_mentioned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Context and sentiment
    contexts TEXT[],                    -- Recent contexts where mentioned (last 5)
    overall_sentiment VARCHAR(20) DEFAULT 'neutral',  -- 'positive', 'negative', 'neutral'
    interest_score INTEGER DEFAULT 5,   -- 1-10, how interested user is

    -- Relationships
    related_entities JSONB DEFAULT '[]'::jsonb,  -- Related entities

    -- User notes/preferences
    user_notes TEXT,
    tags TEXT[],                        -- User-defined tags

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(user_id, platform, entity_type, entity_id),
    CONSTRAINT chk_interest_score CHECK (interest_score >= 1 AND interest_score <= 10)
);

-- Indexes for ava_entity_memory
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_user ON ava_entity_memory(user_id, platform);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_type ON ava_entity_memory(entity_type);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_entity ON ava_entity_memory(entity_id);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_interest ON ava_entity_memory(interest_score DESC);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_mentions ON ava_entity_memory(mention_count DESC);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_last_mentioned ON ava_entity_memory(last_mentioned_at DESC);

-- GIN indexes for array searches
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_contexts ON ava_entity_memory USING GIN(contexts);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_tags ON ava_entity_memory USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_ava_entity_memory_related ON ava_entity_memory USING GIN(related_entities);

COMMENT ON TABLE ava_entity_memory IS 'Entity tracking (tickers, strategies) with user interaction history';
COMMENT ON COLUMN ava_entity_memory.contexts IS 'Recent contexts where entity was mentioned (FIFO, max 5)';
COMMENT ON COLUMN ava_entity_memory.interest_score IS 'User interest score (1-10) based on interaction frequency';


-- ============================================================================
-- Functions and Triggers
-- ============================================================================

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_ava_memory_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER ava_user_memory_updated_at
    BEFORE UPDATE ON ava_user_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_ava_memory_updated_at();

CREATE TRIGGER ava_entity_memory_updated_at
    BEFORE UPDATE ON ava_entity_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_ava_memory_updated_at();

-- Function: Increment memory access count
CREATE OR REPLACE FUNCTION increment_memory_access(
    p_table_name VARCHAR,
    p_id INTEGER
)
RETURNS VOID AS $$
BEGIN
    IF p_table_name = 'ava_user_memory' THEN
        UPDATE ava_user_memory
        SET access_count = access_count + 1,
            last_accessed_at = NOW()
        WHERE id = p_id;
    ELSIF p_table_name = 'ava_conversation_summaries' THEN
        UPDATE ava_conversation_summaries
        SET access_count = access_count + 1,
            last_accessed_at = NOW()
        WHERE id = p_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION increment_memory_access(VARCHAR, INTEGER) IS 'Increment access count and update last_accessed_at for memory tables';

-- Function: Clean up expired memories
CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    rows_deleted INTEGER;
BEGIN
    DELETE FROM ava_user_memory
    WHERE expires_at IS NOT NULL
      AND expires_at < NOW();

    GET DIAGNOSTICS rows_deleted = ROW_COUNT;
    RETURN rows_deleted;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_expired_memories() IS 'Delete expired memories (run as cron job)';

-- Function: Get user memory summary
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

COMMENT ON FUNCTION get_user_memory_summary(VARCHAR, VARCHAR) IS 'Get summary statistics of user memory by type';

-- Function: Search conversation summaries by embedding similarity
CREATE OR REPLACE FUNCTION search_similar_conversations(
    p_user_id VARCHAR,
    p_platform VARCHAR,
    p_embedding VECTOR(1536),
    p_limit INTEGER DEFAULT 5,
    p_similarity_threshold DECIMAL DEFAULT 0.7
)
RETURNS TABLE (
    summary_id INTEGER,
    summary_text TEXT,
    similarity_score DECIMAL,
    conversation_date TIMESTAMP WITH TIME ZONE,
    key_topics TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cs.id,
        cs.summary,
        (1 - (cs.embedding <=> p_embedding))::DECIMAL as similarity,
        cs.conversation_start,
        cs.key_topics
    FROM ava_conversation_summaries cs
    WHERE cs.user_id = p_user_id
      AND cs.platform = p_platform
      AND (1 - (cs.embedding <=> p_embedding)) >= p_similarity_threshold
    ORDER BY cs.embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_similar_conversations IS 'Find similar past conversations using vector similarity search';


-- ============================================================================
-- Views for Analytics
-- ============================================================================

-- View: User memory health
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

COMMENT ON VIEW v_ava_user_memory_health IS 'Overview of memory health per user';

-- View: Conversation summary statistics
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

COMMENT ON VIEW v_ava_conversation_stats IS 'Conversation summary statistics per user';

-- View: Top entities by user
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
WHERE mention_count >= 3  -- Only show entities mentioned at least 3 times
ORDER BY mention_count DESC, interest_score DESC;

COMMENT ON VIEW v_ava_top_entities IS 'Top entities by mention count and interest score';


-- ============================================================================
-- Migration Completion
-- ============================================================================

-- Record migration in schema_migrations table (if it exists)
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

-- Verify pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- ============================================================================
-- Success Message
-- ============================================================================

SELECT 'AVA Memory System migration completed successfully!' as status,
       'Tables created: ava_user_memory, ava_conversation_summaries, ava_entity_memory' as tables_created,
       'pgvector extension enabled for semantic search' as vector_support;
