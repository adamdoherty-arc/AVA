-- ============================================================================
-- Sports Real-Time Data Pipeline - Unified Database Schema
-- ============================================================================
-- Purpose: Store real-time NBA, NCAA games with live odds and AI predictions
-- Database: magnus (PostgreSQL)
-- Created: 2025-11-28
-- ============================================================================

-- ============================================================================
-- Table: nba_games
-- ============================================================================
CREATE TABLE IF NOT EXISTS nba_games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) UNIQUE NOT NULL,
    season VARCHAR(10) NOT NULL,  -- "2024-25"

    -- Teams
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_team_abbr VARCHAR(10),
    away_team_abbr VARCHAR(10),

    -- Schedule
    game_time TIMESTAMP WITH TIME ZONE NOT NULL,
    venue VARCHAR(200),

    -- Score (updated live)
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    quarter INTEGER DEFAULT 0,  -- 0=pregame, 1-4=quarters, 5=OT
    time_remaining VARCHAR(20),
    possession VARCHAR(10),

    -- Game state
    game_status VARCHAR(20) DEFAULT 'scheduled',
    is_live BOOLEAN DEFAULT false,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,

    -- Betting lines
    spread_home DECIMAL(4,1),
    spread_odds_home INTEGER,
    spread_odds_away INTEGER,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    over_under DECIMAL(5,1),
    over_odds INTEGER,
    under_odds INTEGER,

    -- Live odds movement
    opening_spread DECIMAL(4,1),
    opening_total DECIMAL(5,1),
    spread_movement DECIMAL(4,1),  -- Current - Opening
    total_movement DECIMAL(4,1),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_game_data JSONB,

    CONSTRAINT chk_nba_game_status CHECK (game_status IN ('scheduled', 'live', 'halftime', 'final', 'postponed', 'cancelled', 'delayed'))
);

-- Single column indexes
CREATE INDEX IF NOT EXISTS idx_nba_games_status ON nba_games(game_status);
CREATE INDEX IF NOT EXISTS idx_nba_games_live ON nba_games(is_live) WHERE is_live = true;
CREATE INDEX IF NOT EXISTS idx_nba_games_time ON nba_games(game_time);
CREATE INDEX IF NOT EXISTS idx_nba_games_teams ON nba_games(home_team, away_team);

-- COMPOSITE indexes for query optimization (10x faster queries)
CREATE INDEX IF NOT EXISTS idx_nba_games_status_time ON nba_games(game_status, game_time);
CREATE INDEX IF NOT EXISTS idx_nba_games_live_time ON nba_games(is_live, game_time DESC) WHERE is_live = true;

-- ============================================================================
-- Table: ncaa_football_games
-- ============================================================================
CREATE TABLE IF NOT EXISTS ncaa_football_games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) UNIQUE NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Teams
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_team_abbr VARCHAR(20),
    away_team_abbr VARCHAR(20),
    home_rank INTEGER,  -- AP Top 25 ranking
    away_rank INTEGER,
    conference VARCHAR(50),

    -- Schedule
    game_time TIMESTAMP WITH TIME ZONE NOT NULL,
    venue VARCHAR(200),
    is_outdoor BOOLEAN DEFAULT true,

    -- Score
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    quarter INTEGER DEFAULT 0,
    time_remaining VARCHAR(20),
    possession VARCHAR(20),

    -- Game state
    game_status VARCHAR(20) DEFAULT 'scheduled',
    is_live BOOLEAN DEFAULT false,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,

    -- Betting lines
    spread_home DECIMAL(4,1),
    spread_odds_home INTEGER,
    spread_odds_away INTEGER,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    over_under DECIMAL(5,1),
    over_odds INTEGER,
    under_odds INTEGER,

    -- Live odds movement
    opening_spread DECIMAL(4,1),
    opening_total DECIMAL(5,1),

    -- Weather
    temperature INTEGER,
    weather_condition VARCHAR(100),
    wind_speed INTEGER,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_game_data JSONB,

    CONSTRAINT chk_ncaaf_game_status CHECK (game_status IN ('scheduled', 'live', 'halftime', 'final', 'postponed', 'cancelled', 'delayed'))
);

-- Single column indexes
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_status ON ncaa_football_games(game_status);
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_live ON ncaa_football_games(is_live) WHERE is_live = true;
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_time ON ncaa_football_games(game_time);
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_ranked ON ncaa_football_games(home_rank, away_rank) WHERE home_rank IS NOT NULL OR away_rank IS NOT NULL;

-- COMPOSITE indexes for query optimization (10x faster queries)
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_status_time ON ncaa_football_games(game_status, game_time);
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_live_time ON ncaa_football_games(is_live, game_time DESC) WHERE is_live = true;

-- ============================================================================
-- Table: ncaa_basketball_games
-- ============================================================================
CREATE TABLE IF NOT EXISTS ncaa_basketball_games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) UNIQUE NOT NULL,
    season VARCHAR(10) NOT NULL,

    -- Teams
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_team_abbr VARCHAR(20),
    away_team_abbr VARCHAR(20),
    home_rank INTEGER,
    away_rank INTEGER,
    conference VARCHAR(50),

    -- Schedule
    game_time TIMESTAMP WITH TIME ZONE NOT NULL,
    venue VARCHAR(200),

    -- Score
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    half INTEGER DEFAULT 0,  -- 0=pregame, 1-2=halves, 3=OT
    time_remaining VARCHAR(20),
    possession VARCHAR(20),

    -- Game state
    game_status VARCHAR(20) DEFAULT 'scheduled',
    is_live BOOLEAN DEFAULT false,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,

    -- Betting lines
    spread_home DECIMAL(4,1),
    spread_odds_home INTEGER,
    spread_odds_away INTEGER,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    over_under DECIMAL(5,1),
    over_odds INTEGER,
    under_odds INTEGER,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_game_data JSONB,

    CONSTRAINT chk_ncaab_game_status CHECK (game_status IN ('scheduled', 'live', 'halftime', 'final', 'postponed', 'cancelled', 'delayed'))
);

-- Single column indexes
CREATE INDEX IF NOT EXISTS idx_ncaab_games_status ON ncaa_basketball_games(game_status);
CREATE INDEX IF NOT EXISTS idx_ncaab_games_live ON ncaa_basketball_games(is_live) WHERE is_live = true;
CREATE INDEX IF NOT EXISTS idx_ncaab_games_time ON ncaa_basketball_games(game_time);

-- COMPOSITE indexes for query optimization (10x faster queries)
CREATE INDEX IF NOT EXISTS idx_ncaab_games_status_time ON ncaa_basketball_games(game_status, game_time);
CREATE INDEX IF NOT EXISTS idx_ncaab_games_live_time ON ncaa_basketball_games(is_live, game_time DESC) WHERE is_live = true;

-- ============================================================================
-- Table: live_odds_snapshots
-- Store historical odds movement for AI analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS live_odds_snapshots (
    id SERIAL PRIMARY KEY,
    sport VARCHAR(20) NOT NULL,  -- NFL, NBA, NCAAF, NCAAB
    game_id VARCHAR(50) NOT NULL,
    snapshot_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Current odds
    spread_home DECIMAL(4,1),
    spread_odds_home INTEGER,
    spread_odds_away INTEGER,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    over_under DECIMAL(5,1),
    over_odds INTEGER,
    under_odds INTEGER,

    -- Game state at snapshot
    home_score INTEGER,
    away_score INTEGER,
    period INTEGER,
    time_remaining VARCHAR(20),

    -- Source
    odds_source VARCHAR(50),  -- DraftKings, FanDuel, etc.

    CONSTRAINT idx_odds_unique UNIQUE (sport, game_id, snapshot_time)
);

CREATE INDEX IF NOT EXISTS idx_odds_game ON live_odds_snapshots(sport, game_id);
CREATE INDEX IF NOT EXISTS idx_odds_time ON live_odds_snapshots(snapshot_time);

-- ============================================================================
-- Table: ai_betting_recommendations
-- Store AI-generated betting recommendations
-- ============================================================================
CREATE TABLE IF NOT EXISTS ai_betting_recommendations (
    id SERIAL PRIMARY KEY,
    sport VARCHAR(20) NOT NULL,
    game_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Recommendation
    bet_type VARCHAR(30) NOT NULL,  -- spread, moneyline, over, under, live_spread, live_total
    pick VARCHAR(100) NOT NULL,  -- Team or Over/Under
    odds INTEGER,

    -- AI Analysis
    confidence INTEGER CHECK (confidence >= 0 AND confidence <= 100),
    win_probability DECIMAL(5,2),
    expected_value DECIMAL(6,2),  -- Percentage

    -- Reasoning
    key_factors JSONB,  -- Array of factors
    reasoning TEXT,

    -- Live game context (if applicable)
    game_state JSONB,  -- Score, time, momentum
    odds_movement_factor DECIMAL(4,2),  -- How much odds moved influenced pick

    -- Outcome tracking
    is_settled BOOLEAN DEFAULT false,
    result VARCHAR(10),  -- win, loss, push
    settled_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT chk_bet_type CHECK (bet_type IN ('spread', 'moneyline', 'over', 'under', 'live_spread', 'live_total', 'live_ml'))
);

CREATE INDEX IF NOT EXISTS idx_ai_recs_game ON ai_betting_recommendations(sport, game_id);
CREATE INDEX IF NOT EXISTS idx_ai_recs_unsettled ON ai_betting_recommendations(is_settled) WHERE is_settled = false;
CREATE INDEX IF NOT EXISTS idx_ai_recs_confidence ON ai_betting_recommendations(confidence DESC);

-- ============================================================================
-- Auto-update timestamp trigger
-- ============================================================================
CREATE OR REPLACE FUNCTION update_last_updated_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_nba_games_timestamp') THEN
        CREATE TRIGGER update_nba_games_timestamp
            BEFORE UPDATE ON nba_games
            FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ncaaf_games_timestamp') THEN
        CREATE TRIGGER update_ncaaf_games_timestamp
            BEFORE UPDATE ON ncaa_football_games
            FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ncaab_games_timestamp') THEN
        CREATE TRIGGER update_ncaab_games_timestamp
            BEFORE UPDATE ON ncaa_basketball_games
            FOR EACH ROW EXECUTE FUNCTION update_last_updated_column();
    END IF;
END $$;
