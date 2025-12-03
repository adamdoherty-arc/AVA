-- Sports Betting Query Optimization Indexes
-- ==========================================
-- These indexes optimize the most common sports betting queries
--
-- Run with: psql -d trading -f migrations/sports_indexes_migration.sql
--
-- Author: AVA Trading Platform
-- Created: 2025-11-30

-- =============================================================================
-- NFL Games Indexes
-- =============================================================================

-- Primary lookup pattern: live games
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_games_is_live
    ON nfl_games (is_live)
    WHERE is_live = true;

-- Scheduled games lookup (for upcoming games)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_games_scheduled
    ON nfl_games (game_time, game_status)
    WHERE game_status = 'scheduled';

-- Games with odds (for betting opportunities)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_games_with_odds
    ON nfl_games (game_time)
    WHERE moneyline_home IS NOT NULL;

-- Team lookup for predictions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_games_teams
    ON nfl_games (home_team, away_team);

-- Game status filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_games_status
    ON nfl_games (game_status, game_time);

-- Composite index for unified best bets query
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nfl_games_betting
    ON nfl_games (game_status, is_live, game_time)
    INCLUDE (home_team, away_team, moneyline_home, moneyline_away, spread_home, over_under);


-- =============================================================================
-- NBA Games Indexes
-- =============================================================================

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_games_is_live
    ON nba_games (is_live)
    WHERE is_live = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_games_scheduled
    ON nba_games (game_time, game_status)
    WHERE game_status = 'scheduled';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_games_with_odds
    ON nba_games (game_time)
    WHERE moneyline_home IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_games_teams
    ON nba_games (home_team, away_team);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_games_status
    ON nba_games (game_status, game_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nba_games_betting
    ON nba_games (game_status, is_live, game_time)
    INCLUDE (home_team, away_team, moneyline_home, moneyline_away, spread_home, over_under);


-- =============================================================================
-- NCAA Football Games Indexes
-- =============================================================================

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_football_is_live
    ON ncaa_football_games (is_live)
    WHERE is_live = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_football_scheduled
    ON ncaa_football_games (game_time, game_status)
    WHERE game_status = 'scheduled';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_football_teams
    ON ncaa_football_games (home_team, away_team);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_football_status
    ON ncaa_football_games (game_status, game_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_football_ranked
    ON ncaa_football_games (home_rank, away_rank)
    WHERE home_rank IS NOT NULL OR away_rank IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_football_betting
    ON ncaa_football_games (game_status, is_live, game_time)
    INCLUDE (home_team, away_team, spread_home, over_under, home_rank, away_rank);


-- =============================================================================
-- NCAA Basketball Games Indexes
-- =============================================================================

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_basketball_is_live
    ON ncaa_basketball_games (is_live)
    WHERE is_live = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_basketball_scheduled
    ON ncaa_basketball_games (game_time, game_status)
    WHERE game_status = 'scheduled';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_basketball_teams
    ON ncaa_basketball_games (home_team, away_team);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_basketball_status
    ON ncaa_basketball_games (game_status, game_time);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_basketball_ranked
    ON ncaa_basketball_games (home_rank, away_rank)
    WHERE home_rank IS NOT NULL OR away_rank IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ncaa_basketball_betting
    ON ncaa_basketball_games (game_status, is_live, game_time)
    INCLUDE (home_team, away_team, spread_home, over_under, home_rank, away_rank);


-- =============================================================================
-- Odds History Indexes (if table exists)
-- =============================================================================

-- Only create if odds_history table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'odds_history') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_odds_history_game
            ON odds_history (game_id, recorded_at DESC);

        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_odds_history_recent
            ON odds_history (recorded_at DESC)
            WHERE recorded_at > NOW() - INTERVAL '7 days';
    END IF;
END $$;


-- =============================================================================
-- Predictions Cache Table (new)
-- =============================================================================

CREATE TABLE IF NOT EXISTS sports_predictions_cache (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL,
    sport VARCHAR(20) NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    winner VARCHAR(100) NOT NULL,
    win_probability DECIMAL(5,4) NOT NULL,
    confidence VARCHAR(20) NOT NULL,
    expected_value DECIMAL(7,2),
    edge_vs_market DECIMAL(7,2),
    kelly_fraction DECIMAL(6,4),
    recommendation VARCHAR(20),
    reasoning TEXT,
    model_agreement DECIMAL(4,3),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE (game_id, sport)
);

CREATE INDEX IF NOT EXISTS idx_predictions_cache_lookup
    ON sports_predictions_cache (sport, game_id);

CREATE INDEX IF NOT EXISTS idx_predictions_cache_expiry
    ON sports_predictions_cache (expires_at)
    WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_predictions_cache_best_bets
    ON sports_predictions_cache (sport, expected_value DESC, confidence)
    WHERE recommendation IN ('STRONG_BET', 'BET');


-- =============================================================================
-- Analyze Tables (update statistics for query planner)
-- =============================================================================

ANALYZE nfl_games;
ANALYZE nba_games;
ANALYZE ncaa_football_games;
ANALYZE ncaa_basketball_games;
ANALYZE sports_predictions_cache;


-- =============================================================================
-- Query Performance Comments
-- =============================================================================

COMMENT ON INDEX idx_nfl_games_betting IS 'Optimizes unified best bets UNION query - includes all betting columns';
COMMENT ON INDEX idx_nba_games_betting IS 'Optimizes unified best bets UNION query - includes all betting columns';
COMMENT ON INDEX idx_ncaa_football_betting IS 'Optimizes unified best bets UNION query - includes all betting columns';
COMMENT ON INDEX idx_ncaa_basketball_betting IS 'Optimizes unified best bets UNION query - includes all betting columns';

-- Done!
SELECT 'Sports betting indexes created successfully' AS status;
