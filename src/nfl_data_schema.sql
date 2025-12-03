-- ============================================================================
-- NFL Real-Time Data Pipeline - Database Schema
-- ============================================================================
-- Purpose: Store real-time NFL game data, predictions, and market correlations
-- Database: magnus (PostgreSQL)
-- Created: 2025-11-09
-- ============================================================================

-- ============================================================================
-- Table 1: nfl_games
-- ============================================================================
-- Stores NFL game schedules and current state
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) UNIQUE NOT NULL, -- External API game identifier
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,

    -- Teams
    home_team VARCHAR(50) NOT NULL,
    away_team VARCHAR(50) NOT NULL,
    home_team_abbr VARCHAR(10),
    away_team_abbr VARCHAR(10),

    -- Schedule
    game_time TIMESTAMP WITH TIME ZONE NOT NULL,
    venue VARCHAR(200),
    is_outdoor BOOLEAN DEFAULT true,

    -- Score (updated live)
    home_score INTEGER DEFAULT 0,
    away_score INTEGER DEFAULT 0,
    quarter INTEGER DEFAULT 0, -- 0=pregame, 1-4=quarters, 5=OT
    time_remaining VARCHAR(20), -- "12:34" format
    possession VARCHAR(10), -- Team abbreviation

    -- Game state
    game_status VARCHAR(20) DEFAULT 'scheduled', -- scheduled, live, halftime, final, postponed
    is_live BOOLEAN DEFAULT false,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,

    -- Betting lines (from external APIs)
    spread_home DECIMAL(4,1), -- -7.5 means home favored by 7.5
    spread_odds_home INTEGER, -- -110
    spread_odds_away INTEGER,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    over_under DECIMAL(4,1),
    over_odds INTEGER,
    under_odds INTEGER,

    -- Weather (for outdoor games)
    temperature INTEGER, -- Fahrenheit
    weather_condition VARCHAR(100),
    wind_speed INTEGER, -- MPH
    precipitation_chance INTEGER, -- Percentage

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_synced TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Raw API responses (for debugging)
    raw_game_data JSONB,
    raw_weather_data JSONB,

    CONSTRAINT chk_game_status CHECK (game_status IN ('scheduled', 'live', 'halftime', 'final', 'postponed', 'cancelled')),
    CONSTRAINT chk_quarter CHECK (quarter >= 0 AND quarter <= 5)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_nfl_games_status ON nfl_games(game_status);
CREATE INDEX IF NOT EXISTS idx_nfl_games_live ON nfl_games(is_live) WHERE is_live = true;
CREATE INDEX IF NOT EXISTS idx_nfl_games_time ON nfl_games(game_time);
CREATE INDEX IF NOT EXISTS idx_nfl_games_season_week ON nfl_games(season, week);
CREATE INDEX IF NOT EXISTS idx_nfl_games_teams ON nfl_games(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_nfl_games_upcoming ON nfl_games(game_time) WHERE game_status IN ('scheduled', 'live');

COMMENT ON TABLE nfl_games IS 'NFL game schedules and live scores';
COMMENT ON COLUMN nfl_games.game_id IS 'Unique identifier from external API';
COMMENT ON COLUMN nfl_games.is_live IS 'Quick filter for active games';

-- ============================================================================
-- Table 2: nfl_plays
-- ============================================================================
-- Stores play-by-play data for each game
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_plays (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES nfl_games(id) ON DELETE CASCADE,
    play_id VARCHAR(100) UNIQUE NOT NULL, -- External API play identifier

    -- Play sequence
    sequence_number INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    time_remaining VARCHAR(20),

    -- Play details
    play_type VARCHAR(50), -- pass, rush, punt, field_goal, kickoff, timeout, etc.
    description TEXT,
    down INTEGER, -- 1-4
    yards_to_go INTEGER,
    yard_line INTEGER, -- 0-100 (own goal to opponent goal)

    -- Play outcome
    yards_gained INTEGER,
    is_scoring_play BOOLEAN DEFAULT false,
    is_turnover BOOLEAN DEFAULT false,
    is_penalty BOOLEAN DEFAULT false,

    -- Teams/Players
    offense_team VARCHAR(10),
    defense_team VARCHAR(10),
    player_name VARCHAR(200),
    player_position VARCHAR(10),

    -- Scoring
    points_home INTEGER DEFAULT 0,
    points_away INTEGER DEFAULT 0,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_play_data JSONB,

    CONSTRAINT chk_down CHECK (down BETWEEN 1 AND 4 OR down IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_nfl_plays_game ON nfl_plays(game_id);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_sequence ON nfl_plays(game_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_type ON nfl_plays(play_type);
CREATE INDEX IF NOT EXISTS idx_nfl_plays_scoring ON nfl_plays(is_scoring_play) WHERE is_scoring_play = true;
CREATE INDEX IF NOT EXISTS idx_nfl_plays_created ON nfl_plays(created_at DESC);

COMMENT ON TABLE nfl_plays IS 'Play-by-play data for NFL games';

-- ============================================================================
-- Table 3: nfl_player_stats
-- ============================================================================
-- Stores live and aggregated player statistics
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_player_stats (
    id SERIAL PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES nfl_games(id) ON DELETE CASCADE,

    -- Player info
    player_name VARCHAR(200) NOT NULL,
    player_id VARCHAR(100), -- External API player ID
    team VARCHAR(50) NOT NULL,
    position VARCHAR(20) NOT NULL,

    -- Passing stats
    passing_attempts INTEGER DEFAULT 0,
    passing_completions INTEGER DEFAULT 0,
    passing_yards INTEGER DEFAULT 0,
    passing_touchdowns INTEGER DEFAULT 0,
    passing_interceptions INTEGER DEFAULT 0,

    -- Rushing stats
    rushing_attempts INTEGER DEFAULT 0,
    rushing_yards INTEGER DEFAULT 0,
    rushing_touchdowns INTEGER DEFAULT 0,

    -- Receiving stats
    receptions INTEGER DEFAULT 0,
    receiving_yards INTEGER DEFAULT 0,
    receiving_touchdowns INTEGER DEFAULT 0,
    targets INTEGER DEFAULT 0,

    -- Defense stats
    tackles INTEGER DEFAULT 0,
    sacks DECIMAL(3,1) DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    forced_fumbles INTEGER DEFAULT 0,

    -- Special teams
    field_goals_made INTEGER DEFAULT 0,
    field_goals_attempted INTEGER DEFAULT 0,
    extra_points_made INTEGER DEFAULT 0,
    extra_points_attempted INTEGER DEFAULT 0,

    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_stats_data JSONB,

    UNIQUE(game_id, player_id, player_name) -- One row per player per game
);

CREATE INDEX IF NOT EXISTS idx_nfl_player_stats_game ON nfl_player_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_nfl_player_stats_player ON nfl_player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_nfl_player_stats_position ON nfl_player_stats(position);

COMMENT ON TABLE nfl_player_stats IS 'Real-time player statistics for each game';

-- ============================================================================
-- Table 4: nfl_injuries
-- ============================================================================
-- Tracks player injury reports and updates
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_injuries (
    id SERIAL PRIMARY KEY,

    -- Player info
    player_name VARCHAR(200) NOT NULL,
    player_id VARCHAR(100),
    team VARCHAR(50) NOT NULL,
    position VARCHAR(20),

    -- Injury details
    injury_type VARCHAR(200), -- "Ankle", "Concussion", etc.
    injury_status VARCHAR(50) NOT NULL, -- Out, Questionable, Doubtful, Probable, IR
    description TEXT,

    -- Game reference (optional - injuries can be season-long)
    game_id INTEGER REFERENCES nfl_games(id) ON DELETE SET NULL,
    week INTEGER,

    -- Timeline
    reported_at TIMESTAMP WITH TIME ZONE NOT NULL,
    resolved_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,

    -- Metadata
    source VARCHAR(100), -- "ESPN", "NFL.com", "Team Report"
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_injury_data JSONB,

    CONSTRAINT chk_injury_status CHECK (injury_status IN ('Out', 'Questionable', 'Doubtful', 'Probable', 'IR', 'PUP', 'Cleared'))
);

CREATE INDEX IF NOT EXISTS idx_nfl_injuries_player ON nfl_injuries(player_id);
CREATE INDEX IF NOT EXISTS idx_nfl_injuries_team ON nfl_injuries(team);
CREATE INDEX IF NOT EXISTS idx_nfl_injuries_active ON nfl_injuries(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_nfl_injuries_game ON nfl_injuries(game_id);

COMMENT ON TABLE nfl_injuries IS 'Player injury tracking and status updates';

-- ============================================================================
-- Table 5: nfl_social_sentiment
-- ============================================================================
-- Stores aggregated social media sentiment for games and players
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_social_sentiment (
    id SERIAL PRIMARY KEY,
    game_id INTEGER REFERENCES nfl_games(id) ON DELETE CASCADE,

    -- Sentiment target
    entity_type VARCHAR(20) NOT NULL, -- 'game', 'team', 'player'
    entity_id VARCHAR(200) NOT NULL, -- game_id, team name, or player name

    -- Sentiment scores (-1.0 to 1.0, where negative = bearish, positive = bullish)
    sentiment_score DECIMAL(4,3), -- Average sentiment
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    total_mentions INTEGER DEFAULT 0,

    -- Sources
    twitter_mentions INTEGER DEFAULT 0,
    reddit_mentions INTEGER DEFAULT 0,

    -- Time window
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    raw_sentiment_data JSONB,

    CONSTRAINT chk_entity_type CHECK (entity_type IN ('game', 'team', 'player')),
    CONSTRAINT chk_sentiment_score CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_nfl_sentiment_game ON nfl_social_sentiment(game_id);
CREATE INDEX IF NOT EXISTS idx_nfl_sentiment_entity ON nfl_social_sentiment(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_nfl_sentiment_window ON nfl_social_sentiment(window_end DESC);

COMMENT ON TABLE nfl_social_sentiment IS 'Social media sentiment analysis for games and players';

-- ============================================================================
-- Table 6: nfl_kalshi_correlations
-- ============================================================================
-- Links NFL game events to Kalshi market price changes
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_kalshi_correlations (
    id SERIAL PRIMARY KEY,

    -- NFL event
    game_id INTEGER NOT NULL REFERENCES nfl_games(id) ON DELETE CASCADE,
    play_id INTEGER REFERENCES nfl_plays(id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL, -- 'touchdown', 'turnover', 'injury', 'quarter_end'
    event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Kalshi market
    kalshi_market_id INTEGER NOT NULL REFERENCES kalshi_markets(id) ON DELETE CASCADE,
    market_ticker VARCHAR(100) NOT NULL,

    -- Price movement
    price_before DECIMAL(5,4),
    price_after DECIMAL(5,4),
    price_change_pct DECIMAL(6,2), -- Percentage change

    -- Volume impact
    volume_before DECIMAL(15,2),
    volume_after DECIMAL(15,2),
    volume_spike_pct DECIMAL(6,2),

    -- Analysis
    correlation_strength DECIMAL(4,3), -- 0.0 to 1.0
    impact_level VARCHAR(20), -- low, medium, high, extreme

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT chk_impact_level CHECK (impact_level IN ('low', 'medium', 'high', 'extreme'))
);

CREATE INDEX IF NOT EXISTS idx_nfl_kalshi_corr_game ON nfl_kalshi_correlations(game_id);
CREATE INDEX IF NOT EXISTS idx_nfl_kalshi_corr_market ON nfl_kalshi_correlations(kalshi_market_id);
CREATE INDEX IF NOT EXISTS idx_nfl_kalshi_corr_event ON nfl_kalshi_correlations(event_type);
CREATE INDEX IF NOT EXISTS idx_nfl_kalshi_corr_timestamp ON nfl_kalshi_correlations(event_timestamp DESC);

COMMENT ON TABLE nfl_kalshi_correlations IS 'Correlation between NFL events and Kalshi market movements';

-- ============================================================================
-- Table 7: nfl_alert_triggers
-- ============================================================================
-- Configuration for alert conditions and user preferences
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_alert_triggers (
    id SERIAL PRIMARY KEY,

    -- Alert configuration
    alert_name VARCHAR(200) NOT NULL,
    alert_type VARCHAR(50) NOT NULL, -- 'score_change', 'price_movement', 'injury', 'sentiment_shift'
    is_active BOOLEAN DEFAULT true,

    -- Conditions (stored as JSONB for flexibility)
    trigger_conditions JSONB NOT NULL, -- {"min_price_change": 10, "teams": ["Chiefs", "Bills"]}

    -- Target filters
    teams_filter VARCHAR(100)[], -- NULL = all teams
    players_filter VARCHAR(200)[],

    -- Notification settings
    notification_channels VARCHAR(50)[] DEFAULT ARRAY['telegram'], -- telegram, email, sms
    notification_priority VARCHAR(20) DEFAULT 'medium', -- low, medium, high, urgent

    -- Rate limiting
    cooldown_minutes INTEGER DEFAULT 5, -- Minimum time between alerts of same type
    max_alerts_per_day INTEGER DEFAULT 50,

    -- Metadata
    created_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_triggered TIMESTAMP WITH TIME ZONE,
    trigger_count INTEGER DEFAULT 0,

    CONSTRAINT chk_notification_priority CHECK (notification_priority IN ('low', 'medium', 'high', 'urgent'))
);

CREATE INDEX IF NOT EXISTS idx_nfl_alert_triggers_active ON nfl_alert_triggers(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_nfl_alert_triggers_type ON nfl_alert_triggers(alert_type);

COMMENT ON TABLE nfl_alert_triggers IS 'User-defined alert conditions and preferences';

-- ============================================================================
-- Table 8: nfl_alert_history
-- ============================================================================
-- Log of all alerts sent
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_alert_history (
    id SERIAL PRIMARY KEY,

    -- Alert reference
    trigger_id INTEGER REFERENCES nfl_alert_triggers(id) ON DELETE SET NULL,
    alert_type VARCHAR(50) NOT NULL,

    -- Alert content
    subject VARCHAR(500) NOT NULL,
    message TEXT NOT NULL,

    -- Context
    game_id INTEGER REFERENCES nfl_games(id) ON DELETE SET NULL,
    play_id INTEGER REFERENCES nfl_plays(id) ON DELETE SET NULL,
    kalshi_market_id INTEGER REFERENCES kalshi_markets(id) ON DELETE SET NULL,

    -- Delivery
    notification_channel VARCHAR(50) NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    delivery_status VARCHAR(20) DEFAULT 'pending', -- pending, sent, failed
    telegram_message_id VARCHAR(100),
    error_message TEXT,

    -- Metadata
    alert_data JSONB, -- Full alert payload

    CONSTRAINT chk_delivery_status CHECK (delivery_status IN ('pending', 'sent', 'failed', 'queued'))
);

CREATE INDEX IF NOT EXISTS idx_nfl_alert_history_sent ON nfl_alert_history(sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_nfl_alert_history_game ON nfl_alert_history(game_id);
CREATE INDEX IF NOT EXISTS idx_nfl_alert_history_status ON nfl_alert_history(delivery_status);
CREATE INDEX IF NOT EXISTS idx_nfl_alert_history_type ON nfl_alert_history(alert_type);

COMMENT ON TABLE nfl_alert_history IS 'Historical record of all alerts sent';

-- ============================================================================
-- Table 9: nfl_data_sync_log
-- ============================================================================
-- Tracks all data sync operations and performance
-- ============================================================================

CREATE TABLE IF NOT EXISTS nfl_data_sync_log (
    id SERIAL PRIMARY KEY,

    -- Sync details
    sync_type VARCHAR(50) NOT NULL, -- 'scores', 'plays', 'weather', 'injuries', 'kalshi_prices'
    sync_scope VARCHAR(100), -- 'all_games', 'live_games', specific game_id

    -- Performance metrics
    records_fetched INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    duration_ms INTEGER, -- Milliseconds

    -- API usage
    api_calls_made INTEGER DEFAULT 0,
    api_errors INTEGER DEFAULT 0,
    rate_limit_hit BOOLEAN DEFAULT false,

    -- Status
    sync_status VARCHAR(20) DEFAULT 'running', -- running, completed, failed, partial
    error_message TEXT,

    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    sync_metadata JSONB, -- Additional context

    CONSTRAINT chk_sync_status CHECK (sync_status IN ('running', 'completed', 'failed', 'partial'))
);

CREATE INDEX IF NOT EXISTS idx_nfl_sync_log_type ON nfl_data_sync_log(sync_type);
CREATE INDEX IF NOT EXISTS idx_nfl_sync_log_started ON nfl_data_sync_log(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_nfl_sync_log_status ON nfl_data_sync_log(sync_status);

COMMENT ON TABLE nfl_data_sync_log IS 'Performance tracking for all data sync operations';

-- ============================================================================
-- VIEWS - Pre-computed queries for common use cases
-- ============================================================================

-- View: Live games with latest scores
CREATE OR REPLACE VIEW v_nfl_live_games AS
SELECT
    g.id,
    g.game_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.quarter,
    g.time_remaining,
    g.possession,
    g.spread_home,
    g.over_under,
    g.venue,
    g.temperature,
    g.weather_condition,
    g.last_updated,
    -- Calculate score differential
    g.home_score - g.away_score as score_diff,
    -- Link to Kalshi markets
    (SELECT COUNT(*) FROM kalshi_markets km
     WHERE (km.home_team = g.home_team OR km.away_team = g.away_team)
       AND km.status = 'open') as active_markets_count
FROM nfl_games g
WHERE g.is_live = true
ORDER BY g.game_time;

COMMENT ON VIEW v_nfl_live_games IS 'Currently live NFL games with scores and market data';

-- View: Game predictions vs actual results
CREATE OR REPLACE VIEW v_nfl_prediction_accuracy AS
SELECT
    g.id,
    g.game_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.spread_home as predicted_spread,
    (g.home_score - g.away_score) as actual_spread,
    ABS(g.spread_home - (g.home_score - g.away_score)) as spread_error,
    g.over_under as predicted_total,
    (g.home_score + g.away_score) as actual_total,
    ABS(g.over_under - (g.home_score + g.away_score)) as total_error,
    g.game_status,
    g.finished_at
FROM nfl_games g
WHERE g.game_status = 'final'
  AND g.spread_home IS NOT NULL
ORDER BY g.finished_at DESC;

COMMENT ON VIEW v_nfl_prediction_accuracy IS 'Compare betting lines to actual game results';

-- View: High-value Kalshi opportunities based on live game data
CREATE OR REPLACE VIEW v_nfl_kalshi_opportunities AS
SELECT
    g.game_id,
    g.home_team,
    g.away_team,
    g.home_score,
    g.away_score,
    g.quarter,
    g.time_remaining,
    km.ticker,
    km.title,
    km.yes_price,
    km.volume,
    kp.predicted_outcome,
    kp.confidence_score,
    kp.edge_percentage,
    kp.recommended_action,
    -- Calculate live edge based on game state
    CASE
        WHEN g.quarter >= 4 AND g.home_score > g.away_score + 7 THEN 'home_lock'
        WHEN g.quarter >= 4 AND g.away_score > g.home_score + 7 THEN 'away_lock'
        WHEN g.quarter <= 2 THEN 'early'
        ELSE 'competitive'
    END as game_state,
    g.last_updated as game_last_updated,
    km.last_updated as market_last_updated
FROM nfl_games g
INNER JOIN kalshi_markets km ON (
    km.home_team = g.home_team OR km.away_team = g.away_team
)
LEFT JOIN kalshi_predictions kp ON km.id = kp.market_id
WHERE g.is_live = true
  AND km.status = 'open'
  AND kp.edge_percentage > 5 -- Only high-edge opportunities
ORDER BY kp.edge_percentage DESC NULLS LAST;

COMMENT ON VIEW v_nfl_kalshi_opportunities IS 'Live NFL games with high-value Kalshi betting opportunities';

-- View: Recent significant plays
CREATE OR REPLACE VIEW v_nfl_significant_plays AS
SELECT
    p.id,
    p.game_id,
    g.home_team,
    g.away_team,
    p.quarter,
    p.time_remaining,
    p.play_type,
    p.description,
    p.yards_gained,
    p.is_scoring_play,
    p.is_turnover,
    p.offense_team,
    p.player_name,
    p.points_home,
    p.points_away,
    p.created_at
FROM nfl_plays p
INNER JOIN nfl_games g ON p.game_id = g.id
WHERE (p.is_scoring_play = true OR p.is_turnover = true OR ABS(p.yards_gained) >= 20)
  AND p.created_at > NOW() - INTERVAL '2 hours'
ORDER BY p.created_at DESC
LIMIT 100;

COMMENT ON VIEW v_nfl_significant_plays IS 'Recent high-impact plays across all games';

-- ============================================================================
-- FUNCTIONS - Utility functions for common operations
-- ============================================================================

-- Function: Calculate win probability based on current score and time
CREATE OR REPLACE FUNCTION calculate_win_probability(
    score_diff INTEGER,
    quarter INTEGER,
    time_remaining VARCHAR(20)
)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    time_left_seconds INTEGER;
    time_parts TEXT[];
    minutes INTEGER;
    seconds INTEGER;
    base_prob DECIMAL(5,2);
    time_factor DECIMAL(5,2);
BEGIN
    -- Parse time remaining (MM:SS format)
    time_parts := string_to_array(time_remaining, ':');
    minutes := COALESCE(time_parts[1]::INTEGER, 0);
    seconds := COALESCE(time_parts[2]::INTEGER, 0);

    -- Calculate total seconds remaining (including future quarters)
    time_left_seconds := (4 - quarter) * 900 + (minutes * 60) + seconds;

    -- Base probability from score differential (simplified model)
    -- Each point is worth roughly 2% win probability
    base_prob := 50.0 + (score_diff * 2.0);

    -- Time factor: less time = more certainty
    -- At quarter 4, 2 minutes left, multiply certainty by ~1.5x
    time_factor := 1.0 + ((3600 - time_left_seconds)::DECIMAL / 3600.0) * 0.5;

    -- Adjust base probability by time factor
    base_prob := 50.0 + ((base_prob - 50.0) * time_factor);

    -- Clamp between 0.1% and 99.9%
    RETURN GREATEST(0.1, LEAST(99.9, base_prob));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_win_probability IS 'Estimate win probability based on score and time remaining';

-- ============================================================================
-- TRIGGERS - Automatic timestamp updates
-- ============================================================================

-- Update last_updated timestamp on changes
CREATE OR REPLACE FUNCTION update_last_updated_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_nfl_games_timestamp ON nfl_games;
CREATE TRIGGER update_nfl_games_timestamp
    BEFORE UPDATE ON nfl_games
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_timestamp();

DROP TRIGGER IF EXISTS update_nfl_player_stats_timestamp ON nfl_player_stats;
CREATE TRIGGER update_nfl_player_stats_timestamp
    BEFORE UPDATE ON nfl_player_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_timestamp();

DROP TRIGGER IF EXISTS update_nfl_injuries_timestamp ON nfl_injuries;
CREATE TRIGGER update_nfl_injuries_timestamp
    BEFORE UPDATE ON nfl_injuries
    FOR EACH ROW
    EXECUTE FUNCTION update_last_updated_timestamp();

-- ============================================================================
-- SAMPLE QUERIES - Common use cases
-- ============================================================================

-- Get all live games with latest updates
-- SELECT * FROM v_nfl_live_games;

-- Get significant plays in last hour
-- SELECT * FROM v_nfl_significant_plays WHERE created_at > NOW() - INTERVAL '1 hour';

-- Get top Kalshi opportunities for live games
-- SELECT * FROM v_nfl_kalshi_opportunities LIMIT 10;

-- Calculate win probability for a game
-- SELECT calculate_win_probability(7, 4, '05:30'); -- Up by 7, Q4, 5:30 left

-- Get games with Kalshi price movements
-- SELECT * FROM nfl_kalshi_correlations WHERE price_change_pct > 10 ORDER BY event_timestamp DESC;

-- Find all active injury reports for a team
-- SELECT * FROM nfl_injuries WHERE team = 'Kansas City Chiefs' AND is_active = true;

-- Get social sentiment for upcoming game
-- SELECT * FROM nfl_social_sentiment WHERE game_id = 123 ORDER BY window_end DESC LIMIT 1;
