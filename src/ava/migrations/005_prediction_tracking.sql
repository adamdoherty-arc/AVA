-- Migration 005: Prediction Tracking and Odds History
-- Enables AI accuracy tracking, odds movement charts, and personalized recommendations

-- Prediction results tracking (for accuracy metrics)
CREATE TABLE IF NOT EXISTS prediction_results (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(100) UNIQUE,
    game_id VARCHAR(50) NOT NULL,
    sport VARCHAR(20) NOT NULL,

    -- Prediction details
    predicted_winner VARCHAR(100),
    predicted_probability DECIMAL(5,4),  -- 0.0000 to 0.9999
    predicted_spread DECIMAL(4,1),
    predicted_total DECIMAL(5,1),

    -- Actual outcome
    actual_winner VARCHAR(100),
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    was_correct BOOLEAN,

    -- Metadata
    prediction_timestamp TIMESTAMP DEFAULT NOW(),
    game_completed_at TIMESTAMP,
    model_version VARCHAR(20) DEFAULT 'v1.0',
    confidence_tier VARCHAR(20),  -- 'high', 'medium', 'low'

    -- Indexes for fast querying
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_results_sport ON prediction_results(sport);
CREATE INDEX IF NOT EXISTS idx_prediction_results_game_id ON prediction_results(game_id);
CREATE INDEX IF NOT EXISTS idx_prediction_results_timestamp ON prediction_results(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_prediction_results_correct ON prediction_results(was_correct);

-- Odds history for movement tracking
CREATE TABLE IF NOT EXISTS odds_history (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    sport VARCHAR(20) NOT NULL,
    source VARCHAR(30) NOT NULL,  -- 'kalshi', 'draftkings', 'fanduel', etc.

    -- Moneyline odds (American format)
    home_odds INTEGER,
    away_odds INTEGER,

    -- Spread betting
    spread DECIMAL(4,1),
    spread_home_odds INTEGER DEFAULT -110,
    spread_away_odds INTEGER DEFAULT -110,

    -- Totals (over/under)
    total DECIMAL(5,1),
    over_odds INTEGER DEFAULT -110,
    under_odds INTEGER DEFAULT -110,

    -- Implied probabilities
    home_implied_prob DECIMAL(5,4),
    away_implied_prob DECIMAL(5,4),

    -- Timestamp
    recorded_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_odds_history_game ON odds_history(game_id);
CREATE INDEX IF NOT EXISTS idx_odds_history_time ON odds_history(recorded_at);
CREATE INDEX IF NOT EXISTS idx_odds_history_source ON odds_history(source);

-- User betting profile (for personalized recommendations)
CREATE TABLE IF NOT EXISTS user_betting_profile (
    user_id VARCHAR(50) PRIMARY KEY,

    -- Preferences
    preferred_sports TEXT[] DEFAULT ARRAY['NFL', 'NBA'],
    preferred_bet_types TEXT[] DEFAULT ARRAY['moneyline', 'spread'],
    risk_tolerance VARCHAR(20) DEFAULT 'medium',  -- 'conservative', 'medium', 'aggressive'

    -- Bankroll management
    bankroll DECIMAL(12,2) DEFAULT 1000.00,
    kelly_fraction DECIMAL(3,2) DEFAULT 0.25,  -- Fraction of Kelly to use
    max_bet_size DECIMAL(12,2) DEFAULT 100.00,

    -- Tracking
    total_bets INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_profit DECIMAL(12,2) DEFAULT 0.00,

    -- Settings
    notifications_enabled BOOLEAN DEFAULT TRUE,
    confidence_threshold DECIMAL(3,2) DEFAULT 0.60,  -- Min confidence for alerts

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User bet history
CREATE TABLE IF NOT EXISTS user_bets (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES user_betting_profile(user_id),
    prediction_id VARCHAR(100) REFERENCES prediction_results(prediction_id),
    game_id VARCHAR(50) NOT NULL,
    sport VARCHAR(20) NOT NULL,

    -- Bet details
    bet_type VARCHAR(30),  -- 'moneyline', 'spread', 'total', 'prop'
    bet_side VARCHAR(50),  -- 'home', 'away', 'over', 'under'
    bet_amount DECIMAL(12,2),
    odds INTEGER,

    -- Outcome
    potential_payout DECIMAL(12,2),
    actual_payout DECIMAL(12,2),
    is_winner BOOLEAN,

    -- AI recommendation context
    ai_confidence DECIMAL(5,4),
    ai_recommended BOOLEAN DEFAULT FALSE,

    placed_at TIMESTAMP DEFAULT NOW(),
    settled_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_bets_user ON user_bets(user_id);
CREATE INDEX IF NOT EXISTS idx_user_bets_game ON user_bets(game_id);

-- AI prediction snapshots (for live adjustments)
CREATE TABLE IF NOT EXISTS live_prediction_snapshots (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(50) NOT NULL,
    sport VARCHAR(20) NOT NULL,

    -- Pre-game baseline
    pregame_home_prob DECIMAL(5,4),
    pregame_away_prob DECIMAL(5,4),

    -- Current live adjustment
    live_home_prob DECIMAL(5,4),
    live_away_prob DECIMAL(5,4),

    -- Game state at snapshot
    home_score INTEGER,
    away_score INTEGER,
    quarter_period INTEGER,
    time_remaining VARCHAR(10),
    possession VARCHAR(10),

    -- Momentum indicators
    momentum_score DECIMAL(4,2),  -- -1.0 to 1.0 (negative = away, positive = home)
    scoring_run VARCHAR(20),  -- e.g., "10-0 home run"

    -- Metadata
    snapshot_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_snapshots_game ON live_prediction_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_live_snapshots_time ON live_prediction_snapshots(snapshot_at);

-- Model performance metrics (aggregate statistics)
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    sport VARCHAR(20) NOT NULL,
    model_version VARCHAR(20) NOT NULL,

    -- Time period
    period_start DATE,
    period_end DATE,
    period_type VARCHAR(20),  -- 'daily', 'weekly', 'monthly', 'season'

    -- Accuracy metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(5,4),

    -- Calibration metrics
    brier_score DECIMAL(6,5),  -- Lower is better, 0 = perfect
    log_loss DECIMAL(8,5),

    -- ROI metrics (if following recommendations)
    theoretical_roi DECIMAL(8,4),  -- % return if betting $1 on every pick
    high_conf_roi DECIMAL(8,4),  -- ROI on high-confidence picks only

    -- By confidence tier
    high_conf_total INTEGER DEFAULT 0,
    high_conf_correct INTEGER DEFAULT 0,
    med_conf_total INTEGER DEFAULT 0,
    med_conf_correct INTEGER DEFAULT 0,
    low_conf_total INTEGER DEFAULT 0,
    low_conf_correct INTEGER DEFAULT 0,

    calculated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_perf_unique ON model_performance(sport, model_version, period_start, period_end);

-- Function to calculate rolling accuracy
CREATE OR REPLACE FUNCTION calculate_rolling_accuracy(
    p_sport VARCHAR(20),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    total_predictions BIGINT,
    correct_predictions BIGINT,
    accuracy_rate DECIMAL,
    avg_confidence DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_predictions,
        SUM(CASE WHEN was_correct THEN 1 ELSE 0 END)::BIGINT as correct_predictions,
        ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END), 4) as accuracy_rate,
        ROUND(AVG(predicted_probability), 4) as avg_confidence
    FROM prediction_results
    WHERE sport = p_sport
      AND game_completed_at IS NOT NULL
      AND game_completed_at >= NOW() - (p_days || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql;

-- Comment for documentation
COMMENT ON TABLE prediction_results IS 'Stores all AI predictions and their outcomes for accuracy tracking';
COMMENT ON TABLE odds_history IS 'Historical odds data for movement charts and trend analysis';
COMMENT ON TABLE live_prediction_snapshots IS 'Real-time prediction adjustments during live games';
COMMENT ON TABLE model_performance IS 'Aggregate performance metrics by time period and sport';
