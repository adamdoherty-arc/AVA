-- ============================================================================
-- Database Optimization - Add Missing Indexes
-- ============================================================================
-- Purpose: Add missing indexes identified during optimization review
-- Created: 2025-11-25
-- ============================================================================

-- NFL Data Optimizations
-- ============================================================================

-- Filter player stats by team within a game (common for box scores)
CREATE INDEX IF NOT EXISTS idx_nfl_player_stats_game_team ON nfl_player_stats(game_id, team);

-- Filter plays by quarter within a game
CREATE INDEX IF NOT EXISTS idx_nfl_plays_game_quarter ON nfl_plays(game_id, quarter);

-- Filter high-impact correlations
CREATE INDEX IF NOT EXISTS idx_nfl_kalshi_corr_impact ON nfl_kalshi_correlations(game_id, impact_level);

-- Xtrades Data Optimizations
-- ============================================================================

-- Time-series analysis for specific tickers
CREATE INDEX IF NOT EXISTS idx_xtrades_trades_ticker_date ON xtrades_trades(ticker, entry_date DESC);

-- Find high-performing trades
CREATE INDEX IF NOT EXISTS idx_xtrades_trades_pnl_percent ON xtrades_trades(pnl_percent DESC);

-- Filter trades by strategy and outcome
CREATE INDEX IF NOT EXISTS idx_xtrades_trades_strategy_pnl ON xtrades_trades(strategy, pnl_percent);

-- Success Message
SELECT 'Missing indexes added successfully' as status;
