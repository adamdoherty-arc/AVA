-- ============================================================================
-- Performance Optimization - Database Indexes Migration
-- ============================================================================
-- Created: 2025-11-20
-- Purpose: Add strategic indexes to improve query performance across platform
-- Expected Impact: 2-5x faster queries on frequently accessed tables
--
-- IMPORTANT: Run during low-traffic period
-- Estimated execution time: 2-5 minutes depending on data volume
-- ============================================================================

-- Start transaction for rollback capability
BEGIN;

-- ============================================================================
-- XTRADES ALERTS INDEXES
-- ============================================================================

-- Index for profile + status queries (most common filter combination)
CREATE INDEX IF NOT EXISTS idx_xtrades_profile_status
ON xtrades_alerts(profile_id, status)
WHERE status IN ('open', 'closed', 'expired');

-- Index for ticker lookups with date ordering
CREATE INDEX IF NOT EXISTS idx_xtrades_ticker_date
ON xtrades_alerts(ticker, alert_timestamp DESC)
WHERE ticker IS NOT NULL;

-- Index for P/L analysis queries
CREATE INDEX IF NOT EXISTS idx_xtrades_pnl
ON xtrades_alerts(pnl DESC)
WHERE pnl IS NOT NULL;

-- Index for strategy filtering
CREATE INDEX IF NOT EXISTS idx_xtrades_strategy
ON xtrades_alerts(strategy)
WHERE strategy IS NOT NULL;

-- Index for expiration date queries
CREATE INDEX IF NOT EXISTS idx_xtrades_expiration
ON xtrades_alerts(expiration_date)
WHERE expiration_date IS NOT NULL;

-- ============================================================================
-- POSITIONS/CLOSED TRADES INDEXES
-- ============================================================================

-- Index for recent closed trades (most frequently queried)
CREATE INDEX IF NOT EXISTS idx_positions_close_date
ON closed_trades(close_date DESC)
WHERE close_date IS NOT NULL;

-- Index for symbol + date queries
CREATE INDEX IF NOT EXISTS idx_positions_symbol_date
ON closed_trades(symbol, close_date DESC)
WHERE symbol IS NOT NULL;

-- Index for P/L queries
CREATE INDEX IF NOT EXISTS idx_positions_pnl
ON closed_trades(pnl DESC)
WHERE pnl IS NOT NULL;

-- Index for strategy analysis
CREATE INDEX IF NOT EXISTS idx_positions_strategy
ON closed_trades(strategy)
WHERE strategy IS NOT NULL;

-- Index for open date queries
CREATE INDEX IF NOT EXISTS idx_positions_open_date
ON closed_trades(open_date DESC)
WHERE open_date IS NOT NULL;

-- ============================================================================
-- OPTIONS DATA INDEXES
-- ============================================================================

-- Index for DTE + Delta range queries (wheel strategy sweet spot)
CREATE INDEX IF NOT EXISTS idx_options_dte_delta
ON options_data(dte, delta)
WHERE dte BETWEEN 0 AND 45 AND delta BETWEEN -0.5 AND 0;

-- Index for premium-based queries
CREATE INDEX IF NOT EXISTS idx_options_premium
ON options_data(premium_collected DESC)
WHERE premium_collected > 0;

-- Index for symbol + expiration lookups
CREATE INDEX IF NOT EXISTS idx_options_symbol_expiration
ON options_data(symbol, expiration_date)
WHERE symbol IS NOT NULL AND expiration_date IS NOT NULL;

-- Index for strike price queries
CREATE INDEX IF NOT EXISTS idx_options_strike
ON options_data(strike_price)
WHERE strike_price > 0;

-- Index for IV rank queries
CREATE INDEX IF NOT EXISTS idx_options_iv_rank
ON options_data(iv_rank DESC)
WHERE iv_rank IS NOT NULL;

-- ============================================================================
-- KALSHI MARKETS INDEXES
-- ============================================================================

-- Index for sport + event date queries (main page filter)
CREATE INDEX IF NOT EXISTS idx_kalshi_sport_date
ON kalshi_markets(sport, event_date DESC)
WHERE sport IS NOT NULL;

-- Index for market status queries
CREATE INDEX IF NOT EXISTS idx_kalshi_status
ON kalshi_markets(status)
WHERE status IN ('open', 'closed', 'settled');

-- Index for ticker lookups
CREATE INDEX IF NOT EXISTS idx_kalshi_ticker
ON kalshi_markets(ticker)
WHERE ticker IS NOT NULL;

-- Index for event title searches
CREATE INDEX IF NOT EXISTS idx_kalshi_event_title
ON kalshi_markets USING gin(to_tsvector('english', event_title))
WHERE event_title IS NOT NULL;

-- ============================================================================
-- KALSHI PREDICTIONS INDEXES
-- ============================================================================

-- Index for AI confidence queries (high-confidence picks)
CREATE INDEX IF NOT EXISTS idx_kalshi_predictions_confidence
ON kalshi_predictions(ai_confidence DESC)
WHERE ai_confidence > 0.7;

-- Index for market + predictor lookups
CREATE INDEX IF NOT EXISTS idx_kalshi_predictions_market
ON kalshi_predictions(market_ticker, predictor_type);

-- Index for prediction date ordering
CREATE INDEX IF NOT EXISTS idx_kalshi_predictions_date
ON kalshi_predictions(prediction_date DESC);

-- ============================================================================
-- TRADINGVIEW WATCHLISTS INDEXES
-- ============================================================================

-- Index for watchlist name lookups
CREATE INDEX IF NOT EXISTS idx_tv_watchlist_name
ON tradingview_watchlists(watchlist_name);

-- Index for symbol searches
CREATE INDEX IF NOT EXISTS idx_tv_symbol
ON tradingview_watchlist_stocks(symbol);

-- Index for last sync tracking
CREATE INDEX IF NOT EXISTS idx_tv_last_sync
ON tradingview_watchlists(last_sync DESC);

-- ============================================================================
-- DATABASE STOCKS INDEXES
-- ============================================================================

-- Index for sector queries
CREATE INDEX IF NOT EXISTS idx_stocks_sector
ON database_stocks(sector)
WHERE sector IS NOT NULL;

-- Index for market cap filtering
CREATE INDEX IF NOT EXISTS idx_stocks_market_cap
ON database_stocks(market_cap DESC)
WHERE market_cap > 0;

-- Index for symbol lookups (if not already primary key)
CREATE INDEX IF NOT EXISTS idx_stocks_symbol
ON database_stocks(symbol);

-- ============================================================================
-- SUPPLY/DEMAND ZONES INDEXES
-- ============================================================================

-- Index for active zones by symbol
CREATE INDEX IF NOT EXISTS idx_zones_symbol_active
ON supply_demand_zones(symbol, is_active)
WHERE is_active = true;

-- Index for zone type filtering
CREATE INDEX IF NOT EXISTS idx_zones_type
ON supply_demand_zones(zone_type)
WHERE zone_type IN ('demand', 'supply');

-- Index for strength-based queries
CREATE INDEX IF NOT EXISTS idx_zones_strength
ON supply_demand_zones(strength DESC)
WHERE strength > 0;

-- Index for created date tracking
CREATE INDEX IF NOT EXISTS idx_zones_created
ON supply_demand_zones(created_at DESC);

-- ============================================================================
-- QA SYSTEM INDEXES (if tables exist)
-- ============================================================================

-- Index for task status queries
CREATE INDEX IF NOT EXISTS idx_qa_tasks_status
ON qa_tasks(status)
WHERE status IN ('pending', 'in_progress', 'completed', 'failed');

-- Index for agent signoffs
CREATE INDEX IF NOT EXISTS idx_qa_signoffs_agent
ON qa_agent_signoffs(agent_type, status);

-- Index for review date tracking
CREATE INDEX IF NOT EXISTS idx_qa_reviews_date
ON qa_reviews(review_date DESC);

-- ============================================================================
-- ANALYTICS INDEXES
-- ============================================================================

-- Index for performance tracking by date
CREATE INDEX IF NOT EXISTS idx_analytics_date
ON performance_analytics(analytics_date DESC);

-- Index for user activity tracking
CREATE INDEX IF NOT EXISTS idx_analytics_user
ON user_activity(user_id, activity_date DESC);

-- ============================================================================
-- SPORTS GAMES INDEXES (Added 2025-11-29)
-- ============================================================================
-- Critical for optimized UNION queries and batch odds sync

-- NFL Games
CREATE INDEX IF NOT EXISTS idx_nfl_games_status_time
ON nfl_games(game_status, game_time)
WHERE game_status IN ('scheduled', 'live');

CREATE INDEX IF NOT EXISTS idx_nfl_games_live
ON nfl_games(is_live)
WHERE is_live = true;

CREATE INDEX IF NOT EXISTS idx_nfl_games_teams
ON nfl_games(LOWER(home_team), LOWER(away_team));

-- NBA Games
CREATE INDEX IF NOT EXISTS idx_nba_games_status_time
ON nba_games(game_status, game_time)
WHERE game_status IN ('scheduled', 'live');

CREATE INDEX IF NOT EXISTS idx_nba_games_live
ON nba_games(is_live)
WHERE is_live = true;

CREATE INDEX IF NOT EXISTS idx_nba_games_teams
ON nba_games(LOWER(home_team), LOWER(away_team));

-- NCAA Football Games
CREATE INDEX IF NOT EXISTS idx_ncaaf_games_status_time
ON ncaa_football_games(game_status, game_time)
WHERE game_status IN ('scheduled', 'live');

CREATE INDEX IF NOT EXISTS idx_ncaaf_games_live
ON ncaa_football_games(is_live)
WHERE is_live = true;

CREATE INDEX IF NOT EXISTS idx_ncaaf_games_teams
ON ncaa_football_games(LOWER(home_team), LOWER(away_team));

-- NCAA Basketball Games
CREATE INDEX IF NOT EXISTS idx_ncaab_games_status_time
ON ncaa_basketball_games(game_status, game_time)
WHERE game_status IN ('scheduled', 'live');

CREATE INDEX IF NOT EXISTS idx_ncaab_games_live
ON ncaa_basketball_games(is_live)
WHERE is_live = true;

CREATE INDEX IF NOT EXISTS idx_ncaab_games_teams
ON ncaa_basketball_games(LOWER(home_team), LOWER(away_team));

-- ============================================================================
-- LIVE ODDS & AI RECOMMENDATIONS INDEXES (Added 2025-11-29)
-- ============================================================================
-- Critical for sports betting AI and odds analysis

-- Live Odds Snapshots (time-series data)
CREATE INDEX IF NOT EXISTS idx_odds_snapshots_sport_game
ON live_odds_snapshots(sport, game_id);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_time
ON live_odds_snapshots(snapshot_time DESC);

CREATE INDEX IF NOT EXISTS idx_odds_snapshots_game_time
ON live_odds_snapshots(game_id, snapshot_time DESC);

-- AI Betting Recommendations
CREATE INDEX IF NOT EXISTS idx_ai_recs_sport_game
ON ai_betting_recommendations(sport, game_id);

CREATE INDEX IF NOT EXISTS idx_ai_recs_unsettled
ON ai_betting_recommendations(is_settled)
WHERE is_settled = false;

CREATE INDEX IF NOT EXISTS idx_ai_recs_confidence
ON ai_betting_recommendations(confidence DESC)
WHERE confidence >= 70;

CREATE INDEX IF NOT EXISTS idx_ai_recs_created
ON ai_betting_recommendations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ai_recs_result
ON ai_betting_recommendations(result)
WHERE is_settled = true;

-- ============================================================================
-- PREDICTION TRACKING INDEXES (Added 2025-11-29)
-- ============================================================================
-- For model performance analysis and outcome tracking

CREATE INDEX IF NOT EXISTS idx_prediction_results_game
ON prediction_results(sport, game_id)
WHERE sport IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_prediction_results_date
ON prediction_results(prediction_date DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_results_outcome
ON prediction_results(was_correct, prediction_date DESC)
WHERE was_correct IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_model_performance_model
ON model_performance(model_name, evaluation_date DESC);

-- ============================================================================
-- DASHBOARD & PORTFOLIO INDEXES (Added 2025-11-29)
-- ============================================================================
-- Critical for high-traffic dashboard queries

-- Trade History (Dashboard loads this frequently)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_history_executed_at_desc
ON trade_history (executed_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_history_symbol_date
ON trade_history (symbol, executed_at DESC)
WHERE symbol IS NOT NULL;

-- Alerts (Dashboard refreshes frequently)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_active_recent
ON alerts (is_dismissed, created_at DESC, priority DESC)
WHERE is_dismissed = FALSE;

-- Portfolio History (Chart data)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_history_date_range
ON portfolio_history (date DESC, portfolio_value)
WHERE date >= CURRENT_DATE - INTERVAL '365 days';

-- Trade Journal (Frequently accessed)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_journal_entry_date
ON trade_journal (entry_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trade_journal_symbol
ON trade_journal (symbol, entry_date DESC)
WHERE symbol IS NOT NULL;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Display all newly created indexes
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Performance Indexes Created Successfully!';
    RAISE NOTICE '============================================';

    -- Count indexes per table
    RAISE NOTICE 'Indexes created:';
    RAISE NOTICE '- XTrades: %', (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'xtrades_alerts');
    RAISE NOTICE '- Positions: %', (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'closed_trades');
    RAISE NOTICE '- Options: %', (SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'options_data');
    RAISE NOTICE '- Kalshi: %', (SELECT COUNT(*) FROM pg_indexes WHERE tablename LIKE 'kalshi%');
    RAISE NOTICE '============================================';
END $$;

-- Commit transaction
COMMIT;

-- ============================================================================
-- PERFORMANCE TESTING QUERIES
-- ============================================================================

-- Test queries to verify index usage (run with EXPLAIN ANALYZE)

/*
-- Test 1: XTrades profile + status query
EXPLAIN ANALYZE
SELECT * FROM xtrades_alerts
WHERE profile_id = 1 AND status = 'open'
ORDER BY alert_timestamp DESC
LIMIT 100;

-- Test 2: Options DTE + Delta query
EXPLAIN ANALYZE
SELECT * FROM options_data
WHERE dte BETWEEN 7 AND 30
  AND delta BETWEEN -0.3 AND -0.2
ORDER BY premium_collected DESC
LIMIT 50;

-- Test 3: Kalshi sport + date query
EXPLAIN ANALYZE
SELECT * FROM kalshi_markets
WHERE sport = 'NFL'
  AND event_date >= CURRENT_DATE
ORDER BY event_date
LIMIT 100;

-- Test 4: Closed trades P/L analysis
EXPLAIN ANALYZE
SELECT symbol, SUM(pnl) as total_pnl, COUNT(*) as trades
FROM closed_trades
WHERE close_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY symbol
ORDER BY total_pnl DESC
LIMIT 20;
*/

-- ============================================================================
-- MAINTENANCE RECOMMENDATIONS
-- ============================================================================

/*
Run these commands periodically to maintain index performance:

-- Update statistics for query planner
ANALYZE xtrades_alerts;
ANALYZE closed_trades;
ANALYZE options_data;
ANALYZE kalshi_markets;
ANALYZE kalshi_predictions;

-- Vacuum to reclaim space and update statistics
VACUUM ANALYZE xtrades_alerts;
VACUUM ANALYZE closed_trades;
VACUUM ANALYZE options_data;

-- Check index usage statistics
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;

-- Check for unused indexes (idx_scan = 0)
SELECT schemaname, tablename, indexname
FROM pg_stat_user_indexes
WHERE schemaname = 'public' AND idx_scan = 0
ORDER BY tablename, indexname;
*/

-- ============================================================================
-- ROLLBACK SCRIPT (if needed)
-- ============================================================================

/*
If you need to rollback these indexes:

BEGIN;

-- XTrades
DROP INDEX IF EXISTS idx_xtrades_profile_status;
DROP INDEX IF EXISTS idx_xtrades_ticker_date;
DROP INDEX IF EXISTS idx_xtrades_pnl;
DROP INDEX IF EXISTS idx_xtrades_strategy;
DROP INDEX IF EXISTS idx_xtrades_expiration;

-- Positions
DROP INDEX IF EXISTS idx_positions_close_date;
DROP INDEX IF EXISTS idx_positions_symbol_date;
DROP INDEX IF EXISTS idx_positions_pnl;
DROP INDEX IF EXISTS idx_positions_strategy;
DROP INDEX IF EXISTS idx_positions_open_date;

-- Options
DROP INDEX IF EXISTS idx_options_dte_delta;
DROP INDEX IF EXISTS idx_options_premium;
DROP INDEX IF EXISTS idx_options_symbol_expiration;
DROP INDEX IF EXISTS idx_options_strike;
DROP INDEX IF EXISTS idx_options_iv_rank;

-- Kalshi
DROP INDEX IF EXISTS idx_kalshi_sport_date;
DROP INDEX IF EXISTS idx_kalshi_status;
DROP INDEX IF EXISTS idx_kalshi_ticker;
DROP INDEX IF EXISTS idx_kalshi_event_title;
DROP INDEX IF EXISTS idx_kalshi_predictions_confidence;
DROP INDEX IF EXISTS idx_kalshi_predictions_market;
DROP INDEX IF EXISTS idx_kalshi_predictions_date;

-- TradingView
DROP INDEX IF EXISTS idx_tv_watchlist_name;
DROP INDEX IF EXISTS idx_tv_symbol;
DROP INDEX IF EXISTS idx_tv_last_sync;

-- Stocks
DROP INDEX IF EXISTS idx_stocks_sector;
DROP INDEX IF EXISTS idx_stocks_market_cap;
DROP INDEX IF EXISTS idx_stocks_symbol;

-- Zones
DROP INDEX IF EXISTS idx_zones_symbol_active;
DROP INDEX IF EXISTS idx_zones_type;
DROP INDEX IF EXISTS idx_zones_strength;
DROP INDEX IF EXISTS idx_zones_created;

-- QA
DROP INDEX IF EXISTS idx_qa_tasks_status;
DROP INDEX IF EXISTS idx_qa_signoffs_agent;
DROP INDEX IF EXISTS idx_qa_reviews_date;

-- Analytics
DROP INDEX IF EXISTS idx_analytics_date;
DROP INDEX IF EXISTS idx_analytics_user;

-- Sports Games
DROP INDEX IF EXISTS idx_nfl_games_status_time;
DROP INDEX IF EXISTS idx_nfl_games_live;
DROP INDEX IF EXISTS idx_nfl_games_teams;
DROP INDEX IF EXISTS idx_nba_games_status_time;
DROP INDEX IF EXISTS idx_nba_games_live;
DROP INDEX IF EXISTS idx_nba_games_teams;
DROP INDEX IF EXISTS idx_ncaaf_games_status_time;
DROP INDEX IF EXISTS idx_ncaaf_games_live;
DROP INDEX IF EXISTS idx_ncaaf_games_teams;
DROP INDEX IF EXISTS idx_ncaab_games_status_time;
DROP INDEX IF EXISTS idx_ncaab_games_live;
DROP INDEX IF EXISTS idx_ncaab_games_teams;

-- Live Odds & AI Recommendations
DROP INDEX IF EXISTS idx_odds_snapshots_sport_game;
DROP INDEX IF EXISTS idx_odds_snapshots_time;
DROP INDEX IF EXISTS idx_odds_snapshots_game_time;
DROP INDEX IF EXISTS idx_ai_recs_sport_game;
DROP INDEX IF EXISTS idx_ai_recs_unsettled;
DROP INDEX IF EXISTS idx_ai_recs_confidence;
DROP INDEX IF EXISTS idx_ai_recs_created;
DROP INDEX IF EXISTS idx_ai_recs_result;

-- Prediction Tracking
DROP INDEX IF EXISTS idx_prediction_results_game;
DROP INDEX IF EXISTS idx_prediction_results_date;
DROP INDEX IF EXISTS idx_prediction_results_outcome;
DROP INDEX IF EXISTS idx_model_performance_model;

COMMIT;
*/
