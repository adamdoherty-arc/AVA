-- ============================================================================
-- Scanner Performance Optimization - Composite Indexes
-- ============================================================================
-- Created: 2025-12-04
-- Purpose: Add composite indexes for scanner endpoint queries
-- Expected Impact: 2-10x faster queries on premium_opportunities table
--
-- These indexes optimize the /scanner/stored-premiums endpoint which queries:
-- WHERE dte >= $1 AND dte <= $2 AND premium_pct >= $3
-- ORDER BY annualized_return DESC / monthly_return DESC / etc.
-- ============================================================================

BEGIN;

-- ============================================================================
-- PREMIUM OPPORTUNITIES COMPOSITE INDEXES
-- ============================================================================

-- Primary composite index for the most common query pattern:
-- Filter by DTE range + premium_pct, sort by annualized_return
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_premium_opp_dte_premium_annual
ON premium_opportunities(dte, premium_pct, annualized_return DESC)
WHERE annualized_return IS NOT NULL;

-- Composite index for monthly_return sorting
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_premium_opp_dte_premium_monthly
ON premium_opportunities(dte, premium_pct, monthly_return DESC)
WHERE monthly_return IS NOT NULL;

-- Composite index for delta-based sorting (wheel strategy)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_premium_opp_dte_delta
ON premium_opportunities(dte, ABS(delta))
WHERE delta IS NOT NULL;

-- Index for symbol + DTE range queries (watchlist filtering)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_premium_opp_symbol_dte
ON premium_opportunities(symbol, dte, annualized_return DESC);

-- Index for option type filtering with DTE
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_premium_opp_type_dte
ON premium_opportunities(option_type, dte, annualized_return DESC);

-- ============================================================================
-- SCANNER WATCHLISTS INDEXES
-- ============================================================================

-- Index for active watchlist lookups (frequently accessed)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scanner_watchlists_active
ON scanner_watchlists(is_active, sort_order)
WHERE is_active = true;

-- Index for watchlist_id lookups (used in stored-premiums with watchlist_id param)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scanner_watchlists_id
ON scanner_watchlists(watchlist_id)
WHERE is_active = true;

-- ============================================================================
-- PREMIUM SCAN HISTORY INDEXES
-- ============================================================================

-- Index for recent scan lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scan_history_recent
ON premium_scan_history(created_at DESC)
WHERE created_at > NOW() - INTERVAL '7 days';

-- ============================================================================
-- STOCK PREMIUMS TABLE INDEXES (if exists)
-- ============================================================================

-- These indexes help the /scanner/premiums/* endpoints
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_premiums_scanned
ON stock_premiums(scanned_at DESC)
WHERE scanned_at IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_premiums_symbol_annual
ON stock_premiums(symbol, annual_return DESC)
WHERE annual_return IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_stock_premiums_recent_high_return
ON stock_premiums(annual_return DESC)
WHERE scanned_at > NOW() - INTERVAL '24 hours' AND annual_return >= 20;

COMMIT;

-- ============================================================================
-- ANALYZE TABLES
-- ============================================================================
-- Run after creating indexes to update query planner statistics

ANALYZE premium_opportunities;
ANALYZE scanner_watchlists;
ANALYZE premium_scan_history;
ANALYZE stock_premiums;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Scanner Performance Indexes Created!';
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Run EXPLAIN ANALYZE on your queries to verify';
    RAISE NOTICE 'index usage and performance improvement.';
    RAISE NOTICE '============================================';
END $$;

-- ============================================================================
-- ROLLBACK SCRIPT (if needed)
-- ============================================================================

/*
BEGIN;

-- Premium Opportunities
DROP INDEX IF EXISTS idx_premium_opp_dte_premium_annual;
DROP INDEX IF EXISTS idx_premium_opp_dte_premium_monthly;
DROP INDEX IF EXISTS idx_premium_opp_dte_delta;
DROP INDEX IF EXISTS idx_premium_opp_symbol_dte;
DROP INDEX IF EXISTS idx_premium_opp_type_dte;

-- Scanner Watchlists
DROP INDEX IF EXISTS idx_scanner_watchlists_active;
DROP INDEX IF EXISTS idx_scanner_watchlists_id;

-- Scan History
DROP INDEX IF EXISTS idx_scan_history_recent;

-- Stock Premiums
DROP INDEX IF EXISTS idx_stock_premiums_scanned;
DROP INDEX IF EXISTS idx_stock_premiums_symbol_annual;
DROP INDEX IF EXISTS idx_stock_premiums_recent_high_return;

COMMIT;
*/
