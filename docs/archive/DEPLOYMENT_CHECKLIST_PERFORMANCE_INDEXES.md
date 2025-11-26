# Performance Indexes Deployment Checklist

**Date**: 2025-11-20
**Project**: Magnus Options Wheel Strategy Trading System
**Change**: Add 9 Strategic Database Indexes
**Risk Level**: LOW (read-only, fully reversible)

---

## Pre-Deployment

### Environment Verification
- [ ] Database is PostgreSQL 12 or higher
- [ ] Current database version: _______________
- [ ] Database size: _______________
- [ ] Estimated free disk space available: > 50 MB
- [ ] Database is on a supported backup schedule
- [ ] Recent backup exists (< 24 hours old)
- [ ] Backup was verified successfully

### Performance Baseline
- [ ] Captured baseline query times (saved to baseline_metrics.txt)
  - [ ] SELECT * FROM stocks WHERE sector = 'Technology' - _______ ms
  - [ ] SELECT * FROM positions WHERE user_id = '...' AND status = 'open' - _______ ms
  - [ ] SELECT * FROM kalshi_markets WHERE status = 'open' - _______ ms
  - [ ] SELECT * FROM nfl_games WHERE game_status IN ('scheduled','live') AND game_time > NOW() - _______ ms

### Team Notification
- [ ] Development team notified
- [ ] DBA notified (if separate)
- [ ] QA team notified
- [ ] Operations team notified
- [ ] Maintenance window scheduled (if required)

---

## Development Environment Testing

### Apply Migration
- [ ] Connected to development database
- [ ] Backup created: dev_backup_<date>.sql
- [ ] Migration script copied to server
- [ ] Migration executed successfully:
  ```bash
  psql -U postgres -d magnus -f src/database/add_performance_indexes.sql
  ```
- [ ] No errors reported
- [ ] Execution time: _______ seconds

### Verify Indexes Created
- [ ] All 9 indexes exist and are active
- [ ] Used verification query:
  ```sql
  SELECT tablename, indexname FROM pg_indexes
  WHERE schemaname = 'public' AND indexname LIKE 'idx_%'
  ORDER BY tablename, indexname;
  ```
- [ ] Verified against expected index list
- [ ] Confirmed on tables:
  - [ ] stocks (3 indexes)
  - [ ] options_chains (2 indexes)
  - [ ] positions (2 indexes)
  - [ ] kalshi_markets (1 index)
  - [ ] nfl_games (1 index)

### Performance Testing
- [ ] Ran VACUUM ANALYZE on development database
- [ ] Collected post-index performance metrics
- [ ] Performance improvement verified (queries faster)
- [ ] Recorded performance delta: _______ % average improvement

### Application Testing
- [ ] Dashboard loads without errors
- [ ] Stock screening works correctly
- [ ] Portfolio view loads faster
- [ ] Options analysis works
- [ ] Kalshi market queries functional
- [ ] NFL games queries functional
- [ ] No application errors in logs
- [ ] No SQL errors reported

---

## Staging Environment Testing

### Pre-Staging
- [ ] Staging database size similar to production
- [ ] Production data snapshot available
- [ ] Staging backup created before migration

### Apply to Staging
- [ ] Connected to staging database
- [ ] Backup created: staging_backup_<date>.sql
- [ ] Migration executed successfully
- [ ] Execution time: _______ seconds (compare to dev: _______ seconds)

### Load Testing
- [ ] Run application with realistic load
- [ ] Dashboard performance under load: _______ ms avg response
- [ ] Portfolio queries under load: _______ ms avg response
- [ ] No timeout errors observed
- [ ] No lock contention detected
- [ ] CPU usage acceptable (< 80%)
- [ ] Disk I/O acceptable
- [ ] Memory usage acceptable

### 24-Hour Stability Test
- [ ] Indexes running for 24 hours
- [ ] No performance degradation observed
- [ ] Index bloat is minimal
- [ ] No unexpected errors in logs

---

## Production Deployment

### Pre-Production Checklist
- [ ] All development testing passed
- [ ] All staging testing passed
- [ ] Maintenance window scheduled
- [ ] Team ready for support
- [ ] Rollback procedure documented and tested
- [ ] Communication plan prepared

### Deployment Execution
- [ ] Maintenance window started
- [ ] Application servers put in read-only mode (if applicable)
- [ ] Database connections drained
- [ ] Final production backup created
- [ ] Backup verified successfully
- [ ] Migration script copied to production server
- [ ] Migration executed:
  ```bash
  psql -U postgres -d magnus -f src/database/add_performance_indexes.sql
  ```
- [ ] No errors reported
- [ ] Execution time: _______ seconds

### Post-Deployment Verification
- [ ] All 9 indexes present in production
- [ ] Index status normal (no invalid states)
- [ ] VACUUM ANALYZE completed successfully
- [ ] Application able to connect to database
- [ ] Dashboard loads without errors
- [ ] Initial queries functional
- [ ] Performance improvement verified

### Monitoring (First 2 Hours)
- [ ] Error rate: Normal (no spike)
- [ ] Response times: Improved
- [ ] CPU usage: Normal
- [ ] Disk I/O: Normal
- [ ] Database connections: Stable
- [ ] No locks or deadlocks detected
- [ ] User reports: None

---

## Post-Deployment

### Day 1 (First 24 Hours)
- [ ] Monitor error logs regularly
- [ ] Check index usage statistics
- [ ] Verify query performance improvements
- [ ] Confirm no application issues
- [ ] Dashboard responsiveness: _______ ms
- [ ] Portfolio queries: _______ ms avg
- [ ] No customer complaints received

### Week 1
- [ ] Run VACUUM ANALYZE (routine maintenance)
- [ ] Review slow query logs (if available)
- [ ] Collect performance metrics
- [ ] Prepare performance comparison report
- [ ] Team debrief and lessons learned

### Performance Comparison Report

Baseline vs. Post-Deployment metrics

| Query | Before | After | Improvement |
|-------|--------|-------|------------|
| Stock sector filter | _______ ms | _______ ms | _______ % |
| Options chain lookup | _______ ms | _______ ms | _______ % |
| Position queries | _______ ms | _______ ms | _______ % |
| Kalshi active markets | _______ ms | _______ ms | _______ % |
| NFL upcoming games | _______ ms | _______ ms | _______ % |
| Average Improvement | | | _______ % |

### Documentation Updates
- [ ] Update deployment wiki/docs
- [ ] Document performance improvements achieved
- [ ] Add indexes to runbooks
- [ ] Update capacity planning assumptions
- [ ] Archive baseline metrics

---

## Rollback Plan (If Needed)

### Decision Criteria (When to Rollback)
Rollback if ANY of the following occur:
- [ ] Significant query performance degradation (> 20% slower than baseline)
- [ ] Application errors related to queries
- [ ] Database deadlocks or lock contention
- [ ] Disk space issues
- [ ] Data corruption detected
- [ ] Indexes consuming excessive resources

### Rollback Steps
1. [ ] Notify team of rollback decision
2. [ ] Stop application writes (if possible)
3. [ ] Execute rollback script:
   ```bash
   psql -U postgres -d magnus -f src/database/add_performance_indexes_rollback.sql
   ```
4. [ ] Verify indexes are removed
5. [ ] Run VACUUM ANALYZE
6. [ ] Verify application functionality
7. [ ] Monitor performance recovery
8. [ ] Document root cause analysis

---

## Sign-Off

| Role | Name | Date |
|------|------|------|
| Database Administrator | __________ | _______ |
| Development Lead | __________ | _______ |
| Operations Lead | __________ | _______ |
| Release Manager | __________ | _______ |

---

## Key Files

- `src/database/add_performance_indexes.sql` - Main migration
- `src/database/add_performance_indexes_rollback.sql` - Rollback script
- `docs/DATABASE_PERFORMANCE_OPTIMIZATION.md` - Detailed report
- `docs/DATABASE_INDEX_QUICK_REFERENCE.md` - Quick reference
- `database_schema.sql` - Updated main schema
- `src/kalshi_schema.sql` - Updated Kalshi schema
- `src/nfl_data_schema.sql` - Updated NFL schema

---

**Deployment Status**: [ ] Not Started [ ] In Progress [ ] Complete [ ] Rolled Back

