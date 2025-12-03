# AVA Database Dump

PostgreSQL database dump for the AVA trading platform.

## Files

- `schema.sql` - Schema only (tables, indexes, functions, triggers)
- `full_dump.sql` - Complete dump with schema and data

## Database Info

- **Database**: magnus
- **PostgreSQL version**: 14+
- **Total tables**: 76
- **Dump date**: 2024-12-03

## Major Table Categories

### Trading & Options
- `stocks_universe` - 10,000+ optionable stocks with metadata
- `etfs_universe` - ETF listings with option flags
- `premium_opportunities` - Premium selling opportunities
- `scanner_watchlists` - Cached watchlists for fast loading
- `trade_journal` - Trade logging and history

### Sports Betting
- `nfl_games`, `nba_games`, `ncaa_football_games`, `ncaa_basketball_games`
- `kalshi_markets`, `kalshi_predictions`
- `ai_betting_recommendations`
- `user_bets`, `user_betting_profile`

### AI & Automation
- `ava_feature_specs` - Technical specifications for features
- `ava_user_goals` - User goal tracking
- `automations`, `automation_executions`
- `agent_memory`, `agent_performance`

### TradingView Integration
- `tv_watchlists_api` - Synced TradingView watchlists
- `tv_symbols_api` - Symbol mappings

## Restore Instructions

```bash
# Create database
createdb magnus

# Restore schema only
psql -d magnus -f database_dump/schema.sql

# OR restore full database with data
psql -d magnus -f database_dump/full_dump.sql
```

## Notes

- Sensitive data (API keys, passwords) are NOT included
- This dump includes reference data (stocks, ETFs) that may need updating
- Run sync scripts after restore to refresh cached data
