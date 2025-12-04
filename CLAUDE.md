# CLAUDE.md - Project Context for AI Assistants

> **IMPORTANT**: This file should be updated on every commit to keep it fresh. When making commits, ensure this file reflects the current state of the codebase.

## Project Overview

Magnus (AVA Trading Platform) is an advanced options trading platform with AI-powered analysis, designed for cash-secured puts (CSP) and covered call (CC) strategies. It features real-time market data, TradingView integration, sports betting predictions, and comprehensive risk management.

## Architecture

### Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript + Vite + TanStack Query |
| Backend | FastAPI + Python 3.11+ |
| Database | PostgreSQL 14+ with asyncpg (async) |
| Cache | Redis |
| AI/ML | OpenAI, Anthropic, Groq, Ollama (local) |

### Directory Structure

```
Magnus/
├── backend/               # FastAPI backend
│   ├── routers/          # API endpoints
│   ├── services/         # Business logic
│   ├── infrastructure/   # Database, cache, observability
│   │   ├── database.py   # Async database (PRIMARY - use this)
│   │   └── cache.py      # Redis/in-memory cache
│   ├── database/         # Legacy sync pool (DEPRECATED)
│   └── models/           # Pydantic models
├── frontend/             # React frontend
│   ├── src/
│   │   ├── hooks/        # React Query hooks (useMagnusApi.ts)
│   │   ├── lib/          # Axios client, utilities
│   │   ├── components/   # UI components
│   │   └── pages/        # Route pages
│   └── package.json
├── src/                  # Legacy Python modules (standalone scripts)
│   ├── database/         # Old sync database (being phased out)
│   ├── prediction_agents/
│   └── *.py              # Data managers, scanners
└── .claude/              # Claude Code configuration
```

## Database Patterns

### Primary Pattern (Async - USE THIS)

```python
from backend.infrastructure.database import get_database, AsyncDatabaseManager

async def my_endpoint():
    db = await get_database()

    # Single row
    row = await db.fetchrow("SELECT * FROM table WHERE id = $1", id)
    value = row["column_name"]  # Dict access

    # Multiple rows
    rows = await db.fetch("SELECT * FROM table WHERE status = $1", status)

    # Single value
    count = await db.fetchval("SELECT COUNT(*) FROM table")

    # Execute (INSERT/UPDATE/DELETE)
    await db.execute("UPDATE table SET col = $1 WHERE id = $2", value, id)

    # Transaction
    async with db.transaction() as conn:
        await conn.execute("INSERT INTO ...", ...)
        await conn.execute("UPDATE ...", ...)
```

### Legacy Pattern (Sync - DEPRECATED)

```python
# DON'T use in new code - only for backward compatibility
from backend.database.connection import db_pool

with db_pool.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM table WHERE id = %s", (id,))
    row = cursor.fetchone()
    value = row[0]  # Index access
```

### Key Differences

| Aspect | Async (New) | Sync (Legacy) |
|--------|-------------|---------------|
| Import | `backend.infrastructure.database` | `backend.database.connection` |
| Placeholders | `$1, $2, $3` | `%s` |
| Row Access | `row["column"]` | `row[0]` |
| Functions | `async def` + `await` | `def` |
| Logging | `structlog` | `logging` |

## Frontend Query Options

All React Query hooks in `useMagnusApi.ts` use standardized options:

```typescript
// Default - Most endpoints
const DEFAULT_QUERY_OPTIONS = {
    retry: 1,
    retryDelay: 5000,
    refetchOnWindowFocus: false,
    networkMode: 'online',
};

// Polling - Live data endpoints
const POLLING_QUERY_OPTIONS = {
    ...DEFAULT_QUERY_OPTIONS,
    staleTime: 30000,
    refetchInterval: 30000,
};

// AI - Expensive operations
const AI_QUERY_OPTIONS = {
    ...DEFAULT_QUERY_OPTIONS,
    staleTime: 300000,  // 5 minutes
    retry: 0,           // Don't retry expensive ops
};
```

## Critical Files

| File | Purpose |
|------|---------|
| `backend/infrastructure/database.py` | Async database manager (PRIMARY) |
| `backend/database/connection.py` | Legacy sync pool (DEPRECATED) |
| `frontend/src/hooks/useMagnusApi.ts` | All React Query hooks |
| `frontend/src/lib/axios.ts` | Axios instance with interceptors |
| `backend/main.py` | FastAPI app entry point |

## Development Commands

```bash
# Start backend (from Magnus/)
cd backend && uvicorn main:app --reload --port 8000

# Start frontend (from Magnus/frontend/)
npm run dev

# Run backend tests
pytest backend/tests/

# Check Python imports
python -c "from backend.infrastructure.database import get_database"
```

## Recent Migration (2025-12-04)

All backend routers migrated from sync to async database:

- scanner.py, sports.py, watchlist.py, analytics.py
- briefings.py, options.py, agents.py, predictions.py
- integration_test.py, universe_service.py
- dashboard_service.py, prediction_service.py

**Changes made:**
1. `from src.database.connection_pool import` → `from backend.infrastructure.database import get_database`
2. `import logging` → `import structlog`
3. SQL: `%s` → `$1, $2, $3`
4. Row: `row[0]` → `row["column_name"]`
5. Functions: `def` → `async def`

## Sync-on-Commit Guidelines

**This file (CLAUDE.md) must be updated on every significant commit to stay current.**

When committing changes, update this file to reflect:
- New files or directory changes
- API pattern changes
- New dependencies
- Breaking changes
- Migration status

## NO DUMMY DATA POLICY

**CRITICAL**: This project has a strict NO DUMMY DATA policy. All data must come from real sources:
- Real Robinhood API data
- Real TradingView watchlists
- Real market data (yfinance, etc.)
- Real database queries

Never return mock/fake data from API endpoints.

## Observability

```python
import structlog
logger = structlog.get_logger(__name__)

# Structured logging
logger.info("event_name", key1=value1, key2=value2)
logger.error("error_event", error=str(e), context=ctx)
```

## Connection Pool Stats

The legacy sync pool (if still needed) has:
- Min connections: 5
- Max connections: 50
- Connection timeout: 10s
- Query timeout: 30s

## Environment Variables

Required in `.env`:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ava
DB_USER=postgres
DB_PASSWORD=your_password
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Server Ports - HARDCODED

**CRITICAL**: These ports are hardcoded throughout the codebase. Do NOT change without updating all files.

| Service | Port | Notes |
|---------|------|-------|
| Backend API | 8000 | FastAPI server |
| Frontend Dev | 5173 | Vite dev server |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache (optional - falls back to in-memory) |

### Files with Port Configuration

- `backend/config.py` - SERVER_PORT, FRONTEND_PORT
- `backend/main.py` - run_server() function
- `frontend/vite.config.ts` - server.port, proxy target
- `frontend/src/config/api.ts` - API_BASE_URL, API_HOST
- `frontend/src/lib/api-client.ts` - apiClient baseURL
- `frontend/src/hooks/useSportsWebSocket.ts` - WebSocket URL
- `frontend/src/hooks/useMagnusApi.ts` - WebSocket URLs

## API Base URLs

- Backend: `http://localhost:8000/api`
- Frontend: `http://localhost:5173`
- WebSocket: `ws://localhost:8000/ws/*`

## Cache Behavior

Redis is optional. If Redis is not running, the system falls back to in-memory caching:
- Warning in logs: "Redis connection failed... using in-memory fallback"
- No action required - the system works correctly without Redis

---

## Sync Status Panel (2025-12-04)

The Positions page now uses an advanced AI-powered sync system:

### Components

| File | Purpose |
|------|---------|
| `frontend/src/store/syncStore.ts` | Zustand store for sync state management |
| `frontend/src/components/SyncStatusPanel.tsx` | Advanced sync UI with progress stages |

### Features

- **Multi-stage progress tracking** (connecting → authenticating → fetching → processing → completing)
- **AI-powered sync recommendations** based on time elapsed, market hours, failure patterns
- **Connection health monitoring** (healthy/degraded/offline)
- **Sync history** with duration and success rate statistics
- **Persistent state** via Zustand persist middleware

### Usage

```tsx
import { SyncStatusPanel } from '../components/SyncStatusPanel'

// Compact - just buttons
<SyncStatusPanel variant="compact" />

// Expanded - buttons + health + timing
<SyncStatusPanel variant="expanded" />

// Full - all features including history
<SyncStatusPanel variant="full" showHistory={true} showRecommendation={true} />
```

### Store Selectors

```tsx
import { useSyncStore, useSyncStatus, useSyncHealth, useSyncStats } from '../store/syncStore'

// Current sync status
const { stage, progress, currentAction, isSyncing } = useSyncStatus()

// Health monitoring
const { connectionHealth, recommendation } = useSyncHealth()

// Statistics
const { totalSyncs, successRate, averageSyncDuration } = useSyncStats()
```

---

*Last updated: 2025-12-04 - Added AI-powered Sync Status Panel with Zustand store*
