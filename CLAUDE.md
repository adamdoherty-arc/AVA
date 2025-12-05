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
| Cache | Redis (optional, falls back to in-memory) |
| AI/ML | OpenAI, Anthropic, Groq, Ollama (local) |

### Directory Structure

```
Magnus/
├── backend/               # FastAPI backend
│   ├── routers/          # API endpoints
│   ├── services/         # Business logic
│   ├── infrastructure/   # Database, cache, observability
│   │   ├── database.py   # Async database (PRIMARY)
│   │   └── cache.py      # Redis/in-memory cache
│   └── models/           # Pydantic models
├── frontend/             # React frontend
│   ├── src/
│   │   ├── hooks/        # React Query hooks (useMagnusApi.ts)
│   │   ├── lib/          # Axios client, utilities
│   │   ├── components/   # UI components
│   │   └── pages/        # Route pages
│   └── package.json
├── src/                  # Standalone Python scripts & utilities
│   ├── database/         # Sync database for standalone scripts
│   ├── prediction_agents/
│   └── *.py              # Data managers, scanners
└── .claude/              # Claude Code configuration
```

## Database Pattern (Async)

All backend code uses async database access with asyncpg:

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

### Key Points

| Aspect | Pattern |
|--------|---------|
| Import | `from backend.infrastructure.database import get_database` |
| Placeholders | `$1, $2, $3` (asyncpg style) |
| Row Access | `row["column"]` (dict access) |
| Functions | `async def` + `await` |
| Logging | `structlog` |

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
| `backend/infrastructure/database.py` | Async database manager |
| `frontend/src/hooks/useMagnusApi.ts` | All React Query hooks |
| `frontend/src/lib/axios.ts` | Enhanced axios with auto-timeout & circuit breaker |
| `frontend/src/lib/react-query.ts` | Smart cache warming & prefetching |
| `frontend/src/components/APIErrorDisplay.tsx` | Reusable API error component |
| `backend/main.py` | FastAPI app entry point |

## Frontend API Infrastructure (2025-12-05)

### Automatic Timeout Detection

The axios interceptor automatically applies appropriate timeouts based on endpoint patterns:

| Category | Timeout | Endpoints |
|----------|---------|-----------|
| FAST | 10s | `/health`, `/cache/*` |
| STANDARD | 30s | Most endpoints |
| SLOW | 60s | `/portfolio/positions`, `/scanner/watchlists`, `/robinhood/*` |
| AI | 120s | `/ai/*`, `/chat/*`, `/recommendations`, `/predictions/*` |
| SCAN | 180s | `/scanner/scan`, `/backtest`, `/stress-test` |

**No manual timeout overrides needed** - the interceptor handles it automatically.

### Circuit Breaker

After 5 consecutive failures to an endpoint group, requests are automatically blocked for 30 seconds to prevent cascading failures.

```typescript
// Check circuit status
import { getCircuitBreakerStatus } from '@/lib/axios';
console.log(getCircuitBreakerStatus());
// { portfolio: { failures: 3, isOpen: false }, scanner: { failures: 0, isOpen: false } }
```

### Error Handling

Use the `APIErrorDisplay` component for consistent error handling:

```tsx
import { APIErrorDisplay } from '@/components';

// Full display
<APIErrorDisplay
  error={error}
  onRetry={() => refetch()}
  isRetrying={isFetching}
  title="Failed to Load Data"
/>

// Compact inline
<APIErrorDisplay error={error} onRetry={refetch} compact />
```

### Cache Warming

Pre-warm cache before navigation:

```typescript
import { warmCacheForRoute, invalidateCategory } from '@/lib/react-query';

// Warm cache for a route
await warmCacheForRoute('/positions');

// Invalidate a category
await invalidateCategory('portfolio'); // Clears positions, summary, enriched
```

## Development Commands

```bash
# Start backend (from Magnus/)
cd backend && uvicorn main:app --reload --port 8002

# Start frontend (from Magnus/frontend/)
npm run dev

# Run backend tests
pytest backend/tests/

# Check Python imports
python -c "from backend.infrastructure.database import get_database"
```

## Database Pool Configuration

The async database pool settings:
- Min connections: 5
- Max connections: 50
- Connection timeout: 10s
- Query timeout: 30s (configurable)

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

## Server Ports - CENTRALIZED CONFIG

**CRITICAL**: Ports are managed via centralized config files. Update the config, not individual files.

| Service | Port | Config File |
|---------|------|-------------|
| Backend API | 8002 | `backend/config.py` → `SERVER_PORT` |
| Frontend Dev | 5181 | `frontend/vite.config.ts` → `server.port` |
| PostgreSQL | 5432 | `backend/config.py` → `DB_PORT` |
| Redis | 6379 | `backend/config.py` → `REDIS_PORT` |

### Centralized Port Configuration

**Backend** (`backend/config.py`):
```python
SERVER_PORT: int = 8002  # Single source of truth
FRONTEND_PORT: int = 5181
```

**Frontend** (`frontend/src/config/api.ts`):
```typescript
export const BACKEND_URL = 'http://localhost:8002';
export const WS_URL = 'ws://localhost:8002';
```

All other files import from these central configs.

## API Base URLs

- Backend: `http://localhost:8002/api`
- Frontend: `http://localhost:5181`
- WebSocket: `ws://localhost:8002/ws/*`

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

---

*Last updated: 2025-12-05 - Removed legacy sync database code, using async-only database pattern*
