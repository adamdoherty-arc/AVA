# Earnings Sync Service - System Architecture & Design

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Component Design](#component-design)
3. [Data Flow](#data-flow)
4. [API Specifications](#api-specifications)
5. [Database Design](#database-design)
6. [Scaling Considerations](#scaling-considerations)
7. [Security & Reliability](#security--reliability)

---

## High-Level Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WHEEL STRATEGY SYSTEM                         │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Options    │  │  Portfolio   │  │   Earnings   │              │
│  │   Scanner    │  │  Management  │  │Sync Service  │  <-- NEW     │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                  │                  │                      │
│         └──────────────────┼──────────────────┘                     │
│                            │                                         │
│                   ┌────────▼────────┐                               │
│                   │   PostgreSQL    │                               │
│                   │    Database     │                               │
│                   │   (TimescaleDB) │                               │
│                   └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Earnings Sync Service Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EARNINGS SYNC SERVICE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     PRESENTATION LAYER                         │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  CLI Tool          │  Flask API       │  Scheduled Jobs       │  │
│  │  (sync_earnings)   │  (REST endpoints)│  (Cron/APScheduler)  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      SERVICE LAYER                            │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │                                                               │  │
│  │  EarningsSyncService                                         │  │
│  │  ├─ sync_all_stocks_earnings()                               │  │
│  │  ├─ sync_symbol_earnings()                                   │  │
│  │  ├─ get_upcoming_earnings()                                  │  │
│  │  ├─ get_historical_earnings()                                │  │
│  │  └─ calculate_beat_rate()                                    │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    INTEGRATION LAYER                          │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  Robinhood API Client    │    Database Connection Pool       │  │
│  │  - Authentication        │    - Connection management        │  │
│  │  - Rate limiting         │    - Transaction handling         │  │
│  │  - Retry logic           │    - Prepared statements          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                │                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      DATA LAYER                               │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  Tables:             │  Views:              │  Functions:     │  │
│  │  - earnings_history  │  - v_upcoming_       │  - calculate_   │  │
│  │  - earnings_events   │    earnings          │    beat_rate()  │  │
│  │  - earnings_sync_    │  - v_earnings_beat_  │  - get_next_    │  │
│  │    status            │    stats             │    earnings_    │  │
│  │  - earnings_alerts   │  - v_high_conviction_│    date()       │  │
│  │                      │    earnings          │                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. EarningsSyncService Class

**Responsibilities:**
- Orchestrate earnings data synchronization
- Manage Robinhood API authentication
- Handle data parsing and transformation
- Coordinate database operations
- Track sync status and errors

**Key Methods:**

```python
class EarningsSyncService:
    def __init__(self, max_retries=3, retry_delay=5, batch_size=50):
        """Initialize service with configuration"""

    def sync_all_stocks_earnings(self, limit=None, rate_limit_delay=1.0):
        """
        Sync earnings for all stocks in database
        Returns: {'status': str, 'total_stocks': int, ...}
        """

    def sync_symbol_earnings(self, symbol: str):
        """
        Sync earnings for single symbol
        Returns: {'symbol': str, 'status': str, ...}
        """

    def get_upcoming_earnings(self, days_ahead: int):
        """
        Query upcoming earnings events
        Returns: [{'symbol': str, 'earnings_date': date, ...}, ...]
        """

    def get_historical_earnings(self, symbol: str, limit: int):
        """
        Query historical earnings for symbol
        Returns: [{'report_date': date, 'eps_actual': float, ...}, ...]
        """

    def calculate_beat_rate(self, symbol: str, lookback_quarters: int):
        """
        Calculate earnings beat percentage
        Returns: float (0-100)
        """
```

**Design Patterns:**
- **Singleton**: Single service instance manages all operations
- **Repository**: Abstracts database operations
- **Retry Pattern**: Exponential backoff for API calls
- **Batch Processing**: Groups DB operations for efficiency

### 2. Data Parser

**Responsibilities:**
- Parse Robinhood API response
- Extract earnings data fields
- Calculate derived metrics (surprise %, beat/miss)
- Validate data integrity

**Data Transformation Pipeline:**

```
Robinhood API Response
        │
        ▼
┌───────────────────┐
│  Extract Fields   │  - report.date, eps.actual, eps.estimate
│                   │  - call.datetime, year, quarter
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Calculate        │  - eps_surprise = actual - estimate
│  Metrics          │  - eps_surprise_% = (surprise / |estimate|) * 100
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Classify         │  - beat: actual > estimate
│  Beat/Miss        │  - miss: actual < estimate
└───────┬───────────┘  - meet: actual == estimate
        │
        ▼
┌───────────────────┐
│  Validate         │  - Required fields present
│  Data             │  - Date format valid
└───────┬───────────┘  - Numeric values in range
        │
        ▼
  Parsed Record
```

### 3. Database Connection Manager

**Responsibilities:**
- Manage PostgreSQL connection pool
- Handle transaction lifecycle
- Execute parameterized queries
- Manage batch inserts

**Connection Lifecycle:**

```
┌────────────────┐
│  Get Config    │  Load from environment
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  Create Pool   │  psycopg2.pool.SimpleConnectionPool
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  Get Conn      │  pool.getconn()
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  Execute       │  cur.execute() / execute_batch()
│  Queries       │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  Commit/       │  conn.commit() / conn.rollback()
│  Rollback      │
└────────┬───────┘
         │
         ▼
┌────────────────┐
│  Return Conn   │  pool.putconn(conn)
└────────────────┘
```

### 4. Retry Mechanism

**Implementation:**

```python
def fetch_earnings_with_retry(self, symbol: str) -> Optional[List[Dict]]:
    for attempt in range(1, self.max_retries + 1):
        try:
            earnings_data = rh.get_earnings(symbol)
            return earnings_data

        except RateLimitError as e:
            wait_time = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
            logger.warning(f"Rate limited on {symbol}, waiting {wait_time}s")
            time.sleep(wait_time)

        except NetworkError as e:
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
            else:
                logger.error(f"Network error after {attempt} attempts: {e}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    return None
```

**Retry Strategy:**
- **Max Attempts**: 3 (configurable)
- **Backoff**: Exponential (delay * 2^attempt)
- **Jitter**: Optional random jitter to prevent thundering herd

---

## Data Flow

### Sync Flow (Single Symbol)

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Login to Robinhood     │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Fetch earnings data    │──┐
│  (with retry logic)     │  │ Retry on failure
└──────┬──────────────────┘  │ (up to 3 times)
       │                     │
       │◄────────────────────┘
       ▼
┌─────────────────────────┐
│  Parse each record      │
│  - Extract fields       │
│  - Calculate surprise   │
│  - Classify beat/miss   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Split into:            │
│  - Historical (past)    │
│  - Upcoming (future)    │
└──────┬──────────────────┘
       │
       ├────────────────────┐
       │                    │
       ▼                    ▼
┌──────────────┐   ┌──────────────┐
│ Upsert to    │   │ Upsert to    │
│ earnings_    │   │ earnings_    │
│ history      │   │ events       │
└──────┬───────┘   └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
                ▼
┌───────────────────────────┐
│  Update sync_status       │
│  - last_sync_at = NOW()   │
│  - status = 'success'     │
│  - counts updated         │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│  Return result            │
│  {status, counts, error}  │
└───────────────────────────┘
```

### Batch Sync Flow (All Stocks)

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│  Get all stock symbols  │
│  from database          │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  FOR EACH symbol:       │
│  ├─ Sync earnings       │◄───┐
│  ├─ Update counters     │    │ Loop through
│  └─ Rate limit delay    │    │ all symbols
└──────┬──────────────────┘    │
       │                       │
       │───────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Generate summary       │
│  - Total stocks         │
│  - Success count        │
│  - Failure count        │
│  - Total records        │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Logout from Robinhood  │
└──────┬──────────────────┘
       │
       ▼
┌─────────────┐
│    END      │
└─────────────┘
```

---

## API Specifications

### REST API Endpoints (Flask Example)

#### GET /api/earnings/upcoming

**Description**: Get upcoming earnings events

**Query Parameters:**
- `days` (int, default=30): Days to look ahead
- `min_beat_rate` (float, optional): Filter by beat rate percentage
- `sector` (string, optional): Filter by sector

**Response:**
```json
{
  "success": true,
  "count": 15,
  "events": [
    {
      "symbol": "NVDA",
      "earnings_date": "2025-11-15",
      "earnings_time": "amc",
      "eps_estimate": 0.75,
      "historical_beat_rate_pct": 87.5,
      "last_quarter_surprise_pct": 15.2
    }
  ]
}
```

#### GET /api/earnings/history/{symbol}

**Description**: Get earnings history for a symbol

**Path Parameters:**
- `symbol` (string): Stock ticker

**Query Parameters:**
- `quarters` (int, default=8): Number of quarters

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "count": 8,
  "history": [
    {
      "report_date": "2025-10-31",
      "fiscal_year": 2025,
      "fiscal_quarter": 4,
      "eps_actual": 1.29,
      "eps_estimate": 1.26,
      "eps_surprise_percent": 2.38,
      "beat_miss": "beat"
    }
  ]
}
```

#### GET /api/earnings/beat-rate/{symbol}

**Description**: Get earnings beat rate

**Path Parameters:**
- `symbol` (string): Stock ticker

**Query Parameters:**
- `quarters` (int, default=8): Lookback period

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "beat_rate": 75.0,
  "lookback_quarters": 8
}
```

#### POST /api/earnings/sync/{symbol}

**Description**: Trigger sync for a symbol

**Path Parameters:**
- `symbol` (string): Stock ticker

**Response:**
```json
{
  "success": true,
  "result": {
    "symbol": "AAPL",
    "status": "success",
    "historical_count": 8,
    "upcoming_count": 1
  }
}
```

#### GET /api/earnings/stats

**Description**: Get sync statistics

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_symbols": 999,
    "successful_syncs": 950,
    "failed_syncs": 10,
    "total_historical_records": 7600,
    "total_upcoming_events": 150,
    "last_sync": "2025-10-28T06:00:00Z"
  }
}
```

---

## Database Design

### Normalization Strategy

**3NF (Third Normal Form)** - Applied to earnings tables to:
- Eliminate data redundancy
- Ensure data integrity
- Allow efficient updates

**Denormalization** - Applied to views for:
- Fast read performance
- Pre-calculated aggregations
- Simplified queries

### Index Strategy

```sql
-- Primary lookup indexes
CREATE INDEX idx_earnings_history_symbol ON earnings_history(symbol);
CREATE INDEX idx_earnings_events_symbol ON earnings_events(symbol);

-- Time-based indexes
CREATE INDEX idx_earnings_history_date ON earnings_history(report_date DESC);
CREATE INDEX idx_earnings_events_date ON earnings_events(earnings_date);

-- Composite indexes for common queries
CREATE INDEX idx_earnings_history_symbol_date
  ON earnings_history(symbol, report_date DESC);

-- Partial indexes for filtered queries
CREATE INDEX idx_earnings_events_upcoming
  ON earnings_events(earnings_date)
  WHERE has_occurred = FALSE AND earnings_date >= CURRENT_DATE;
```

### Data Partitioning (Future)

For scaling beyond 10M+ records:

```sql
-- Partition earnings_history by year
CREATE TABLE earnings_history_2025 PARTITION OF earnings_history
  FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE earnings_history_2024 PARTITION OF earnings_history
  FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

---

## Scaling Considerations

### Current Capacity

**Database:**
- 1000 stocks × 8 quarters = 8,000 historical records
- Storage: ~50MB (with indexes)
- Query time: <100ms for most queries

**API:**
- Sync rate: ~2-3 seconds per symbol
- Full sync: ~50-80 minutes for 1000 stocks
- Rate limit: ~1 request/second (Robinhood limit)

### Scaling Strategies

#### 1. Horizontal Scaling (Database)

```
┌──────────────┐
│   Master     │  (Write)
│  PostgreSQL  │
└──────┬───────┘
       │
       ├────────────┬────────────┐
       ▼            ▼            ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Read     │  │ Read     │  │ Read     │
│ Replica 1│  │ Replica 2│  │ Replica 3│
└──────────┘  └──────────┘  └──────────┘
```

#### 2. Caching Layer

```
┌──────────────┐
│   Redis      │  (Cache upcoming earnings)
│   Cache      │
└──────┬───────┘
       │ Cache miss
       ▼
┌──────────────┐
│  PostgreSQL  │
└──────────────┘
```

**Cache Strategy:**
- Cache upcoming earnings (30 days): 1 hour TTL
- Cache beat rates: 24 hour TTL
- Invalidate on sync

#### 3. Queue-Based Sync

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Producer   │─────>│   RabbitMQ   │─────>│   Workers    │
│  (Scheduler) │      │    Queue     │      │   (1-10)     │
└──────────────┘      └──────────────┘      └──────────────┘
```

**Benefits:**
- Parallel processing
- Fault tolerance
- Load balancing

#### 4. Microservices Architecture (Future)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Earnings   │  │  Options    │  │ Portfolio   │
│  Service    │  │  Service    │  │  Service    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                ┌───────▼────────┐
                │   API Gateway  │
                └────────────────┘
```

---

## Security & Reliability

### Security Measures

1. **Credentials Management**
   - Environment variables (not hardcoded)
   - Secrets manager (AWS Secrets Manager, HashiCorp Vault)
   - Encrypted at rest

2. **Database Security**
   - Connection over SSL/TLS
   - Least privilege access (read-only users)
   - Row-level security (RLS) for multi-tenant

3. **API Security**
   - Rate limiting (per user/IP)
   - Authentication (JWT tokens)
   - Input validation and sanitization

### Reliability Measures

1. **Error Handling**
   - Try-catch blocks at every layer
   - Graceful degradation
   - Meaningful error messages

2. **Monitoring**
   - Sync success rate tracking
   - API error rate monitoring
   - Database performance metrics

3. **Backup & Recovery**
   ```bash
   # Daily database backup
   pg_dump -U postgres magnus > backup_$(date +%Y%m%d).sql

   # Restore
   psql -U postgres magnus < backup_20251028.sql
   ```

4. **Health Checks**
   ```python
   @app.route('/health')
   def health_check():
       try:
           # Check database
           conn = service.get_db_connection()
           conn.close()

           # Check Robinhood (optional)
           # rh_status = check_robinhood_api()

           return jsonify({'status': 'healthy'}), 200
       except Exception as e:
           return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
   ```

### SLA Targets

| Metric | Target | Current |
|--------|--------|---------|
| Sync Success Rate | >95% | ~95% |
| API Response Time | <200ms | ~50-100ms |
| Database Query Time | <100ms | ~10-50ms |
| Uptime | 99.9% | - |

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.9+ | Service implementation |
| Database | PostgreSQL | 14+ | Data storage |
| API Client | robin_stocks | latest | Robinhood integration |
| DB Driver | psycopg2 | 2.9+ | PostgreSQL connector |
| CLI | argparse | stdlib | Command-line interface |
| Logging | logging | stdlib | Application logging |

### Optional Technologies (Future)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | Flask/FastAPI | REST API |
| Cache | Redis | Performance optimization |
| Queue | RabbitMQ/Celery | Background jobs |
| Monitoring | Prometheus/Grafana | Metrics & dashboards |
| Scheduler | APScheduler | Task scheduling |

---

## Deployment Architecture

### Production Deployment

```
┌────────────────────────────────────────────────────────┐
│                    PRODUCTION                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────┐         ┌──────────────┐            │
│  │   Load       │         │  PostgreSQL  │            │
│  │   Balancer   │────────>│   Primary    │            │
│  └──────┬───────┘         └──────┬───────┘            │
│         │                        │                     │
│  ┌──────┴───────┐                │                     │
│  │              │                ▼                     │
│  ▼              ▼         ┌──────────────┐            │
│  ┌──────────┐  ┌──────────┐  PostgreSQL  │            │
│  │  App     │  │  App     │   Replica    │            │
│  │ Server 1 │  │ Server 2 │              │            │
│  └──────────┘  └──────────┘ └──────────────┘            │
│                                                        │
│  ┌──────────────────────────────────────┐             │
│  │   Scheduler (Cron/APScheduler)       │             │
│  │   - Daily sync: 6 AM                 │             │
│  │   - Health check: Every 5 min        │             │
│  └──────────────────────────────────────┘             │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Conclusion

The Earnings Sync Service is designed with:
- **Robustness**: Comprehensive error handling and retry logic
- **Scalability**: Batch processing, indexing, and future microservices ready
- **Maintainability**: Clean code, clear separation of concerns, extensive documentation
- **Performance**: Optimized queries, batch operations, caching strategies
- **Reliability**: Health checks, monitoring, backup strategies

This architecture supports current needs (1000 stocks) and can scale to 10,000+ stocks with minimal modifications.

---

**Version:** 1.0.0
**Last Updated:** 2025-10-28
**Author:** Wheel Strategy System Team
