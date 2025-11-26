# Performance Enhancements Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the new performance enhancements into the AVA trading platform. These enhancements improve page load times, reduce database load, and provide a better user experience.

---

## Quick Start

### 1. Using Pagination

Add pagination to any large table to improve rendering performance:

```python
from src.components.pagination_component import paginate_dataframe

# Original code:
# st.dataframe(df)

# Enhanced code with pagination:
paginated = paginate_dataframe(df, page_size=50, key_prefix="my_table")
st.dataframe(paginated)
```

**Parameters:**
- `df`: DataFrame to paginate
- `page_size`: Number of rows per page (default: 50)
- `key_prefix`: Unique identifier for session state (required)
- `show_info`: Whether to show page info (default: True)

**Example with custom settings:**
```python
paginated = paginate_dataframe(
    df,
    page_size=25,
    key_prefix="opportunities_table",
    show_info=True
)
st.dataframe(paginated)
```

---

### 2. Using Error Handling

Gracefully handle errors with user-friendly messages:

```python
from src.utils.error_handling import with_error_handling, safe_cache_data

# Decorator for any function
@with_error_handling(fallback_value=[])
def my_query():
    return database.query()

# Combined with caching
@safe_cache_data(ttl=300)
def cached_query():
    return database.expensive_query()
```

**Error Types Handled:**
- `APIRateLimitError`: Shows rate limit warning
- `DataNotAvailableError`: Shows temporary unavailability message
- `ConnectionError`: Shows connection error
- `Exception`: Generic error handling

**Custom error messages:**
```python
@with_error_handling(
    fallback_value=None,
    show_error=True,
    error_message="Failed to load market data"
)
def get_market_data():
    return api.fetch()
```

---

### 3. Using Progressive Loading

Load critical data first, then load secondary data progressively:

```python
from src.utils.progressive_loading import load_progressively

def load_positions():
    st.subheader("Positions")
    # Load and display positions
    return positions_df

def load_charts():
    st.subheader("Charts")
    # Load and display charts
    return chart_data

def load_stats():
    st.subheader("Statistics")
    # Load and display stats
    return stats_data

# Load progressively
load_progressively([
    ("Positions", load_positions, True),   # Critical - loads first
    ("Charts", load_charts, False),        # Non-critical - loads after
    ("Stats", load_stats, False)           # Non-critical - loads last
])
```

**Class-based approach:**
```python
from src.utils.progressive_loading import ProgressiveLoader

loader = ProgressiveLoader()
loader.add_section("Positions", load_positions, critical=True)
loader.add_section("Charts", load_charts, critical=False)
loader.add_section("Stats", load_stats, critical=False)
loader.load_all()
```

**With timeout protection:**
```python
from src.utils.progressive_loading import load_with_timeout

result = load_with_timeout(
    loader=expensive_function,
    timeout=5.0,
    fallback=[]
)
```

---

### 4. Using Redis Cache (Optional)

Distributed caching for multi-instance deployments:

```python
from src.utils.redis_cache import cache_with_redis

@cache_with_redis("api_data", ttl=60)
def fetch_api():
    return api.fetch()
```

**Manual cache operations:**
```python
from src.utils.redis_cache import get_redis_cache, get_cache_stats

cache = get_redis_cache()

# Set value
cache.set("key", "value", ttl=300)

# Get value
value = cache.get("key")

# Delete key
cache.delete("key")

# Clear pattern
cache.clear_pattern("market_data:*")

# Get stats
stats = get_cache_stats()
print(stats)
```

---

## Configuration

### Redis Setup (Optional)

Add to `.env`:
```
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

Redis will only be used if:
1. Redis is installed (`pip install redis`)
2. `REDIS_ENABLED=true` in `.env`
3. Redis server is running and accessible

If Redis is unavailable, the system gracefully falls back to local Streamlit caching.

---

## Implementation Examples

### Example 1: Adding Pagination to a Page

**Before:**
```python
def display_opportunities():
    df = get_opportunities()
    st.dataframe(df)
```

**After:**
```python
from src.components.pagination_component import paginate_dataframe

def display_opportunities():
    df = get_opportunities()
    paginated = paginate_dataframe(df, page_size=50, key_prefix="opportunities")
    st.dataframe(paginated)
```

---

### Example 2: Adding Error Handling to Database Query

**Before:**
```python
@st.cache_data(ttl=300)
def get_trades():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trades")
    return cur.fetchall()
```

**After:**
```python
from src.utils.error_handling import safe_cache_data

@safe_cache_data(ttl=300)
def get_trades():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM trades")
    return cur.fetchall()
```

---

### Example 3: Progressive Page Loading

**Before:**
```python
def display_dashboard():
    st.title("Dashboard")

    # All load at once
    positions = load_positions()
    display_positions(positions)

    charts = load_charts()
    display_charts(charts)

    stats = load_stats()
    display_stats(stats)
```

**After:**
```python
from src.utils.progressive_loading import load_progressively

def display_dashboard():
    st.title("Dashboard")

    def load_and_display_positions():
        positions = load_positions()
        display_positions(positions)

    def load_and_display_charts():
        charts = load_charts()
        display_charts(charts)

    def load_and_display_stats():
        stats = load_stats()
        display_stats(stats)

    # Load critical first, then others progressively
    load_progressively([
        ("Positions", load_and_display_positions, True),
        ("Charts", load_and_display_charts, False),
        ("Stats", load_and_display_stats, False)
    ])
```

---

### Example 4: Redis Caching for API Calls

**Before:**
```python
@st.cache_data(ttl=60)
def get_market_data(symbol):
    return api.fetch(symbol)
```

**After:**
```python
from src.utils.redis_cache import cache_with_redis

@cache_with_redis("market_data", ttl=60)
def get_market_data(symbol):
    return api.fetch(symbol)
```

This provides distributed caching if Redis is available, with automatic fallback to local caching.

---

## Performance Best Practices

### 1. Caching Strategy

- **Short TTL (60s)**: Real-time data (prices, positions)
- **Medium TTL (300s)**: Semi-static data (stocks list, options chains)
- **Long TTL (600s+)**: Static data (company info, ETFs)

### 2. Pagination Guidelines

- Use pagination for tables with **>50 rows**
- Page size recommendations:
  - **25 rows**: Dense data (many columns)
  - **50 rows**: Standard tables
  - **100 rows**: Simple data (few columns)

### 3. Progressive Loading

Load in this order:
1. **Critical**: User positions, account balance
2. **High Priority**: Current opportunities, alerts
3. **Medium Priority**: Charts, analysis
4. **Low Priority**: Historical data, statistics

### 4. Error Handling

Always use error handling for:
- Database queries
- External API calls
- File operations
- Data processing that might fail

---

## Testing

Run the automated test suite:

```bash
pytest tests/test_cache_performance.py -v
```

**Tests included:**
- Cache speed improvement verification
- Data consistency checks
- Error handling validation
- Pagination component functionality
- Progressive loading utilities
- Redis cache operations

---

## Monitoring

### Check Cache Performance

```python
from src.utils.redis_cache import get_cache_stats

stats = get_cache_stats()
st.json(stats)
```

### Check Page Load Times

Add timing to your pages:

```python
import time

start = time.time()
load_page_data()
load_time = time.time() - start

st.caption(f"Loaded in {load_time:.2f}s")
```

---

## Troubleshooting

### Pagination Not Working

**Issue**: Pagination buttons not responding

**Solution**: Ensure `key_prefix` is unique per table
```python
# Wrong - same key used twice
paginate_dataframe(df1, key_prefix="table")
paginate_dataframe(df2, key_prefix="table")  # ❌

# Right - unique keys
paginate_dataframe(df1, key_prefix="trades_table")
paginate_dataframe(df2, key_prefix="opportunities_table")  # ✅
```

---

### Redis Connection Errors

**Issue**: Redis connection failed warnings

**Solution**:
1. Check Redis is running: `redis-cli ping`
2. Verify `.env` settings
3. Check firewall/network access
4. System will automatically fall back to local caching

---

### Cache Not Clearing

**Issue**: Cached data not updating after changes

**Solution**:
```python
# Clear specific Streamlit cache
@st.cache_data
def my_function():
    pass

my_function.clear()

# Clear all Streamlit caches
st.cache_data.clear()

# Clear Redis cache
from src.utils.redis_cache import clear_redis_cache
clear_redis_cache("pattern:*")
```

---

## Migration Checklist

When adding enhancements to a page:

- [ ] Import required modules
- [ ] Add pagination to large tables (>50 rows)
- [ ] Wrap database queries with error handling
- [ ] Add progressive loading if page has multiple sections
- [ ] Test with and without Redis enabled
- [ ] Update page documentation
- [ ] Add performance monitoring (optional)

---

## Performance Metrics

Expected improvements after implementation:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Initial page load | 3-5s | 1-2s | 60-70% faster |
| Table rendering (1000+ rows) | 5-10s | 1-2s | 80% faster |
| Cached query response | 500ms | <50ms | 90% faster |
| Database load | High | Medium | 40-60% reduction |

---

## Additional Resources

- **Pagination Component**: `src/components/pagination_component.py`
- **Error Handling**: `src/utils/error_handling.py`
- **Progressive Loading**: `src/utils/progressive_loading.py`
- **Redis Cache**: `src/utils/redis_cache.py`
- **Test Suite**: `tests/test_cache_performance.py`

---

## Support

For issues or questions:

1. Check this guide first
2. Review test files for examples
3. Check component source code
4. Review existing implementations in:
   - `premium_flow_page.py`
   - `earnings_calendar_page.py`
   - `sector_analysis_page.py`

---

## Summary

These performance enhancements provide:

- **Faster page loads** through progressive loading
- **Better UX** with pagination for large datasets
- **Improved reliability** with error handling
- **Reduced database load** through caching
- **Scalability** with optional Redis support

All enhancements are backward compatible and gracefully degrade if dependencies are unavailable.
