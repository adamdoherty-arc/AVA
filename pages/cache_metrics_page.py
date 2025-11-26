"""
Cache Performance Metrics Dashboard
Monitor caching effectiveness and performance improvements
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time

def show_cache_metrics():
    """Display cache performance metrics and statistics"""

    st.title("🔍 Cache Performance Metrics")
    st.caption("Monitor caching effectiveness and performance improvements across the platform")

    # Initialize metrics in session state
    if 'cache_metrics' not in st.session_state:
        st.session_state.cache_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'queries_executed': 0,
            'time_saved_seconds': 0.0,
            'startup_time': datetime.now()
        }

    metrics = st.session_state.cache_metrics

    # Calculate derived metrics
    total_requests = metrics['cache_hits'] + metrics['cache_misses']
    hit_rate = (metrics['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
    uptime = (datetime.now() - metrics['startup_time']).total_seconds() / 3600  # hours

    # Estimated performance metrics
    avg_uncached_time = 2.5  # seconds
    avg_cached_time = 0.05   # seconds
    time_saved = metrics['cache_hits'] * (avg_uncached_time - avg_cached_time)

    # Top Metrics Row
    st.markdown("### 📊 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Cache Hit Rate",
            f"{hit_rate:.1f}%",
            delta=f"Target: 85%",
            delta_color="normal" if hit_rate >= 85 else "inverse"
        )

    with col2:
        st.metric(
            "Time Saved",
            f"{time_saved:.1f}s",
            delta=f"This session"
        )

    with col3:
        st.metric(
            "Queries Eliminated",
            f"{metrics['cache_hits']:,}",
            delta=f"Since startup"
        )

    with col4:
        st.metric(
            "Uptime",
            f"{uptime:.1f}h",
            delta=f"Running"
        )


    # Detailed Statistics
    st.markdown("### 📈 Detailed Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cache Performance")

        perf_data = pd.DataFrame({
            'Metric': ['Cache Hits', 'Cache Misses', 'Total Requests', 'Hit Rate'],
            'Value': [
                f"{metrics['cache_hits']:,}",
                f"{metrics['cache_misses']:,}",
                f"{total_requests:,}",
                f"{hit_rate:.2f}%"
            ]
        })

        st.dataframe(perf_data, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### Performance Impact")

        impact_data = pd.DataFrame({
            'Metric': [
                'Avg Query Time (Uncached)',
                'Avg Query Time (Cached)',
                'Speedup Factor',
                'Total Time Saved'
            ],
            'Value': [
                f"{avg_uncached_time:.2f}s",
                f"{avg_cached_time:.3f}s",
                f"{avg_uncached_time / avg_cached_time:.0f}x",
                f"{time_saved:.1f}s"
            ]
        })

        st.dataframe(impact_data, hide_index=True, use_container_width=True)


    # Cache Status by Page
    st.markdown("### 📄 Cache Status by Feature")

    # Mock data for different pages (in production, track per-page)
    page_metrics = pd.DataFrame({
        'Page': [
            'Positions',
            'XTrades Watchlists',
            'Supply/Demand Zones',
            'Calendar Spreads',
            'Kalshi NFL Markets',
            'Options Analysis',
            'Game Cards',
            'Premium Flow'
        ],
        'Cache Hit Rate': [92, 88, 85, 91, 87, 90, 83, 86],
        'Queries Eliminated': [45, 38, 32, 29, 41, 36, 28, 31],
        'Avg Load Time': ['0.65s', '0.72s', '0.78s', '0.68s', '0.71s', '0.69s', '0.81s', '0.74s']
    })

    st.dataframe(
        page_metrics.style.background_gradient(subset=['Cache Hit Rate'], cmap='RdYlGn', vmin=80, vmax=100),
        hide_index=True,
        use_container_width=True
    )


    # Actions
    st.markdown("### ⚙️ Cache Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Clear All Caches", type="secondary", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ All caches cleared!")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("📊 Reset Metrics", type="secondary", use_container_width=True):
            st.session_state.cache_metrics = {
                'cache_hits': 0,
                'cache_misses': 0,
                'queries_executed': 0,
                'time_saved_seconds': 0.0,
                'startup_time': datetime.now()
            }
            st.success("✅ Metrics reset!")
            time.sleep(1)
            st.rerun()

    with col3:
        if st.button("🔥 Warm Caches Now", type="primary", use_container_width=True):
            with st.spinner("Warming caches..."):
                # Trigger cache warming
                try:
                    from positions_page_improved import get_closed_trades_cached
                    from xtrades_watchlists_page import get_active_trades_cached
                    get_closed_trades_cached(days_back=7)
                    get_active_trades_cached(limit=100)
                    st.success("✅ Caches warmed!")
                except Exception as e:
                    st.warning(f"⚠️ Partial warming: {e}")
                time.sleep(1)
                st.rerun()


    # Best Practices
    with st.expander("💡 Cache Optimization Tips"):
        st.markdown("""
        **Maximizing Cache Effectiveness:**

        1. **Use Filters Consistently**: Same filters = cache hits
        2. **Avoid Frequent Cache Clears**: Let TTL handle invalidation
        3. **Monitor Hit Rate**: Target 85%+ for optimal performance
        4. **Report Slow Pages**: Help us optimize further

        **Current TTL Settings:**
        - Real-time data: 60 seconds
        - Database queries: 300 seconds (5 minutes)
        - Static data: 600 seconds (10 minutes)
        """)

if __name__ == "__main__":
    show_cache_metrics()
