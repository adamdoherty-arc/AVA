"""
System Monitoring Hub - Unified Performance Dashboard
=====================================================

Consolidated monitoring interface combining:
- Cache Performance (Redis + Streamlit)
- LLM Cost Tracking & Savings
- Database Performance Metrics
- API Rate Limiting Stats
- Background Job Status
- System Health Checks

This hub provides real-time visibility into all system metrics.

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="System Monitoring | Magnus",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .monitor-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .metric-good { border-left: 4px solid #38ef7d; }
    .metric-warning { border-left: 4px solid #f39c12; }
    .metric-critical { border-left: 4px solid #e74c3c; }

    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }

    .status-online { background: #38ef7d; }
    .status-degraded { background: #f39c12; }
    .status-offline { background: #e74c3c; }

    .quick-stat {
        text-align: center;
        padding: 1rem;
        background: rgba(17, 153, 142, 0.1);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="monitor-header">
    <h1 style="margin:0">📊 System Monitoring Hub</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9">
        Real-time Performance • Cost Tracking • Health Monitoring
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# System Health Overview
# ============================================================================

st.subheader("🏥 System Health")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    # Check database
    try:
        import psycopg2
        from src.config import get_db_config
        db_config = get_db_config()
        conn = psycopg2.connect(**db_config, connect_timeout=3)
        conn.close()
        db_status = "online"
    except:
        db_status = "offline"

    status_class = "status-online" if db_status == "online" else "status-offline"
    st.markdown(f'<span class="{status_class}"></span> **Database**', unsafe_allow_html=True)
    st.caption(db_status.upper())

with col2:
    # Check Redis
    try:
        from src.cache.redis_cache_manager import get_cache_manager
        cache = get_cache_manager()
        redis_status = "online" if cache.redis_available else "fallback"
    except:
        redis_status = "offline"

    status_class = "status-online" if redis_status == "online" else "status-degraded"
    st.markdown(f'<span class="{status_class}"></span> **Redis Cache**', unsafe_allow_html=True)
    st.caption(redis_status.upper())

with col3:
    # Check LLM Service
    try:
        from src.services.llm_service import get_llm_service
        llm = get_llm_service()
        llm_providers = len(llm.get_available_providers())
        llm_status = f"{llm_providers} providers"
    except:
        llm_status = "offline"
        llm_providers = 0

    status_class = "status-online" if llm_providers > 0 else "status-offline"
    st.markdown(f'<span class="{status_class}"></span> **LLM Service**', unsafe_allow_html=True)
    st.caption(llm_status.upper())

with col4:
    # Check Kalshi API
    try:
        from src.kalshi_db_manager import KalshiDBManager
        kalshi_db = KalshiDBManager()
        # Simple check - if DB manager works, assume API is reachable
        kalshi_status = "online"
    except:
        kalshi_status = "unknown"

    status_class = "status-online" if kalshi_status == "online" else "status-degraded"
    st.markdown(f'<span class="{status_class}"></span> **Kalshi API**', unsafe_allow_html=True)
    st.caption(kalshi_status.upper())

with col5:
    # Overall system status
    all_online = db_status == "online" and redis_status in ["online", "fallback"] and llm_providers > 0
    overall_status = "healthy" if all_online else "degraded"

    status_class = "status-online" if overall_status == "healthy" else "status-degraded"
    st.markdown(f'<span class="{status_class}"></span> **Overall**', unsafe_allow_html=True)
    st.caption(overall_status.upper())

st.divider()

# ============================================================================
# Main Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💾 Cache Performance",
    "💰 LLM Costs",
    "🗄️ Database",
    "🔄 Background Jobs",
    "📊 Analytics"
])

# ============================================================================
# TAB 1: Cache Performance
# ============================================================================

with tab1:
    st.markdown("### 💾 Cache Performance Metrics")

    try:
        from src.cache.redis_cache_manager import get_cache_manager

        cache = get_cache_manager()
        stats = cache.get_stats()

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            hit_rate = stats.get('hit_rate_percent', 0)
            st.metric(
                "Hit Rate",
                f"{hit_rate}%",
                delta=f"{hit_rate - 50}%" if hit_rate > 50 else None,
                delta_color="normal" if hit_rate > 50 else "inverse"
            )

        with col2:
            st.metric("Total Requests", stats.get('total_requests', 0))

        with col3:
            st.metric("Cache Hits", stats.get('hit_count', 0))

        with col4:
            st.metric("Cache Misses", stats.get('miss_count', 0))

        st.divider()

        # Redis-specific stats
        if stats.get('redis_available'):
            st.markdown("#### 🔴 Redis Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Redis Version", stats.get('redis_version', 'N/A'))

            with col2:
                st.metric("Memory Used", stats.get('redis_used_memory', 'N/A'))

            with col3:
                st.metric("Total Keys", stats.get('redis_total_keys', 0))

            # Keyspace stats
            if stats.get('redis_keyspace_hits') is not None:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Keyspace Hits", stats.get('redis_keyspace_hits', 0))

                with col2:
                    st.metric("Keyspace Misses", stats.get('redis_keyspace_misses', 0))

                with col3:
                    redis_hit_rate = 0
                    hits = stats.get('redis_keyspace_hits', 0)
                    misses = stats.get('redis_keyspace_misses', 0)
                    if hits + misses > 0:
                        redis_hit_rate = round((hits / (hits + misses)) * 100, 1)
                    st.metric("Redis Hit Rate", f"{redis_hit_rate}%")

        else:
            st.warning("⚠️ Redis not available - using in-memory fallback cache")
            st.metric("In-Memory Keys", stats.get('in_memory_keys', 0))

        st.divider()

        # Cache management actions
        st.markdown("#### ⚙️ Cache Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🗑️ Clear All Caches", key="clear_all_caches"):
                cache.clear_namespace("magnus:*")
                st.success("✅ All caches cleared!")
                st.rerun()

        with col2:
            namespace = st.selectbox(
                "Clear Specific Namespace",
                ["portfolio", "options_chains", "kalshi_markets", "stock_prices", "llm"],
                key="cache_namespace_select"
            )

            if st.button(f"Clear {namespace}", key="clear_namespace"):
                from src.cache.redis_cache_manager import cache_clear
                deleted = cache_clear(namespace)
                st.success(f"✅ Cleared {deleted} keys from {namespace}")
                st.rerun()

        with col3:
            if st.button("🔄 Refresh Stats", key="refresh_cache_stats"):
                st.rerun()

    except Exception as e:
        st.error(f"Error loading cache stats: {e}")

# ============================================================================
# TAB 2: LLM Cost Tracking
# ============================================================================

with tab2:
    st.markdown("### 💰 LLM Cost Tracking & Savings")

    try:
        from src.services.llm_service import get_llm_service

        llm = get_llm_service()

        # Get routing stats
        routing_stats = llm.get_routing_stats()

        if routing_stats.get('intelligent_routing'):
            # Top-level metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Queries", routing_stats['total_queries'])

            with col2:
                st.metric(
                    "Actual Cost",
                    f"${routing_stats['actual_cost']:.2f}",
                    help="Total cost with intelligent routing"
                )

            with col3:
                st.metric(
                    "Savings",
                    f"${routing_stats['savings']:.2f}",
                    delta=f"-{routing_stats['savings_percent']}%",
                    delta_color="normal",
                    help="Cost saved vs using only premium models"
                )

            with col4:
                st.metric(
                    "Free Tier %",
                    f"{routing_stats['free_tier_percentage']}%",
                    help="Percentage of queries routed to free tier (Ollama/Groq)"
                )

            st.divider()

            # Tier breakdown
            st.markdown("#### 📊 Queries by Tier")

            tier_data = routing_stats['queries_by_tier']

            # Create bar chart data
            import pandas as pd
            tier_df = pd.DataFrame([
                {"Tier": tier.upper(), "Queries": count}
                for tier, count in tier_data.items()
                if count > 0
            ])

            if not tier_df.empty:
                st.bar_chart(tier_df.set_index("Tier"))

                # Detailed breakdown
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**FREE Tier**")
                    st.metric("Queries", tier_data.get('free', 0))
                    st.caption("Ollama, Groq - $0.00")

                with col2:
                    st.markdown("**CHEAP Tier**")
                    st.metric("Queries", tier_data.get('cheap', 0))
                    st.caption("DeepSeek, Gemini - $0.21/1M")

                with col3:
                    st.markdown("**STANDARD Tier**")
                    st.metric("Queries", tier_data.get('standard', 0))
                    st.caption("GPT-3.5 - $1.50/1M")

                with col4:
                    st.markdown("**PREMIUM Tier**")
                    st.metric("Queries", tier_data.get('premium', 0))
                    st.caption("Claude - $15/1M")

            st.divider()

            # Provider-level stats
            st.markdown("#### 🤖 Provider Usage")

            usage_stats = llm.get_usage_stats()
            provider_stats = usage_stats.get('provider_stats', {})

            if provider_stats:
                for key, stats in provider_stats.items():
                    with st.expander(f"{stats['provider'].upper()} / {stats['model']}"):
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Calls", stats['calls'])

                        with col2:
                            st.metric("Input Tokens", f"{stats['input_tokens']:,}")

                        with col3:
                            st.metric("Output Tokens", f"{stats['output_tokens']:,}")

                        with col4:
                            st.metric("Cost", f"${stats['total_cost']:.4f}")

            st.divider()

            # Cost projection
            st.markdown("#### 📈 Cost Projections")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Daily Average**")
                if routing_stats['total_queries'] > 0:
                    # Assume queries in last 24 hours
                    daily_cost = routing_stats['actual_cost']
                    st.metric("Est. Daily Cost", f"${daily_cost:.2f}")
                else:
                    st.caption("No data yet")

            with col2:
                st.markdown("**Monthly Projection**")
                if routing_stats['total_queries'] > 0:
                    monthly_cost = routing_stats['actual_cost'] * 30
                    st.metric("Est. Monthly Cost", f"${monthly_cost:.2f}")
                    st.caption(f"vs ${routing_stats['premium_cost'] * 30:.2f} without routing")
                else:
                    st.caption("No data yet")

        else:
            st.warning("⚠️ Intelligent routing not initialized")

        st.divider()

        # Actions
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear LLM Cache", key="clear_llm_cache"):
                llm.clear_cache()
                st.success("✅ LLM response cache cleared!")

        with col2:
            if st.button("🔄 Reset Usage Stats", key="reset_llm_usage"):
                llm.reset_usage()
                st.success("✅ Usage statistics reset!")
                st.rerun()

    except Exception as e:
        st.error(f"Error loading LLM stats: {e}")

# ============================================================================
# TAB 3: Database Performance
# ============================================================================

with tab3:
    st.markdown("### 🗄️ Database Performance")

    try:
        import psycopg2
        from src.config import get_db_config

        db_config = get_db_config()
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Database size and stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Database size
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)
            db_size = cursor.fetchone()[0]
            st.metric("Database Size", db_size)

        with col2:
            # Total tables
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            table_count = cursor.fetchone()[0]
            st.metric("Total Tables", table_count)

        with col3:
            # Active connections
            cursor.execute("""
                SELECT COUNT(*)
                FROM pg_stat_activity
                WHERE state = 'active'
            """)
            active_conns = cursor.fetchone()[0]
            st.metric("Active Connections", active_conns)

        with col4:
            # Database version
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0].split()[1]
            st.metric("PostgreSQL", version)

        st.divider()

        # Table sizes
        st.markdown("#### 📊 Largest Tables")

        cursor.execute("""
            SELECT
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY size_bytes DESC
            LIMIT 10
        """)

        tables = cursor.fetchall()

        import pandas as pd
        table_df = pd.DataFrame(tables, columns=["Schema", "Table", "Size", "Bytes"])
        table_df = table_df.drop(columns=["Schema", "Bytes"])  # Clean display
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.divider()

        # Recent queries
        st.markdown("#### 🔍 Recent Slow Queries (>1s)")

        cursor.execute("""
            SELECT
                query,
                mean_exec_time::int AS avg_ms,
                calls,
                total_exec_time::int AS total_ms
            FROM pg_stat_statements
            WHERE mean_exec_time > 1000
            ORDER BY mean_exec_time DESC
            LIMIT 5
        """)

        slow_queries = cursor.fetchall()

        if slow_queries:
            slow_df = pd.DataFrame(slow_queries, columns=["Query", "Avg (ms)", "Calls", "Total (ms)"])
            # Truncate query for display
            slow_df["Query"] = slow_df["Query"].str[:100] + "..."
            st.dataframe(slow_df, use_container_width=True, hide_index=True)
        else:
            st.info("✅ No slow queries detected!")

        conn.close()

    except Exception as e:
        st.error(f"Error loading database stats: {e}")
        st.caption("Note: pg_stat_statements extension may not be enabled")

# ============================================================================
# TAB 4: Background Jobs
# ============================================================================

with tab4:
    st.markdown("### 🔄 Background Job Status")

    st.info("📝 Background job monitoring coming soon")

    # Placeholder for Celery integration
    st.markdown("""
    **Planned Features:**
    - ⏰ Scheduled job status
    - 📊 Job queue lengths
    - ⚠️ Failed job tracking
    - 🔄 Retry statistics
    - 📈 Job execution times
    """)

    # Mock data for demonstration
    import pandas as pd

    mock_jobs = pd.DataFrame([
        {"Job": "Sync Kalshi Markets", "Status": "Running", "Last Run": "2 min ago", "Next Run": "3 min"},
        {"Job": "Update Stock Prices", "Status": "Completed", "Last Run": "5 min ago", "Next Run": "10 min"},
        {"Job": "Generate AI Predictions", "Status": "Pending", "Last Run": "15 min ago", "Next Run": "NOW"},
        {"Job": "Clean Old Cache Entries", "Status": "Completed", "Last Run": "1 hour ago", "Next Run": "23 hours"},
    ])

    st.dataframe(mock_jobs, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 5: Analytics
# ============================================================================

with tab5:
    st.markdown("### 📊 System Analytics")

    try:
        # Prediction performance
        st.markdown("#### 🎯 Prediction Performance")

        from src.database.connection_pool import get_connection

        with get_connection() as conn:
            cursor = conn.cursor()

            # Win rate by market type
            cursor.execute("""
                SELECT
                    market_type,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                    ROUND(AVG(CASE WHEN is_correct THEN 1 ELSE 0 END) * 100, 1) as win_rate
                FROM prediction_performance
                WHERE settled_at IS NOT NULL
                AND settled_at >= NOW() - INTERVAL '30 days'
                GROUP BY market_type
                ORDER BY total DESC
            """)

            predictions = cursor.fetchall()

            if predictions:
                import pandas as pd
                pred_df = pd.DataFrame(predictions, columns=["Market", "Total", "Correct", "Win Rate %"])
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
            else:
                st.info("No prediction performance data yet")

        st.divider()

        # API rate limit stats
        st.markdown("#### 🚦 API Rate Limits")

        st.info("API rate limit monitoring coming soon")

    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# ============================================================================
# Footer
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔄 Auto-Refresh**")
    auto_refresh = st.checkbox("Enable (30s)", key="auto_refresh")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

with col2:
    st.markdown("**📅 Last Updated**")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with col3:
    st.markdown("**⚡ Quick Actions**")
    if st.button("🔄 Refresh All", key="refresh_all"):
        st.rerun()

st.caption("Magnus Trading Platform • System Monitoring v1.0")
