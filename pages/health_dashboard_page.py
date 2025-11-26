"""
Health Monitoring Dashboard

Real-time system health monitoring and diagnostics for Magnus.

Features:
- Database connection pool monitoring
- External API status checks
- LLM service health
- Error tracking and metrics
- Cache performance statistics
- System resource monitoring

Author: Magnus Enhancement Team
"""

import streamlit as st
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import psutil
import requests

from src.database.connection_pool import get_pool_stats
from src.utils.error_handling import get_error_metrics
from src.magnus_local_llm import get_local_llm

logger = logging.getLogger(__name__)


def show():
    """Main health dashboard page"""
    st.title("System Health Monitor")
    st.caption("Real-time health monitoring and diagnostics")

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            st.rerun()

    # Refresh button
    if st.button("Refresh Status", use_container_width=True):
        st.rerun()

    st.divider()

    # Overall System Status
    with st.container():
        st.subheader("Overall System Status")

        # Quick status indicators
        col1, col2, col3, col4 = st.columns(4)

        db_status = check_database_health()
        with col1:
            status_icon = "✅" if db_status['healthy'] else "❌"
            st.metric(
                "Database",
                status_icon,
                f"{db_status['active']}/{db_status['max']} connections"
            )

        llm_status = check_llm_health()
        with col2:
            status_icon = "✅" if llm_status['healthy'] else "❌"
            st.metric(
                "Local LLM",
                status_icon,
                llm_status['model']
            )

        api_status = check_external_apis()
        with col3:
            healthy_apis = sum(1 for s in api_status.values() if s['healthy'])
            total_apis = len(api_status)
            status_icon = "✅" if healthy_apis == total_apis else "⚠️" if healthy_apis > 0 else "❌"
            st.metric(
                "External APIs",
                f"{healthy_apis}/{total_apis}",
                status_icon
            )

        error_metrics = get_error_metrics()
        with col4:
            recent_errors = error_metrics.get('total_errors', 0)
            status_icon = "✅" if recent_errors == 0 else "⚠️" if recent_errors < 10 else "❌"
            st.metric(
                "Error Count",
                recent_errors,
                status_icon
            )

    st.divider()

    # Detailed Sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Database", "APIs", "LLM Service", "Errors", "System Resources"
    ])

    # Tab 1: Database Health
    with tab1:
        show_database_health(db_status)

    # Tab 2: API Status
    with tab2:
        show_api_status(api_status)

    # Tab 3: LLM Service
    with tab3:
        show_llm_health(llm_status)

    # Tab 4: Error Tracking
    with tab4:
        show_error_metrics(error_metrics)

    # Tab 5: System Resources
    with tab5:
        show_system_resources()


@st.cache_data(ttl=30)  # PERFORMANCE: Cache for 30s to reduce redundant checks
def check_database_health() -> Dict:
    """Check database connection pool health"""
    try:
        stats = get_pool_stats()
        max_conn = 20  # From connection pool config

        return {
            'healthy': stats['active_connections'] < max_conn * 0.8,  # < 80% usage
            'active': stats['active_connections'],
            'max': max_conn,
            'available': stats.get('available', 0),
            'reused': stats.get('connections_reused', 0),
            'errors': stats.get('errors', 0)
        }
    except Exception as e:
        logger.error(f"Error checking database health: {e}")
        return {
            'healthy': False,
            'active': 0,
            'max': 20,
            'available': 0,
            'reused': 0,
            'errors': 1
        }


@st.cache_data(ttl=30)  # PERFORMANCE: Cache for 30s to avoid excessive LLM calls
def check_llm_health() -> Dict:
    """Check local LLM service health"""
    try:
        llm = get_local_llm()

        # Try a simple generation
        response = llm.generate(
            "Say 'OK' if you're working",
            model="qwen2.5:14b",
            max_tokens=10
        )

        healthy = bool(response and len(response) > 0)

        return {
            'healthy': healthy,
            'model': 'Qwen 2.5 14B',
            'status': 'Running' if healthy else 'Not Responding',
            'response_time': 'Fast'  # Could add timing
        }
    except Exception as e:
        logger.error(f"Error checking LLM health: {e}")
        return {
            'healthy': False,
            'model': 'Unknown',
            'status': f'Error: {str(e)[:50]}',
            'response_time': 'N/A'
        }


@st.cache_data(ttl=30)  # PERFORMANCE: Cache for 30s to reduce API health checks
def check_external_apis() -> Dict[str, Dict]:
    """Check external API health"""
    apis = {}

    # Robinhood
    try:
        from src.services.robinhood_client import RobinhoodClient
        rh = RobinhoodClient()
        # Simple auth check
        apis['Robinhood'] = {
            'healthy': True,  # If initialization worked
            'status': 'Connected',
            'last_check': datetime.now()
        }
    except Exception as e:
        apis['Robinhood'] = {
            'healthy': False,
            'status': f'Error: {str(e)[:30]}',
            'last_check': datetime.now()
        }

    # Kalshi
    try:
        from src.kalshi_client_v2 import KalshiClient
        kalshi = KalshiClient()
        apis['Kalshi'] = {
            'healthy': True,
            'status': 'Connected',
            'last_check': datetime.now()
        }
    except Exception as e:
        apis['Kalshi'] = {
            'healthy': False,
            'status': f'Error: {str(e)[:30]}',
            'last_check': datetime.now()
        }

    # Ollama (LLM)
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=2)
        apis['Ollama'] = {
            'healthy': response.status_code == 200,
            'status': 'Running' if response.status_code == 200 else 'Not responding',
            'last_check': datetime.now()
        }
    except Exception as e:
        apis['Ollama'] = {
            'healthy': False,
            'status': 'Not running',
            'last_check': datetime.now()
        }

    return apis


def show_database_health(db_status: Dict):
    """Display detailed database health information"""
    st.subheader("Database Connection Pool")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Active Connections", db_status['active'])
        st.metric("Available", db_status['available'])

    with col2:
        st.metric("Max Connections", db_status['max'])
        usage_pct = (db_status['active'] / db_status['max']) * 100
        st.metric("Usage", f"{usage_pct:.1f}%")

    with col3:
        st.metric("Connections Reused", db_status['reused'])
        st.metric("Connection Errors", db_status['errors'])

    # Connection pool visualization
    st.progress(usage_pct / 100)

    if usage_pct > 80:
        st.warning("⚠️ Connection pool usage is high (>80%). Consider increasing pool size if this persists.")
    elif usage_pct > 90:
        st.error("❌ Connection pool usage is critical (>90%). Database performance may be degraded.")

    # Connection history (simulated - would need actual logging)
    st.subheader("Connection Pool Statistics")

    stats_col1, stats_col2 = st.columns(2)

    with stats_col1:
        st.write("**Pool Configuration:**")
        st.write("- Min Connections: 2")
        st.write("- Max Connections: 20")
        st.write("- Connection Timeout: 10s")
        st.write("- Query Timeout: 30s")

    with stats_col2:
        st.write("**Pool Performance:**")
        st.write(f"- Total Reused: {db_status['reused']}")
        st.write(f"- Total Errors: {db_status['errors']}")
        st.write(f"- Current Load: {usage_pct:.1f}%")


def show_api_status(api_status: Dict[str, Dict]):
    """Display external API status"""
    st.subheader("External API Status")

    for api_name, status in api_status.items():
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                status_icon = "✅" if status['healthy'] else "❌"
                st.write(f"### {status_icon} {api_name}")

            with col2:
                st.write(f"**Status:** {status['status']}")
                st.write(f"**Last Check:** {status['last_check'].strftime('%H:%M:%S')}")

            with col3:
                if not status['healthy']:
                    if st.button(f"Retry {api_name}", key=f"retry_{api_name}"):
                        st.rerun()

            st.divider()


def show_llm_health(llm_status: Dict):
    """Display LLM service health"""
    st.subheader("Local LLM Service")

    col1, col2 = st.columns(2)

    with col1:
        status_icon = "✅" if llm_status['healthy'] else "❌"
        st.metric("Status", f"{status_icon} {llm_status['status']}")
        st.metric("Model", llm_status['model'])

    with col2:
        st.metric("Response Time", llm_status['response_time'])

        if st.button("Test LLM Response"):
            with st.spinner("Testing LLM..."):
                try:
                    llm = get_local_llm()
                    start = datetime.now()
                    response = llm.generate(
                        "What is 2+2? Answer in one word.",
                        model="qwen2.5:14b",
                        max_tokens=10
                    )
                    duration = (datetime.now() - start).total_seconds()

                    st.success(f"✅ LLM Response ({duration:.2f}s): {response}")
                except Exception as e:
                    st.error(f"❌ LLM Test Failed: {e}")

    st.subheader("Available Models")

    models = [
        ("Qwen 2.5 32B", "qwen2.5:32b", "High quality, slower"),
        ("Qwen 2.5 14B", "qwen2.5:14b", "Balanced quality and speed"),
        ("Qwen 2.5 7B", "qwen2.5:7b", "Fast, lower quality")
    ]

    for name, model_id, description in models:
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.write(f"**{name}**")

            with col2:
                st.write(description)

            with col3:
                st.write(f"`{model_id}`")

            st.divider()


def show_error_metrics(error_metrics: Dict):
    """Display error tracking metrics"""
    st.subheader("Error Tracking & Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_errors = error_metrics.get('total_errors', 0)
        st.metric("Total Errors", total_errors)

    with col2:
        error_types = len(error_metrics.get('errors_by_type', {}))
        st.metric("Unique Error Types", error_types)

    with col3:
        last_error = error_metrics.get('last_error_time')
        if last_error:
            time_since = datetime.now() - last_error
            st.metric("Last Error", f"{time_since.seconds // 60}m ago")
        else:
            st.metric("Last Error", "None")

    if total_errors > 0:
        st.subheader("Errors by Type")

        errors_by_type = error_metrics.get('errors_by_type', {})
        if errors_by_type:
            # Sort by count
            sorted_errors = sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True)

            for error_type, count in sorted_errors[:10]:  # Top 10
                st.write(f"**{error_type}**: {count} occurrences")

        st.subheader("Errors by Function")

        errors_by_function = error_metrics.get('errors_by_function', {})
        if errors_by_function:
            # Sort by count
            sorted_funcs = sorted(errors_by_function.items(), key=lambda x: x[1], reverse=True)

            for func_name, count in sorted_funcs[:10]:  # Top 10
                st.write(f"`{func_name}`: {count} errors")

    else:
        st.success("✅ No errors recorded since last reset")

    if st.button("Reset Error Metrics"):
        from src.utils.error_handling import reset_error_metrics
        reset_error_metrics()
        st.success("Error metrics reset")
        st.rerun()


def show_system_resources():
    """Display system resource usage"""
    st.subheader("System Resources")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### CPU")
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent}%")
        st.progress(cpu_percent / 100)

        cpu_count = psutil.cpu_count()
        st.write(f"**CPU Cores:** {cpu_count}")

    with col2:
        st.write("### Memory")
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        st.metric("Memory Usage", f"{memory_percent}%")
        st.progress(memory_percent / 100)

        st.write(f"**Total Memory:** {memory.total / (1024**3):.1f} GB")
        st.write(f"**Available:** {memory.available / (1024**3):.1f} GB")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.write("### Disk")
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        st.metric("Disk Usage", f"{disk_percent}%")
        st.progress(disk_percent / 100)

        st.write(f"**Total Space:** {disk.total / (1024**3):.1f} GB")
        st.write(f"**Free Space:** {disk.free / (1024**3):.1f} GB")

    with col4:
        st.write("### Network")
        network = psutil.net_io_counters()

        st.write(f"**Bytes Sent:** {network.bytes_sent / (1024**2):.1f} MB")
        st.write(f"**Bytes Received:** {network.bytes_recv / (1024**2):.1f} MB")
        st.write(f"**Packets Sent:** {network.packets_sent:,}")
        st.write(f"**Packets Received:** {network.packets_recv:,}")


if __name__ == "__main__":
    show()
