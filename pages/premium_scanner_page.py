"""
Premium Scanner - Advanced Options Premium Analysis
Modern implementation with connection pooling, advanced features, and optimized performance
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict, Any, List
import time
from io import BytesIO

# Import connection pool
from src.database import get_db_connection
from src.stock_data_sync import StockDataSync

logger = logging.getLogger(__name__)

# Note: st.set_page_config is called in dashboard.py, not here
# This page is loaded as a module by dashboard.py


# ============================================================================
# CONFIGURATION
# ============================================================================

class ScannerConfig:
    """Configuration constants for premium scanner"""
    # DTE Ranges
    WEEKLY_DTE_MIN = 5
    WEEKLY_DTE_MAX = 9
    MONTHLY_DTE_MIN = 25
    MONTHLY_DTE_MAX = 35

    # Cache TTL
    OPPORTUNITIES_CACHE_TTL = 60
    SYNC_TIME_CACHE_TTL = 300
    STATS_CACHE_TTL = 60

    # Defaults
    DEFAULT_DELTA_MIN = -0.4
    DEFAULT_DELTA_MAX = -0.2
    DEFAULT_MAX_STOCK_PRICE = 10000
    DEFAULT_MIN_PREMIUM = 0.0
    DEFAULT_MIN_ANNUAL = 0.0
    DEFAULT_MIN_VOLUME = 0

    # Advanced defaults
    DEFAULT_MIN_OPEN_INTEREST = 0
    DEFAULT_MAX_BID_ASK_SPREAD = 5.0
    DEFAULT_IV_RANGE = (0, 200)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_next_friday() -> datetime:
    """
    Get the next Friday's date for 7-day options

    Returns:
        datetime: Next Friday's date
    """
    today = datetime.now()
    days_ahead = 4 - today.weekday()  # Friday is 4
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def initialize_session_state():
    """Initialize session state variables for filters and preferences"""
    if 'scanner_filters' not in st.session_state:
        st.session_state.scanner_filters = {
            'max_stock_price': ScannerConfig.DEFAULT_MAX_STOCK_PRICE,
            'delta_range': (ScannerConfig.DEFAULT_DELTA_MIN, ScannerConfig.DEFAULT_DELTA_MAX),
            'min_premium': ScannerConfig.DEFAULT_MIN_PREMIUM,
            'min_annual_return': ScannerConfig.DEFAULT_MIN_ANNUAL,
            'min_volume': ScannerConfig.DEFAULT_MIN_VOLUME,
            'selected_sectors': [],
            'iv_range': ScannerConfig.DEFAULT_IV_RANGE,
            'min_open_interest': ScannerConfig.DEFAULT_MIN_OPEN_INTEREST,
            'max_bid_ask_spread': ScannerConfig.DEFAULT_MAX_BID_ASK_SPREAD
        }

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []

    if 'selected_for_comparison' not in st.session_state:
        st.session_state.selected_for_comparison = set()


# ============================================================================
# DATA FETCHING WITH OPTIMIZED SQL
# ============================================================================

@st.cache_data(ttl=ScannerConfig.OPPORTUNITIES_CACHE_TTL, show_spinner=False)
def fetch_opportunities(
    dte_min: int,
    dte_max: int,
    delta_min: float = ScannerConfig.DEFAULT_DELTA_MIN,
    delta_max: float = ScannerConfig.DEFAULT_DELTA_MAX,
    min_premium: float = 0.0,
    min_stock_price: float = 0.0,
    max_stock_price: float = 10000.0
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Fetch premium opportunities from database with calculations done in SQL

    Args:
        dte_min: Minimum days to expiration
        dte_max: Maximum days to expiration
        delta_min: Minimum delta value
        delta_max: Maximum delta value
        min_premium: Minimum premium in dollars
        min_stock_price: Minimum stock price filter
        max_stock_price: Maximum stock price filter

    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        with get_db_connection() as conn:
            # Optimized query with calculations in SQL
            query = '''
                WITH ranked_premiums AS (
                    SELECT
                        sp.symbol,
                        sd.current_price as stock_price,
                        sp.strike_price,
                        sp.premium,
                        sp.dte,
                        sp.premium_pct,
                        sp.annual_return,
                        sp.delta,
                        sp.prob_profit,
                        sp.implied_volatility,
                        sp.volume,
                        sp.open_interest,
                        sp.strike_type,
                        sp.expiration_date,
                        sp.bid,
                        sp.ask,
                        s.company_name,
                        s.sector,
                        -- Calculations done in SQL for better performance
                        (sp.premium_pct * 365.0 / NULLIF(sp.dte, 0)) as annualized_52wk,
                        (sp.premium / NULLIF(sp.dte, 0)) as premium_per_day,
                        CASE
                            WHEN sp.ask IS NOT NULL AND sp.bid IS NOT NULL
                            THEN sp.ask - sp.bid
                            ELSE 0
                        END as bid_ask_spread,
                        ROW_NUMBER() OVER (
                            PARTITION BY sp.symbol
                            ORDER BY (sp.premium / NULLIF(sp.dte, 0)) DESC
                        ) as rn
                    FROM stock_premiums sp
                    LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
                    LEFT JOIN stocks s ON sp.symbol = s.symbol
                    WHERE sp.dte BETWEEN %s AND %s
                      AND sp.premium >= %s
                      AND sp.delta BETWEEN %s AND %s
                      AND sp.strike_price > 0
                      AND sp.strike_price < sd.current_price
                      AND (sd.current_price BETWEEN %s AND %s OR sd.current_price IS NULL)
                )
                SELECT
                    symbol, stock_price, strike_price, premium, dte,
                    premium_pct, annual_return, delta, prob_profit,
                    implied_volatility, volume, open_interest, strike_type,
                    expiration_date, bid, ask, company_name, sector,
                    annualized_52wk, premium_per_day, bid_ask_spread
                FROM ranked_premiums
                WHERE rn = 1
                ORDER BY premium_per_day DESC
            '''

            df = pd.read_sql(
                query,
                conn,
                params=(dte_min, dte_max, min_premium, delta_min, delta_max,
                       min_stock_price, max_stock_price)
            )

            return df, None

    except Exception as e:
        error_msg = f"Error fetching opportunities: {str(e)}"
        logger.error(error_msg)
        return pd.DataFrame(), error_msg


@st.cache_data(ttl=ScannerConfig.SYNC_TIME_CACHE_TTL, show_spinner=False)
def get_last_sync_time(dte_min: int, dte_max: int) -> Optional[datetime]:
    """
    Get last sync timestamp for specific DTE range

    Args:
        dte_min: Minimum DTE
        dte_max: Maximum DTE

    Returns:
        Last sync datetime or None
    """
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT MAX(last_updated) as last_sync
                FROM stock_premiums
                WHERE dte BETWEEN %s AND %s
            '''

            df = pd.read_sql(query, conn, params=(dte_min, dte_max))
            return df['last_sync'].iloc[0] if not df.empty and df['last_sync'].iloc[0] else None

    except Exception as e:
        logger.error(f"Error getting last sync time: {e}")
        return None


@st.cache_data(ttl=ScannerConfig.STATS_CACHE_TTL, show_spinner=False)
def get_stats(dte_min: int, dte_max: int) -> Optional[Dict[str, Any]]:
    """
    Get summary statistics

    Args:
        dte_min: Minimum DTE
        dte_max: Maximum DTE

    Returns:
        Dictionary of statistics
    """
    try:
        with get_db_connection() as conn:
            query = '''
                SELECT
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(*) as total_opportunities,
                    AVG(premium_pct) as avg_premium_pct,
                    MAX(premium_pct) as max_premium_pct
                FROM stock_premiums
                WHERE dte BETWEEN %s AND %s
            '''

            df = pd.read_sql(query, conn, params=(dte_min, dte_max))

            if not df.empty:
                return {
                    'unique_symbols': int(df['unique_symbols'].iloc[0] or 0),
                    'total_opportunities': int(df['total_opportunities'].iloc[0] or 0),
                    'avg_premium_pct': float(df['avg_premium_pct'].iloc[0] or 0),
                    'max_premium_pct': float(df['max_premium_pct'].iloc[0] or 0)
                }
            return None

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return None


def get_available_sectors() -> List[str]:
    """Get list of available sectors from database"""
    try:
        with get_db_connection() as conn:
            query = "SELECT DISTINCT sector FROM stocks WHERE sector IS NOT NULL ORDER BY sector"
            df = pd.read_sql(query, conn)
            return df['sector'].tolist()
    except Exception as e:
        logger.error(f"Error getting sectors: {e}")
        return []


# ============================================================================
# SYNC FUNCTIONS
# ============================================================================

def sync_premiums_for_dte(target_dte: int, dte_label: str) -> Tuple[int, int, int]:
    """
    Sync premium data for all watchlist symbols with progress indicator

    Args:
        target_dte: Target days to expiration (7 for weekly, 30 for monthly)
        dte_label: Label for progress display (e.g., "7-Day" or "30-Day")

    Returns:
        Tuple of (success_count, failed_count, total_count)
    """
    try:
        # Get all symbols from watchlists
        with get_db_connection() as conn:
            query = "SELECT DISTINCT symbol FROM tv_symbols_api ORDER BY symbol"
            df = pd.read_sql(query, conn)
            symbols = df['symbol'].tolist()

        if not symbols:
            st.warning("No symbols found in watchlists. Please sync TradingView watchlists first.")
            return 0, 0, 0

        # Create sync instance
        syncer = StockDataSync()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_estimate = st.empty()

        success_count = 0
        failed_count = 0
        total_count = len(symbols)
        start_time = time.time()

        # Sync each symbol
        for idx, symbol in enumerate(symbols, 1):
            # Calculate time estimate
            if idx > 1:
                elapsed = time.time() - start_time
                estimated_total = (elapsed / (idx - 1)) * total_count
                remaining = estimated_total - elapsed
                time_estimate.caption(f"⏱️ Est. {remaining:.0f}s remaining")

            # Update progress
            progress_pct = idx / total_count
            progress_bar.progress(progress_pct)
            status_text.text(f"Syncing {dte_label}: {symbol} ({idx}/{total_count})")

            # Sync stock data first
            if syncer.sync_stock_data(symbol):
                # Then sync premiums for target DTE
                if syncer.sync_premiums(symbol, target_dte=target_dte):
                    success_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1

        # Clean up
        syncer.close()
        elapsed_time = time.time() - start_time

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        time_estimate.empty()

        # Show completion message
        st.success(f"✅ Completed in {elapsed_time:.1f}s")

        return success_count, failed_count, total_count

    except Exception as e:
        logger.error(f"Error during sync: {e}")
        st.error(f"Sync error: {str(e)}")
        return 0, 0, 0


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def apply_advanced_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply advanced filters to DataFrame

    Args:
        df: Input DataFrame
        filters: Dictionary of filter parameters

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Annual return filter
    if filters.get('min_annual_return', 0) > 0:
        filtered_df = filtered_df[filtered_df['annualized_52wk'] >= filters['min_annual_return']]

    # Volume filter
    if filters.get('min_volume', 0) > 0:
        filtered_df = filtered_df[filtered_df['volume'] >= filters['min_volume']]

    # Sector filter
    if filters.get('selected_sectors') and len(filters['selected_sectors']) > 0:
        filtered_df = filtered_df[filtered_df['sector'].isin(filters['selected_sectors'])]

    # IV range filter
    iv_range = filters.get('iv_range', (0, 200))
    if iv_range != (0, 200):
        filtered_df = filtered_df[
            (filtered_df['implied_volatility'] >= iv_range[0]) &
            (filtered_df['implied_volatility'] <= iv_range[1])
        ]

    # Open interest filter
    if filters.get('min_open_interest', 0) > 0:
        filtered_df = filtered_df[filtered_df['open_interest'] >= filters['min_open_interest']]

    # Bid-ask spread filter
    if filters.get('max_bid_ask_spread', 5.0) < 5.0:
        filtered_df = filtered_df[filtered_df['bid_ask_spread'] <= filters['max_bid_ask_spread']]

    return filtered_df


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_to_csv(df: pd.DataFrame, filename: str):
    """Export DataFrame to CSV with download button"""
    if df.empty:
        st.warning("No data to export")
        return

    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"download_csv_{filename}"
    )


def export_to_excel(df: pd.DataFrame, filename: str):
    """Export DataFrame to Excel with formatting"""
    if df.empty:
        st.warning("No data to export")
        return

    try:
        buffer = BytesIO()

        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Opportunities', index=False)

            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Opportunities']

            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white',
                'border': 1
            })

            money_format = workbook.add_format({'num_format': '$#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})

            # Apply header formatting
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(0, col_num, col_name, header_format)

                # Auto-adjust column width
                max_len = max(
                    df[col_name].astype(str).apply(len).max(),
                    len(col_name)
                ) + 2
                worksheet.set_column(col_num, col_num, min(max_len, 30))

                # Apply number formatting
                if any(word in col_name.lower() for word in ['premium', 'price', 'strike']):
                    worksheet.set_column(col_num, col_num, 12, money_format)
                elif any(word in col_name.lower() for word in ['pct', 'return', 'iv', 'volatility']):
                    # Note: percentages are already in the right format from database
                    pass

        st.download_button(
            label="📊 Download Excel",
            data=buffer.getvalue(),
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_excel_{filename}"
        )

    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def render_premium_heatmap(df: pd.DataFrame):
    """Heatmap of average premiums by sector and DTE"""
    if df.empty or 'sector' not in df.columns:
        st.info("No data available for heatmap")
        return

    # Filter out rows with null sector
    df_filtered = df[df['sector'].notna()].copy()

    if df_filtered.empty:
        st.info("No sector data available")
        return

    # Create pivot table
    pivot = df_filtered.pivot_table(
        values='premium_pct',
        index='sector',
        columns='dte',
        aggfunc='mean'
    )

    if pivot.empty:
        st.info("Insufficient data for heatmap")
        return

    fig = px.imshow(
        pivot,
        labels=dict(x="DTE", y="Sector", color="Avg Premium %"),
        title="Premium Opportunities Heatmap by Sector and DTE",
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_scatter_analysis(df: pd.DataFrame):
    """Scatter plot: Risk (Delta) vs Reward (Annual Return)"""
    if df.empty:
        st.info("No data available for scatter plot")
        return

    fig = px.scatter(
        df,
        x='delta',
        y='annualized_52wk',
        size='premium',
        color='sector',
        hover_data=['symbol', 'premium', 'strike_price', 'dte'],
        title="Risk (Delta) vs Reward (Annual Return)",
        labels={'delta': 'Delta (Risk)', 'annualized_52wk': 'Annual Return (%)'}
    )

    # Add median lines
    if not df['annualized_52wk'].isna().all():
        fig.add_hline(y=df['annualized_52wk'].median(), line_dash="dash",
                     line_color="gray", annotation_text="Median Return")
    if not df['delta'].isna().all():
        fig.add_vline(x=df['delta'].median(), line_dash="dash",
                     line_color="gray", annotation_text="Median Delta")

    st.plotly_chart(fig, use_container_width=True)


def render_distribution_charts(df: pd.DataFrame):
    """Distribution of key metrics"""
    if df.empty:
        st.info("No data available for distributions")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x='premium_pct',
            nbins=30,
            title="Premium % Distribution",
            labels={'premium_pct': 'Premium %'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'sector' in df.columns and df['sector'].notna().any():
            fig = px.box(
                df[df['sector'].notna()],
                x='sector',
                y='annualized_52wk',
                title="Annual Return by Sector",
                labels={'annualized_52wk': 'Annual Return (%)'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sector data available for box plot")


# ============================================================================
# REUSABLE SCANNER COMPONENT
# ============================================================================

def render_scanner_section(
    scanner_type: str,
    dte_min: int,
    dte_max: int,
    title: str,
    icon: str,
    description: str,
    filters: Dict[str, Any],
    expanded: bool = True
):
    """
    Reusable scanner section component

    Args:
        scanner_type: Identifier ('7day' or '30day')
        dte_min: Minimum DTE
        dte_max: Maximum DTE
        title: Section title
        icon: Section icon
        description: Section description
        filters: Filter parameters
        expanded: Whether expander is expanded by default
    """
    with st.expander(f"{icon} **{title}**", expanded=expanded):
        st.caption(description)

        # Sync controls
        col_sync1, col_sync2, col_sync3 = st.columns([1, 1, 3])

        with col_sync1:
            target_dte = 7 if scanner_type == '7day' else 30
            if st.button(f"🔄 Sync {title.split()[0]}", key=f"sync_{scanner_type}", type="primary"):
                with st.spinner(f"Syncing {title.split()[0]} options data..."):
                    success, failed, total = sync_premiums_for_dte(target_dte, title.split()[0])

                    if total > 0:
                        if success > 0:
                            st.success(f"✅ Synced {success}/{total} symbols successfully!")
                        if failed > 0:
                            st.warning(f"⚠️ {failed}/{total} symbols failed to sync")

                        # Clear caches to show fresh data
                        fetch_opportunities.clear()
                        get_last_sync_time.clear()
                        get_stats.clear()
                        st.rerun()

        with col_sync2:
            last_sync = get_last_sync_time(dte_min, dte_max)
            if last_sync:
                st.caption(f"Last sync: {last_sync.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.caption("Never synced")

        with col_sync3:
            stats = get_stats(dte_min, dte_max)
            if stats:
                st.caption(f"📊 {stats['unique_symbols']} symbols • {stats['total_opportunities']} opportunities")

        # Fetch opportunities
        with st.spinner("Loading opportunities..."):
            df, error = fetch_opportunities(
                dte_min, dte_max,
                delta_min=filters['delta_range'][0],
                delta_max=filters['delta_range'][1],
                min_premium=filters['min_premium'],
                min_stock_price=0,
                max_stock_price=filters['max_stock_price']
            )

        if error:
            st.error(f"❌ {error}")
            return

        # Apply advanced filters
        df_filtered = apply_advanced_filters(df, filters)

        if not df_filtered.empty:
            # Summary metrics
            st.markdown(f"### 📈 {title} Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric("Opportunities", len(df_filtered))

            with metric_col2:
                avg_premium = df_filtered['premium_pct'].mean()
                period = "Weekly" if scanner_type == '7day' else "Monthly"
                st.metric(f"Avg {period} Return", f"{avg_premium:.2f}%")

            with metric_col3:
                avg_annual = df_filtered['annualized_52wk'].mean()
                st.metric("Avg Annualized", f"{avg_annual:.1f}%")

            with metric_col4:
                best_premium = df_filtered['premium_pct'].max()
                st.metric(f"Best {period}", f"{best_premium:.2f}%")

            # Export buttons
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                export_to_csv(df_filtered, f"{scanner_type}_premiums")
            with export_col2:
                export_to_excel(df_filtered, f"{scanner_type}_premiums")

            # All opportunities table
            st.markdown(f"### 🏆 {title} Opportunities")

            # Sort by premium descending and display
            display_df = df_filtered.sort_values('premium', ascending=False).copy()

            # Add TradingView chart link
            display_df['chart'] = display_df['symbol'].apply(
                lambda x: f"https://www.tradingview.com/chart/?symbol={x}"
            )

            st.dataframe(
                display_df[['symbol', 'chart', 'company_name', 'stock_price', 'strike_price',
                           'premium', 'dte', 'premium_pct', 'annualized_52wk', 'delta',
                           'implied_volatility', 'volume', 'open_interest', 'sector']],
                use_container_width=True,
                height=500,
                column_config={
                    "symbol": "Symbol",
                    "chart": st.column_config.LinkColumn("📊 Chart", display_text="View"),
                    "company_name": "Company",
                    "stock_price": st.column_config.NumberColumn("Stock $", format="$%.2f"),
                    "strike_price": st.column_config.NumberColumn("Strike", format="$%.2f"),
                    "premium": st.column_config.NumberColumn("Premium", format="$%.2f"),
                    "dte": "DTE",
                    "premium_pct": st.column_config.NumberColumn(
                        f"{'Weekly' if scanner_type == '7day' else 'Monthly'}%",
                        format="%.2f%%"
                    ),
                    "annualized_52wk": st.column_config.NumberColumn("Annual%", format="%.1f%%"),
                    "delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                    "implied_volatility": st.column_config.NumberColumn("IV", format="%.1f%%"),
                    "volume": "Volume",
                    "open_interest": "Open Int",
                    "sector": "Sector"
                }
            )

        else:
            st.info(f"📭 No {title.lower()} opportunities found. Try adjusting filters or running sync.")


# ============================================================================
# MAIN PAGE
# ============================================================================

def main():
    """Main application entry point"""

    # Initialize session state
    initialize_session_state()

    # Page header
    st.title("💎 Premium Scanner")
    st.markdown("""
    **Advanced 7-Day and 30-Day Options Premium Analysis**

    Automatically scans all database stocks for cash-secured put opportunities with advanced filtering and analytics.
    """)

    # Sidebar filters
    with st.sidebar:
        st.header("🎯 Filters")

        with st.form("filter_form"):
            st.subheader("Basic Filters")

            max_stock_price = st.number_input(
                "Max Stock Price ($)",
                min_value=0,
                value=st.session_state.scanner_filters['max_stock_price'],
                step=10,
                help="Maximum stock price to include in scan"
            )

            delta_range = st.slider(
                "Delta Range",
                min_value=-1.0,
                max_value=0.0,
                value=st.session_state.scanner_filters['delta_range'],
                step=0.05,
                help="Delta represents probability of option expiring ITM. -0.30 ≈ 70% PoP"
            )

            min_premium = st.number_input(
                "Min Premium ($)",
                min_value=0.0,
                value=st.session_state.scanner_filters['min_premium'],
                step=10.0,
                help="Minimum premium per contract"
            )

            min_annual_return = st.number_input(
                "Min Annualized (%)",
                min_value=0.0,
                value=st.session_state.scanner_filters['min_annual_return'],
                step=5.0,
                help="Minimum annualized return percentage"
            )

            min_volume = st.number_input(
                "Min Volume",
                min_value=0,
                value=st.session_state.scanner_filters['min_volume'],
                step=100,
                help="Minimum option volume for liquidity"
            )

            st.subheader("Advanced Filters")

            # Sector filter
            available_sectors = get_available_sectors()
            selected_sectors = st.multiselect(
                "Sectors",
                options=available_sectors,
                default=st.session_state.scanner_filters['selected_sectors'],
                help="Filter by specific sectors"
            )

            # IV range
            iv_range = st.slider(
                "IV Range (%)",
                min_value=0,
                max_value=200,
                value=st.session_state.scanner_filters['iv_range'],
                help="Implied volatility range filter"
            )

            # Open interest
            min_open_interest = st.number_input(
                "Min Open Interest",
                min_value=0,
                value=st.session_state.scanner_filters['min_open_interest'],
                step=100,
                help="Minimum open interest for liquidity"
            )

            # Bid-ask spread
            max_bid_ask_spread = st.number_input(
                "Max Bid-Ask Spread ($)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.scanner_filters['max_bid_ask_spread'],
                step=0.1,
                help="Maximum bid-ask spread (lower is better)"
            )

            # Submit button
            apply_filters = st.form_submit_button("🔍 Apply Filters", type="primary", use_container_width=True)

            if apply_filters:
                st.session_state.scanner_filters = {
                    'max_stock_price': max_stock_price,
                    'delta_range': delta_range,
                    'min_premium': min_premium,
                    'min_annual_return': min_annual_return,
                    'min_volume': min_volume,
                    'selected_sectors': selected_sectors,
                    'iv_range': iv_range,
                    'min_open_interest': min_open_interest,
                    'max_bid_ask_spread': max_bid_ask_spread
                }
                # Clear caches
                fetch_opportunities.clear()
                st.rerun()

        # Help section
        st.divider()
        with st.expander("❓ Quick Help"):
            st.markdown("""
            **Understanding Metrics:**
            - **Delta**: ~-0.30 = 70% profit probability
            - **IV**: Higher = more expensive options
            - **DTE**: Days to expiration
            - **Annual%**: Return if repeated 52 weeks

            **Recommended Settings:**
            - Conservative: Delta -0.30 to -0.20
            - Aggressive: Delta -0.40 to -0.30
            """)

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["⚡ 7-Day Scanner", "📅 30-Day Scanner", "📊 Analytics"])

    filters = st.session_state.scanner_filters
    next_friday = get_next_friday()
    days_to_friday = (next_friday - datetime.now()).days

    with tab1:
        render_scanner_section(
            scanner_type="7day",
            dte_min=ScannerConfig.WEEKLY_DTE_MIN,
            dte_max=ScannerConfig.WEEKLY_DTE_MAX,
            title="7-Day Scanner (Weekly)",
            icon="⚡",
            description=f"Options expiring on {next_friday.strftime('%B %d, %Y')} ({days_to_friday} days away)",
            filters=filters,
            expanded=True
        )

    with tab2:
        render_scanner_section(
            scanner_type="30day",
            dte_min=ScannerConfig.MONTHLY_DTE_MIN,
            dte_max=ScannerConfig.MONTHLY_DTE_MAX,
            title="30-Day Scanner (Monthly)",
            icon="📅",
            description="Standard monthly wheel strategy with 30-day options",
            filters=filters,
            expanded=True
        )

    with tab3:
        st.subheader("📊 Analytics Dashboard")

        # Fetch both datasets for analytics
        df_7day, _ = fetch_opportunities(
            ScannerConfig.WEEKLY_DTE_MIN, ScannerConfig.WEEKLY_DTE_MAX,
            delta_min=filters['delta_range'][0], delta_max=filters['delta_range'][1],
            min_premium=filters['min_premium'], max_stock_price=filters['max_stock_price']
        )

        df_30day, _ = fetch_opportunities(
            ScannerConfig.MONTHLY_DTE_MIN, ScannerConfig.MONTHLY_DTE_MAX,
            delta_min=filters['delta_range'][0], delta_max=filters['delta_range'][1],
            min_premium=filters['min_premium'], max_stock_price=filters['max_stock_price']
        )

        # Combine datasets
        df_combined = pd.concat([
            apply_advanced_filters(df_7day, filters),
            apply_advanced_filters(df_30day, filters)
        ]).drop_duplicates(subset=['symbol'], keep='first')

        if not df_combined.empty:
            # Visualizations
            st.markdown("### Premium Heatmap")
            render_premium_heatmap(df_combined)

            st.markdown("### Risk vs Reward Analysis")
            render_scatter_analysis(df_combined)

            st.markdown("### Distribution Analysis")
            render_distribution_charts(df_combined)
        else:
            st.info("No data available for analytics. Try adjusting filters or syncing data.")

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("💎 Premium Scanner • Connection pooling enabled • Advanced analytics")


# Execute main function when module is loaded
main()
