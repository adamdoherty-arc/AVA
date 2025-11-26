"""
Earnings Calendar Page - User-Friendly UI with buttons
"""
import streamlit as st
import pandas as pd
import os
import time
from src.services import get_tradingview_manager  # Use centralized service registry
import robin_stocks.robinhood as rh
from src.components.pagination_component import paginate_dataframe


def display_quality_opportunities():
    """Display high-quality earnings opportunities"""
    st.markdown("### ⭐ High-Quality Earnings Opportunities")
    st.caption("Top-rated stocks reporting earnings in next 7 days")

    tv = get_tv_db_manager()
    conn = tv.get_connection()
    cur = conn.cursor()

    # Get upcoming earnings with quality patterns
    cur.execute("""
        SELECT
            e.symbol,
            e.earnings_date,
            e.earnings_time,
            e.expected_move_pct,
            e.pre_earnings_iv,
            p.beat_rate_8q,
            p.avg_surprise_pct_8q,
            p.quality_score,
            s.company_name,
            s.sector
        FROM earnings_events e
        LEFT JOIN earnings_pattern_analysis p ON e.symbol = p.symbol
        LEFT JOIN stocks s ON e.symbol = s.symbol
        WHERE e.earnings_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
          AND e.has_occurred = FALSE
          AND p.quality_score IS NOT NULL
        ORDER BY p.quality_score DESC, e.earnings_date
        LIMIT 10
    """)

    opportunities = cur.fetchall()
    cur.close()
    conn.close()

    if opportunities:
        df_opp = pd.DataFrame(opportunities, columns=[
            'Symbol', 'Date', 'Time', 'Expected Move %', 'IV Rank',
            'Beat Rate %', 'Avg Surprise %', 'Quality Score', 'Company', 'Sector'
        ])

        # Format columns
        df_opp['Date'] = pd.to_datetime(df_opp['Date']).dt.strftime('%Y-%m-%d')
        df_opp['Expected Move %'] = df_opp['Expected Move %'].apply(lambda x: f"±{x:.1f}%" if pd.notna(x) else "N/A")
        df_opp['IV Rank'] = df_opp['IV Rank'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        df_opp['Beat Rate %'] = df_opp['Beat Rate %'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        df_opp['Avg Surprise %'] = df_opp['Avg Surprise %'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")

        st.dataframe(
            df_opp,
            hide_index=True,
            column_config={
                "Quality Score": st.column_config.ProgressColumn(
                    "Quality",
                    min_value=0,
                    max_value=100,
                    format="%d"
                ),
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Company": st.column_config.TextColumn("Company", width="large"),
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Expected Move %": st.column_config.TextColumn("Exp Move", width="small"),
                "Beat Rate %": st.column_config.TextColumn("Beat Rate", width="small"),
                "Avg Surprise %": st.column_config.TextColumn("Avg Surprise", width="small")
            },
            use_container_width=True
        )

        # Add explanation
        with st.expander("ℹ️ What is Quality Score?"):
            st.markdown("""
            **Quality Score (0-100)** measures earnings predictability and opportunity:

            - **70-100**: Excellent - Consistent beaters with strong patterns
            - **50-69**: Good - Reliable earnings with moderate consistency
            - **Below 50**: Caution - Unpredictable or inconsistent results

            **Components:**
            - Beat Rate (40%): Percentage of earnings beats
            - Avg Surprise (30%): Average EPS surprise magnitude
            - Consistency (30%): Standard deviation of surprises (lower is better)

            **Expected Move** is calculated from options pricing (ATM straddle × 0.85)
            """)
    else:
        st.info("📭 No high-quality opportunities in next 7 days")
        st.caption("Opportunities appear here when stocks with quality scores >70 have upcoming earnings")


# ========================================================================
# PERFORMANCE OPTIMIZATION: Cached Database Queries
# ========================================================================

@st.cache_resource
def get_tv_db_manager():
    """
    Get TradingViewDBManager from centralized service registry.

    The registry ensures singleton behavior across the application.
    """
    return get_tradingview_manager()


@st.cache_data(ttl=300)  # 5-minute cache
def check_earnings_table_exists():
    """Cached check for earnings table existence"""
    tv = get_tv_db_manager()
    conn = tv.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'earnings_events'
    """)
    table_exists = cur.fetchone()[0] > 0
    cur.close()
    conn.close()

    return table_exists


@st.cache_data(ttl=300)  # 5-minute cache for earnings count
def get_earnings_count():
    """Cached query for total earnings count"""
    tv = get_tv_db_manager()
    conn = tv.get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM earnings_events")
    count = cur.fetchone()[0]

    cur.close()
    conn.close()

    return count


@st.cache_data(ttl=60)  # 1-minute cache for earnings data
def get_earnings_data_cached(date_filter, time_filter, result_filter):
    """
    Cached query for earnings data with filters
    Returns DataFrame ready for display
    """
    tv = get_tv_db_manager()
    conn = tv.get_connection()
    cur = conn.cursor()

    query = """
        SELECT
            e.symbol,
            e.earnings_date,
            e.earnings_time,
            e.eps_estimate,
            e.eps_actual,
            e.revenue_estimate,
            e.revenue_actual,
            CASE
                WHEN e.eps_actual > e.eps_estimate THEN 'Beat'
                WHEN e.eps_actual < e.eps_estimate THEN 'Miss'
                WHEN e.eps_actual = e.eps_estimate THEN 'Meet'
                ELSE 'Pending'
            END as result,
            s.company_name,
            s.sector
        FROM earnings_events e
        LEFT JOIN stocks s ON e.symbol = s.symbol
        WHERE 1=1
    """

    params = []

    # Date filter
    if date_filter == "This Week":
        query += " AND e.earnings_date >= NOW() AND e.earnings_date <= NOW() + INTERVAL '7 days'"
    elif date_filter == "Next Week":
        query += " AND e.earnings_date >= NOW() + INTERVAL '7 days' AND e.earnings_date <= NOW() + INTERVAL '14 days'"
    elif date_filter == "This Month":
        query += " AND e.earnings_date >= NOW() AND e.earnings_date <= NOW() + INTERVAL '30 days'"
    elif date_filter == "Next Month":
        query += " AND e.earnings_date >= NOW() + INTERVAL '30 days' AND e.earnings_date <= NOW() + INTERVAL '60 days'"

    # Time filter
    if time_filter != "All":
        query += " AND e.earnings_time = %s"
        params.append(time_filter)

    query += " ORDER BY e.earnings_date DESC LIMIT 200"

    cur.execute(query, params)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    if rows:
        df = pd.DataFrame(rows, columns=[
            'Symbol', 'Date', 'Time', 'EPS Est', 'EPS Act',
            'Rev Est', 'Rev Act', 'Result', 'Company', 'Sector'
        ])

        # Format date
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M')

        # Filter by result if needed
        if result_filter != "All":
            df = df[df['Result'] == result_filter]

        return df
    else:
        return None


def show_earnings_calendar():
    """Display earnings calendar with user-friendly UI"""
    st.title("📅 Earnings Calendar")
    
    # Sync status widget (earnings data doesn't have dedicated sync, show stock_data)
    from src.components.sync_status_widget import SyncStatusWidget
    sync_widget = SyncStatusWidget()
    sync_widget.display(
        table_name="stock_data",
        title="Earnings Data Sync",
        compact=True
    )

    # PERFORMANCE: Use cached manager and queries
    tv = get_tv_db_manager()
    table_exists = check_earnings_table_exists()

    # Setup section - if tables don't exist
    if not table_exists:
        st.warning("📋 Earnings tables not initialized yet")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Click the button to automatically create the earnings database tables")
        with col2:
            if st.button("🔧 Initialize Database", type="primary", width='stretch'):
                with st.spinner("Creating earnings tables..."):
                    try:
                        # Create earnings_events table
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS earnings_events (
                                id SERIAL PRIMARY KEY,
                                symbol VARCHAR(10) NOT NULL,
                                earnings_date TIMESTAMP WITH TIME ZONE,
                                earnings_time VARCHAR(10),
                                eps_estimate DECIMAL(10, 2),
                                eps_actual DECIMAL(10, 2),
                                revenue_estimate DECIMAL(15, 2),
                                revenue_actual DECIMAL(15, 2),
                                surprise_percent DECIMAL(10, 2),
                                pre_earnings_iv DECIMAL(10, 4),
                                post_earnings_iv DECIMAL(10, 4),
                                pre_earnings_price DECIMAL(10, 2),
                                post_earnings_price DECIMAL(10, 2),
                                price_move_percent DECIMAL(10, 2),
                                volume_ratio DECIMAL(10, 2),
                                options_volume INTEGER,
                                whisper_number DECIMAL(10, 2),
                                created_at TIMESTAMP DEFAULT NOW(),
                                updated_at TIMESTAMP DEFAULT NOW(),
                                UNIQUE(symbol, earnings_date)
                            )
                        """)

                        # Create index
                        cur.execute("""
                            CREATE INDEX IF NOT EXISTS idx_earnings_date
                            ON earnings_events(earnings_date DESC)
                        """)

                        conn.commit()
                        cur.close()
                        conn.close()
                        st.success("✅ Earnings tables created successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating tables: {e}")
                        conn.rollback()
                        cur.close()
                        conn.close()
                return

    # Main interface - tables exist
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])

    with col_s1:
        # PERFORMANCE: Use cached count
        count = get_earnings_count()
        st.metric("📊 Earnings Events", f"{count:,}")

    with col_s2:
        if st.button("🔄 Sync Earnings", type="primary", width='stretch'):
            # Get fresh connection for sync operation
            conn = tv.get_connection()
            cur = conn.cursor()
            sync_earnings_from_robinhood(conn, cur)
            cur.close()
            conn.close()
            # Clear cache after sync
            get_earnings_count.clear()
            get_earnings_data_cached.clear()

    with col_s3:
        st.caption("💡 Auto-syncs daily at 6 AM")


    # Display earnings data (PERFORMANCE: Use cached count)
    count = get_earnings_count()

    if count > 0:
        display_earnings_table()
    else:
        st.warning("📭 No earnings data yet")
        st.info("👆 Click 'Sync Earnings' to fetch data from Robinhood (takes 2-3 minutes for 100 stocks)")


def sync_earnings_from_robinhood(conn, cur):
    """Sync earnings from Robinhood API"""
    with st.spinner("🔄 Syncing earnings from Robinhood..."):
        try:
            # Login to Robinhood
            username = os.getenv('ROBINHOOD_USERNAME')
            password = os.getenv('ROBINHOOD_PASSWORD')

            if not username or not password:
                st.error("Robinhood credentials not found in .env file")
                return

            rh.login(username, password)

            # Get stocks to sync
            cur.execute("SELECT symbol FROM stocks WHERE is_active = TRUE LIMIT 100")
            symbols = [row[0] for row in cur.fetchall()]

            synced = 0
            errors = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, symbol in enumerate(symbols):
                try:
                    status_text.text(f"Syncing {symbol}... ({i+1}/{len(symbols)})")

                    # Get earnings from Robinhood
                    earnings = rh.get_earnings(symbol)

                    if earnings and len(earnings) > 0:
                        # Process latest earnings
                        latest = earnings[0]
                        report_data = latest.get('report', {})
                        eps_data = latest.get('eps', {})

                        report_date = report_data.get('date')
                        eps_actual = eps_data.get('actual')
                        eps_estimate = eps_data.get('estimate')

                        if report_date:
                            # Insert or update
                            cur.execute("""
                                INSERT INTO earnings_events (
                                    symbol, earnings_date, eps_actual, eps_estimate, updated_at
                                )
                                VALUES (%s, %s, %s, %s, NOW())
                                ON CONFLICT (symbol, earnings_date)
                                DO UPDATE SET
                                    eps_actual = EXCLUDED.eps_actual,
                                    eps_estimate = EXCLUDED.eps_estimate,
                                    updated_at = NOW()
                            """, (symbol, report_date, eps_actual, eps_estimate))

                            synced += 1

                    progress_bar.progress((i + 1) / len(symbols))

                except Exception as e:
                    errors += 1
                    continue

            conn.commit()
            rh.logout()

            status_text.empty()
            progress_bar.empty()

            st.success(f"✅ Sync Complete! {synced} earnings synced, {errors} errors")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Sync error: {str(e)}")
            conn.rollback()


def display_earnings_table():
    """Display earnings data in a table - PERFORMANCE OPTIMIZED"""

    # Show high-quality opportunities first
    display_quality_opportunities()

    st.markdown("---")
    st.markdown("### 📅 Full Earnings Calendar")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        date_filter = st.selectbox(
            "📅 Date Range",
            ["All Time", "This Week", "Next Week", "This Month", "Next Month"],
            key="earn_date_filter"
        )

    with col_f2:
        time_filter = st.selectbox(
            "⏰ Time",
            ["All", "BMO", "AMC"],
            key="earn_time_filter"
        )

    with col_f3:
        result_filter = st.selectbox(
            "📊 Result",
            ["All", "Beat", "Miss", "Meet", "Pending"],
            key="earn_result_filter"
        )

    # PERFORMANCE: Use cached query with filters as parameters
    df = get_earnings_data_cached(date_filter, time_filter, result_filter)

    if df is not None and not df.empty:

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Events", len(df))
        with col2:
            beats = len(df[df['Result'] == 'Beat'])
            st.metric("✅ Beats", beats, delta=f"{beats/len(df)*100:.0f}%" if len(df) > 0 else "0%")
        with col3:
            misses = len(df[df['Result'] == 'Miss'])
            st.metric("❌ Misses", misses, delta=f"-{misses/len(df)*100:.0f}%" if len(df) > 0 else "0%")
        with col4:
            pending = len(df[df['Result'] == 'Pending'])
            st.metric("⏳ Pending", pending)

        st.markdown("### 📊 Earnings Calendar")

        # PERFORMANCE: Add pagination for large tables
        paginated_df = paginate_dataframe(df, page_size=50, key_prefix="earnings_calendar")

        # Display table
        st.dataframe(
            paginated_df,
            hide_index=True,
            width='stretch',
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Time": st.column_config.TextColumn("Time", width="small"),
                "EPS Est": st.column_config.NumberColumn("EPS Est", format="%.2f"),
                "EPS Act": st.column_config.NumberColumn("EPS Act", format="%.2f"),
                "Rev Est": st.column_config.NumberColumn("Rev Est ($M)", format="%.0f"),
                "Rev Act": st.column_config.NumberColumn("Rev Act ($M)", format="%.0f"),
                "Result": st.column_config.TextColumn("Result", width="small"),
                "Company": st.column_config.TextColumn("Company", width="large"),
                "Sector": st.column_config.TextColumn("Sector", width="medium")
            }
        )

        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="earnings_calendar.csv",
            mime="text/csv"
        )

    else:
        st.info("📭 No earnings found for selected filters")
