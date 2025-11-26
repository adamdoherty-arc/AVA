"""
Sector Analysis Page - Comprehensive sector breakdown with AI insights
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.services import get_tradingview_manager  # Use centralized service registry
from src.components.pagination_component import paginate_dataframe
import logging

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Sector Analysis", layout="wide")


# ========================================================================
# PERFORMANCE OPTIMIZATION: Cached Database Queries
# ========================================================================

@st.cache_resource
def get_sector_tv_manager():
    """
    Get TradingViewDBManager from centralized service registry.

    The registry ensures singleton behavior across the application.
    """
    return get_tradingview_manager()


@st.cache_data(ttl=300)  # 5-minute cache
def check_sector_tables_exist():
    """Cached check for sector analysis tables"""
    tv_manager = get_sector_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'stock_sectors'
        )
    """)
    tables_exist = cur.fetchone()[0]
    cur.close()
    conn.close()

    return tables_exist


@st.cache_data(ttl=300)  # 5-minute cache for sector data
def get_sector_data_cached():
    """Cached query for sector analysis data"""
    tv_manager = get_sector_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            sector,
            stock_count,
            avg_premium_yield,
            avg_monthly_return,
            overall_score,
            trend_direction,
            recommended_etf,
            ai_recommendation,
            best_strategy,
            risk_level,
            top_stocks
        FROM sector_analysis
        WHERE stock_count > 0
        ORDER BY overall_score DESC
    """)

    sector_data = cur.fetchall()
    cur.close()
    conn.close()

    return sector_data


@st.cache_data(ttl=60)  # 1-minute cache for sector stocks
def get_sector_stocks_cached(sector):
    """Cached query for stocks in a specific sector with premium data"""
    tv_manager = get_sector_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT DISTINCT ON (sp.symbol)
            ss.symbol,
            sd.current_price as stock_price,
            ss.market_cap,
            sp.strike_price,
            sp.premium,
            sp.delta,
            sp.monthly_return,
            sp.annualized_return,
            sp.implied_volatility as iv
        FROM stock_sectors ss
        LEFT JOIN stock_data sd ON ss.symbol = sd.symbol
        LEFT JOIN stock_premiums sp ON ss.symbol = sp.symbol
            AND sp.dte BETWEEN 28 AND 32
            AND sp.option_type = 'put'
        WHERE ss.sector = %s
        ORDER BY sp.symbol, sp.monthly_return DESC
        LIMIT 50
    """, (sector,))

    stocks = cur.fetchall()
    cur.close()
    conn.close()

    return stocks


@st.cache_data(ttl=600)  # 10-minute cache for ETF data (static reference)
def get_sector_etfs_cached():
    """Cached query for sector ETF data"""
    tv_manager = get_sector_tv_manager()
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            se.etf_symbol,
            se.etf_name,
            se.sector,
            se.expense_ratio,
            se.description,
            sa.overall_score,
            sa.avg_monthly_return
        FROM sector_etfs se
        LEFT JOIN sector_analysis sa ON se.sector = sa.sector
        ORDER BY sa.overall_score DESC
    """)

    etf_data = cur.fetchall()
    cur.close()
    conn.close()

    return etf_data


def display_sector_analysis_page():
    st.title("🏭 Sector Analysis - Wheel Strategy Edition")
    st.caption("AI-powered sector insights for cash-secured puts and covered calls")
    
    # Sync status widget
    from src.components.sync_status_widget import SyncStatusWidget
    sync_widget = SyncStatusWidget()
    sync_widget.display(
        table_name="stock_data",
        title="Sector Data Sync",
        compact=True
    )

    # PERFORMANCE: Use cached manager and queries
    tv_manager = get_sector_tv_manager()
    tables_exist = check_sector_tables_exist()

    if not tables_exist:
        st.warning("⚠️ Sector Analysis tables not yet created. Run migration first.")

        if st.button("🔧 Run Migration Now"):
            with st.spinner("Creating sector analysis tables..."):
                try:
                    # Read and execute migration
                    import os
                    migration_path = os.path.join(os.path.dirname(__file__), 'migrations', 'add_sector_analysis_tables.sql')

                    with open(migration_path, 'r') as f:
                        migration_sql = f.read()

                    cur.execute(migration_sql)
                    conn.commit()

                    st.success("✅ Migration complete! Refresh page to continue.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Migration failed: {e}")

        return

    # Get sector data (PERFORMANCE: Use cached query)
    sector_data = get_sector_data_cached()

    if not sector_data:
        st.info("📊 No sector data yet. Let's classify your stocks!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔍 Classify Stocks by Sector"):
                with st.spinner("Classifying stocks... this may take a few minutes"):
                    from src.sector_classifier import SectorClassifier
                    classifier = SectorClassifier()
                    count = classifier.classify_batch(100)
                    st.success(f"✅ Classified {count} stocks! Refresh to see sectors.")

        with col2:
            st.caption("This will fetch sector info from Yahoo Finance for your stocks")

        return

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sector Overview",
        "🎯 Top Stocks by Sector",
        "📈 Sector ETFs",
        "💡 Trading Strategies"
    ])

    # Tab 1: Sector Overview
    with tab1:
        st.subheader("Sector Performance Dashboard")

        # Convert to DataFrame
        df_sectors = pd.DataFrame(sector_data, columns=[
            'Sector', 'Stock Count', 'Avg Premium Yield', 'Avg Monthly Return',
            'Overall Score', 'Trend', 'Recommended ETF', 'AI Recommendation',
            'Best Strategy', 'Risk Level', 'Top Stocks'
        ])

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sectors", len(df_sectors))
        with col2:
            avg_score = df_sectors['Overall Score'].mean()
            st.metric("Avg Sector Score", f"{avg_score:.1f}/100")
        with col3:
            top_sector = df_sectors.iloc[0]['Sector'] if len(df_sectors) > 0 else "N/A"
            st.metric("Top Sector", top_sector)
        with col4:
            csp_count = (df_sectors['Best Strategy'] == 'CSP').sum()
            st.metric("CSP-Recommended Sectors", csp_count)

        # Sector heatmap
        st.markdown("### 🌡️ Sector Heatmap")

        fig = px.treemap(
            df_sectors,
            path=['Sector'],
            values='Stock Count',
            color='Overall Score',
            color_continuous_scale='RdYlGn',
            hover_data=['Avg Monthly Return', 'Risk Level']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sector comparison table
        st.markdown("### 📋 Sector Comparison Table")

        display_df = df_sectors[[
            'Sector', 'Stock Count', 'Overall Score', 'Avg Premium Yield',
            'Avg Monthly Return', 'Trend', 'Best Strategy', 'Risk Level'
        ]].copy()

        # PERFORMANCE: Add pagination for sector comparison
        paginated_sectors = paginate_dataframe(display_df, page_size=25, key_prefix="sector_comparison")

        st.dataframe(
            paginated_sectors,
            hide_index=True,
            width='stretch',
            column_config={
                "Sector": st.column_config.TextColumn("Sector"),
                "Stock Count": st.column_config.NumberColumn("Stocks"),
                "Overall Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "Avg Premium Yield": st.column_config.NumberColumn("Premium Yield", format="%.2f%%"),
                "Avg Monthly Return": st.column_config.NumberColumn("Monthly Return", format="%.2f%%"),
                "Trend": st.column_config.TextColumn("Trend"),
                "Best Strategy": st.column_config.TextColumn("Strategy"),
                "Risk Level": st.column_config.TextColumn("Risk")
            }
        )

        # AI Recommendations
        st.markdown("### 🤖 AI Sector Recommendations")

        for _, row in df_sectors.head(5).iterrows():
            with st.expander(f"💼 {row['Sector']} - Score: {row['Overall Score']:.1f}/100"):
                st.markdown(f"**Recommended Strategy:** {row['Best Strategy']}")
                st.markdown(f"**Risk Level:** {row['Risk Level']}")
                st.markdown(f"**Sector ETF:** {row['Recommended ETF']}")

                if row['AI Recommendation']:
                    st.info(row['AI Recommendation'])

                if row['Top Stocks']:
                    st.markdown("**Top Stocks:**")
                    top_stocks = row['Top Stocks'][:10] if isinstance(row['Top Stocks'], list) else []
                    if top_stocks:
                        st.write(", ".join(top_stocks))

    # Tab 2: Top Stocks by Sector
    with tab2:
        st.subheader("🎯 Best Stocks for Wheel Strategy (by Sector)")

        selected_sector = st.selectbox(
            "Select Sector:",
            df_sectors['Sector'].tolist()
        )

        if selected_sector:
            # Get stocks for sector with premium data (PERFORMANCE: Use cached query)
            stocks = get_sector_stocks_cached(selected_sector)

            if stocks:
                df_stocks = pd.DataFrame(stocks, columns=[
                    'Symbol', 'Stock Price', 'Market Cap', 'Strike', 'Premium',
                    'Delta', 'Monthly %', 'Annual %', 'IV'
                ])

                # Convert to numeric
                for col in ['Stock Price', 'Market Cap', 'Strike', 'Premium', 'Delta', 'Monthly %', 'Annual %', 'IV']:
                    df_stocks[col] = pd.to_numeric(df_stocks[col], errors='coerce')

                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stocks in Sector", len(df_stocks))
                with col2:
                    avg_monthly = df_stocks['Monthly %'].mean()
                    st.metric("Avg Monthly Return", f"{avg_monthly:.2f}%")
                with col3:
                    avg_iv = df_stocks['IV'].mean() * 100
                    st.metric("Avg IV", f"{avg_iv:.1f}%")

                # PERFORMANCE: Add pagination for sector stocks
                paginated_stocks = paginate_dataframe(df_stocks, page_size=25, key_prefix="sector_stocks")

                # Display table
                st.dataframe(
                    paginated_stocks,
                    hide_index=True,
                    width='stretch',
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol"),
                        "Stock Price": st.column_config.NumberColumn("Stock $", format="$%.2f"),
                        "Market Cap": st.column_config.NumberColumn("Market Cap", format="$%.0f"),
                        "Strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                        "Premium": st.column_config.NumberColumn("Premium", format="$%.2f"),
                        "Delta": st.column_config.NumberColumn("Delta", format="%.3f"),
                        "Monthly %": st.column_config.NumberColumn("Monthly %", format="%.2f%%"),
                        "Annual %": st.column_config.NumberColumn("Annual %", format="%.1f%%"),
                        "IV": st.column_config.NumberColumn("IV", format="%.1f%%")
                    }
                )
            else:
                st.info(f"No options data available for {selected_sector} stocks. Sync options data first.")

    # Tab 3: Sector ETFs
    with tab3:
        st.subheader("📈 Sector ETF Explorer")

        # PERFORMANCE: Use cached query
        etf_data = get_sector_etfs_cached()

        if etf_data:
            df_etfs = pd.DataFrame(etf_data, columns=[
                'ETF Symbol', 'ETF Name', 'Sector', 'Expense Ratio',
                'Description', 'Sector Score', 'Sector Monthly Return'
            ])

            # PERFORMANCE: Add pagination for ETF table
            paginated_etfs = paginate_dataframe(df_etfs, page_size=25, key_prefix="sector_etfs")

            st.dataframe(
                paginated_etfs,
                hide_index=True,
                width='stretch',
                column_config={
                    "ETF Symbol": st.column_config.TextColumn("ETF"),
                    "ETF Name": st.column_config.TextColumn("Name"),
                    "Sector": st.column_config.TextColumn("Sector"),
                    "Expense Ratio": st.column_config.NumberColumn("Expense Ratio", format="%.4f"),
                    "Description": st.column_config.TextColumn("Description"),
                    "Sector Score": st.column_config.NumberColumn("Sector Score", format="%.1f"),
                    "Sector Monthly Return": st.column_config.NumberColumn("Sector Return", format="%.2f%%")
                }
            )

            st.caption("💡 Tip: ETFs are great for low-volatility, diversified exposure to sectors")
        else:
            st.info("No ETF data loaded. Run migration to seed ETF database.")

    # Tab 4: Trading Strategies
    with tab4:
        st.subheader("💡 Sector Trading Strategies")

        st.markdown("""
        ### 🎯 Wheel Strategy by Sector

        **Best Sectors for CSP (Cash-Secured Puts):**
        - Technology: High premiums, moderate volatility
        - Healthcare: Steady growth, good premiums
        - Financials: Cyclical opportunities, elevated IV during volatility

        **ETF-Recommended Sectors:**
        - Utilities: Low volatility, better via ETF
        - Consumer Staples: Stable but low premiums
        - Real Estate: REITs better via ETF exposure

        **Risk Management:**
        - Diversify across 3-5 sectors
        - Avoid overweighting any single sector (>30%)
        - Monitor sector rotation signals
        - Use ETFs for defensive sectors

        **Seasonal Patterns:**
        - Technology: Strong Q4 (holiday sales)
        - Energy: Volatile, correlates with oil prices
        - Healthcare: Steady year-round
        - Financials: Strong in rising rate environments
        """)

        # Sector rotation indicator
        st.markdown("### 🔄 Sector Rotation Indicator")

        top_3 = df_sectors.head(3)
        bottom_3 = df_sectors.tail(3)

        col1, col2 = st.columns(2)

        with col1:
            st.success("**Overweight (Top 3 Sectors)**")
            for _, row in top_3.iterrows():
                st.write(f"✅ {row['Sector']} (Score: {row['Overall Score']:.1f})")

        with col2:
            st.error("**Underweight/Avoid (Bottom 3 Sectors)**")
            for _, row in bottom_3.iterrows():
                st.write(f"❌ {row['Sector']} (Score: {row['Overall Score']:.1f})")


if __name__ == "__main__":
    display_sector_analysis_page()
