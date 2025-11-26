"""
Premium Hunter - Simple, Clean Cash-Secured Put Opportunity Finder
Shows ALL options from ALL stocks in ONE sortable table
"""

import streamlit as st
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Premium Hunter", page_icon="💰", layout="wide")

# Database connection
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

# Header
st.title("💰 Premium Hunter - Cash-Secured Put Opportunities")
st.caption("🎯 Target: 30 Delta (~70% win rate) | 30-45 DTE Sweet Spot | Sorted by Monthly Return")

# Filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    min_monthly = st.number_input("Min Monthly %", value=0.0, step=0.5)

with col2:
    max_delta = st.number_input("Max Delta (abs)", value=0.50, step=0.05, help="Lower delta = less risk")

with col3:
    min_dte = st.number_input("Min DTE", value=0, step=5)

with col4:
    max_dte = st.number_input("Max DTE", value=60, step=5)

# Load data
@st.cache_data(ttl=60)
def load_opportunities(min_monthly, max_delta, min_dte, max_dte):
    conn = get_connection()
    cur = conn.cursor()

    query = """
        SELECT
            sp.symbol,
            sd.current_price as stock_price,
            sp.strike_price,
            sp.dte,
            sp.expiration_date,
            sp.premium,
            sp.monthly_return,
            sp.annual_return,
            sp.delta,
            sp.implied_volatility,
            sp.bid,
            sp.ask,
            sp.volume,
            sp.open_interest
        FROM stock_premiums sp
        LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
        WHERE sp.strike_type IN ('30_delta', '5%_OTM')
            AND sp.monthly_return >= %s
            AND ABS(sp.delta) <= %s
            AND sp.dte >= %s
            AND sp.dte <= %s
            AND sp.delta IS NOT NULL
        ORDER BY sp.monthly_return DESC
    """

    cur.execute(query, (min_monthly, max_delta, min_dte, max_dte))

    rows = cur.fetchall()
    cur.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        'Symbol', 'Stock Price', 'Strike', 'DTE', 'Expiration',
        'Premium', 'Monthly %', 'Annual %', 'Delta', 'IV',
        'Bid', 'Ask', 'Volume', 'OI'
    ])

    # Calculate additional metrics
    df['Prob Win %'] = 100 + (df['Delta'] * 100)  # Delta -0.30 = 70% win probability
    df['$ to Collateral'] = df['Strike'] * 100  # Cash needed per contract
    df['ROI'] = (df['Premium'] / df['$ to Collateral']) * 100

    return df

# Load and display
with st.spinner("Loading opportunities..."):
    df = load_opportunities(min_monthly, max_delta, min_dte, max_dte)

if df.empty:
    st.warning("No opportunities found with current filters. Try adjusting the filters above.")
else:
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Opportunities", len(df))

    with col2:
        st.metric("Unique Stocks", df['Symbol'].nunique())

    with col3:
        st.metric("Avg Monthly Return", f"{df['Monthly %'].mean():.2f}%")

    with col4:
        near_30_delta = len(df[df['Delta'].abs().between(0.25, 0.35)])
        st.metric("Near 30Δ (25-35)", near_30_delta)

    with col5:
        high_iv = len(df[df['IV'] > 0.40])
        st.metric("High IV (>40%)", high_iv)


    # Format display dataframe
    display_df = df.copy()
    display_df['Stock Price'] = display_df['Stock Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
    display_df['Premium'] = display_df['Premium'].apply(lambda x: f"${x:.0f}")
    display_df['Monthly %'] = display_df['Monthly %'].apply(lambda x: f"{x:.2f}%")
    display_df['Annual %'] = display_df['Annual %'].apply(lambda x: f"{x:.1f}%")
    display_df['Delta'] = display_df['Delta'].apply(lambda x: f"{x:.3f}")
    display_df['IV'] = display_df['IV'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) and x > 0 else "N/A")
    display_df['Prob Win %'] = display_df['Prob Win %'].apply(lambda x: f"{x:.0f}%")
    display_df['$ to Collateral'] = display_df['$ to Collateral'].apply(lambda x: f"${x:,.0f}")
    display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.2f}%")
    display_df['Bid'] = display_df['Bid'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display_df['Ask'] = display_df['Ask'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")

    # Main table - show key columns
    st.markdown("### 🔥 All Premium Opportunities")

    key_cols = ['Symbol', 'Stock Price', 'Strike', 'DTE', 'Premium', 'Monthly %',
                'Annual %', 'Delta', 'Prob Win %', 'IV', 'Expiration']

    st.dataframe(
        display_df[key_cols],
        width='stretch',
        height=600,
        hide_index=True
    )

    # Detailed view in expander
    with st.expander("📊 View All Columns (Including Bid/Ask/Volume/OI)"):
        st.dataframe(
            display_df,
            width='stretch',
            height=600,
            hide_index=True
        )

    # Download CSV
    st.download_button(
        label="📥 Download as CSV",
        data=df.to_csv(index=False),
        file_name="premium_opportunities.csv",
        mime="text/csv"
    )

    # Best opportunities highlight
    st.markdown("### ⭐ Top 10 Best Opportunities")

    top_10 = display_df.head(10)[key_cols]
    st.dataframe(top_10, width='stretch', hide_index=True)

    # Near perfect delta opportunities
    st.markdown("### 🎯 Near Perfect 30 Delta (-0.25 to -0.35)")

    perfect_delta_df = df[df['Delta'].abs().between(0.25, 0.35)].copy()

    if not perfect_delta_df.empty:
        perfect_display = perfect_delta_df.copy()
        perfect_display['Stock Price'] = perfect_display['Stock Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        perfect_display['Strike'] = perfect_display['Strike'].apply(lambda x: f"${x:.2f}")
        perfect_display['Premium'] = perfect_display['Premium'].apply(lambda x: f"${x:.0f}")
        perfect_display['Monthly %'] = perfect_display['Monthly %'].apply(lambda x: f"{x:.2f}%")
        perfect_display['Delta'] = perfect_display['Delta'].apply(lambda x: f"{x:.3f}")
        perfect_display['Prob Win %'] = perfect_display['Prob Win %'].apply(lambda x: f"{x:.0f}%")

        st.dataframe(
            perfect_display[key_cols].head(20),
            width='stretch',
            height=400,
            hide_index=True
        )
    else:
        st.info("No opportunities found in the 25-35 delta range with current filters.")

# Refresh button
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Footer
st.caption("💡 Best Practice: Target 30 delta for ~70% win rate | 30-45 DTE for optimal theta decay | High IV = Higher premiums")
