# This is the CORRECT implementation for the TradingView Watchlists Auto-Sync tab
# Replace lines 1215-1585 in dashboard.py with this code

# Helper function: Get main table data with 7, 14, 30 DTE columns
def get_multi_dte_table(tv_manager, stock_symbols, min_stock_price, max_stock_price,
                       min_delta, max_delta, min_premium, min_monthly):
    """Returns table with best options for 7, 14, and 30 DTE"""
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    query = """
        WITH Options7 AS (
            SELECT DISTINCT ON (sp.symbol)
                sp.symbol,
                sp.premium as premium_7,
                sp.delta as delta_7,
                sp.monthly_return as monthly_7,
                sp.dte as dte_7
            FROM stock_premiums sp
            WHERE sp.symbol = ANY(%s)
                AND sp.dte BETWEEN 5 AND 9
                AND ABS(sp.delta) BETWEEN %s AND %s
                AND sp.premium >= %s
                AND sp.monthly_return >= %s
            ORDER BY sp.symbol, sp.monthly_return DESC
        ),
        Options14 AS (
            SELECT DISTINCT ON (sp.symbol)
                sp.symbol,
                sp.premium as premium_14,
                sp.delta as delta_14,
                sp.monthly_return as monthly_14,
                sp.dte as dte_14
            FROM stock_premiums sp
            WHERE sp.symbol = ANY(%s)
                AND sp.dte BETWEEN 12 AND 16
                AND ABS(sp.delta) BETWEEN %s AND %s
                AND sp.premium >= %s
                AND sp.monthly_return >= %s
            ORDER BY sp.symbol, sp.monthly_return DESC
        ),
        Options30 AS (
            SELECT DISTINCT ON (sp.symbol)
                sp.symbol,
                sp.premium as premium_30,
                sp.delta as delta_30,
                sp.monthly_return as monthly_30,
                sp.dte as dte_30
            FROM stock_premiums sp
            WHERE sp.symbol = ANY(%s)
                AND sp.dte BETWEEN 28 AND 32
                AND ABS(sp.delta) BETWEEN %s AND %s
                AND sp.premium >= %s
                AND sp.monthly_return >= %s
            ORDER BY sp.symbol, sp.monthly_return DESC
        )
        SELECT
            sd.symbol,
            sd.current_price as stock_price,
            o7.premium_7, o7.delta_7, o7.monthly_7, o7.dte_7,
            o14.premium_14, o14.delta_14, o14.monthly_14, o14.dte_14,
            o30.premium_30, o30.delta_30, o30.monthly_30, o30.dte_30
        FROM stock_data sd
        LEFT JOIN Options7 o7 ON sd.symbol = o7.symbol
        LEFT JOIN Options14 o14 ON sd.symbol = o14.symbol
        LEFT JOIN Options30 o30 ON sd.symbol = o30.symbol
        WHERE sd.symbol = ANY(%s)
            AND sd.current_price BETWEEN %s AND %s
            AND (o7.premium_7 IS NOT NULL OR o14.premium_14 IS NOT NULL OR o30.premium_30 IS NOT NULL)
        ORDER BY GREATEST(
            COALESCE(o7.monthly_7, 0),
            COALESCE(o14.monthly_14, 0),
            COALESCE(o30.monthly_30, 0)
        ) DESC
    """

    params = (
        stock_symbols, min_delta, max_delta, min_premium, min_monthly,  # For 7 DTE
        stock_symbols, min_delta, max_delta, min_premium, min_monthly,  # For 14 DTE
        stock_symbols, min_delta, max_delta, min_premium, min_monthly,  # For 30 DTE
        stock_symbols, min_stock_price, max_stock_price  # Final filter
    )

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

# Helper function: Get all strikes for specific DTE ranges when expanding
def get_expanded_strikes(tv_manager, symbol):
    """Returns multiple strikes for 7, 14, 30 DTE ranges"""
    conn = tv_manager.get_connection()
    cur = conn.cursor()

    # Get actual DTEs closest to 7, 14, 30
    query_dtes = """
        WITH dte_distances AS (
            SELECT DISTINCT
                dte,
                ABS(dte - 7) as dist7,
                ABS(dte - 14) as dist14,
                ABS(dte - 30) as dist30
            FROM stock_premiums
            WHERE symbol = %s
            AND dte BETWEEN 5 AND 35
        )
        SELECT DISTINCT dte
        FROM (
            (SELECT dte FROM dte_distances ORDER BY dist7 LIMIT 1)
            UNION
            (SELECT dte FROM dte_distances ORDER BY dist14 LIMIT 1)
            UNION
            (SELECT dte FROM dte_distances ORDER BY dist30 LIMIT 1)
        ) as dtes
        ORDER BY dte
    """
    cur.execute(query_dtes, (symbol,))
    available_dtes = [row[0] for row in cur.fetchall()]

    if not available_dtes:
        cur.close()
        conn.close()
        return []

    # Get all strikes for those DTEs
    query = """
        SELECT
            sp.strike_price,
            sp.dte,
            sp.premium,
            sp.bid,
            sp.ask,
            sp.delta,
            sp.implied_volatility,
            sp.volume,
            sp.open_interest,
            sp.monthly_return
        FROM stock_premiums sp
        WHERE sp.symbol = %s
            AND sp.dte = ANY(%s)
            AND ABS(sp.delta) BETWEEN 0.25 AND 0.40
        ORDER BY sp.dte, sp.strike_price
    """

    cur.execute(query, (symbol, available_dtes))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

# Initialize session state
if 'expanded_symbol' not in st.session_state:
    st.session_state.expanded_symbol = None

# Filters Section
st.markdown("### 💵 Cash-Secured Put Options - Multi-DTE View")
st.caption("See best premiums for 7, 14, and 30 days out. Click any row to see all available strikes.")

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    min_stock_price = st.number_input("Min Stock Price ($)", value=10.0, min_value=0.0, step=10.0, key="filter_min_stock_price")
    max_stock_price = st.number_input("Max Stock Price ($)", value=1000.0, min_value=10.0, step=50.0, key="filter_max_stock_price")
with col_f2:
    delta_range = st.slider("Delta Range", min_value=0.20, max_value=0.40, value=(0.25, 0.35), step=0.01, key="filter_delta")
    min_premium = st.number_input("Min Premium ($)", value=2.0, min_value=0.0, step=0.5, key="filter_min_premium")
with col_f3:
    min_monthly = st.number_input("Min Monthly %", value=3.0, min_value=0.0, step=0.5, key="filter_min_monthly")
    if st.button("Reset Filters", key="reset_filters"):
        st.rerun()

# Load data
if not stock_symbols:
    rows = []
else:
    rows = get_multi_dte_table(
        tv_manager,
        stock_symbols,
        min_stock_price,
        max_stock_price,
        delta_range[0],
        delta_range[1],
        min_premium,
        min_monthly
    )

if not rows:
    st.warning(f"⏳ No options found for {selected_watchlist} matching your filters. Try adjusting filters or click 'Sync Prices & Premiums'.")
else:
    # Create DataFrame with proper structure
    df = pd.DataFrame(rows, columns=[
        'Symbol', 'Stock Price',
        'Premium_7', 'Delta_7', 'Monthly_7', 'DTE_7',
        'Premium_14', 'Delta_14', 'Monthly_14', 'DTE_14',
        'Premium_30', 'Delta_30', 'Monthly_30', 'DTE_30'
    ])

    # Convert numeric columns
    numeric_cols = ['Stock Price',
                    'Premium_7', 'Delta_7', 'Monthly_7', 'DTE_7',
                    'Premium_14', 'Delta_14', 'Monthly_14', 'DTE_14',
                    'Premium_30', 'Delta_30', 'Monthly_30', 'DTE_30']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate score (average of available monthly returns)
    df['Score'] = df[['Monthly_7', 'Monthly_14', 'Monthly_30']].mean(axis=1, skipna=True)

    # Format display columns
    display_df = df[['Symbol', 'Stock Price']].copy()
    display_df['7 DTE'] = df['Premium_7'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    display_df['14 DTE'] = df['Premium_14'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    display_df['30 DTE'] = df['Premium_30'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    display_df['Score'] = df['Score'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
    display_df['Stock Price'] = df['Stock Price'].apply(lambda x: f"${x:.2f}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Found", len(df))
    with col2:
        avg_7 = df['Premium_7'].mean()
        st.metric("Avg 7-Day Premium", f"${avg_7:.2f}" if pd.notna(avg_7) else "N/A")
    with col3:
        avg_14 = df['Premium_14'].mean()
        st.metric("Avg 14-Day Premium", f"${avg_14:.2f}" if pd.notna(avg_14) else "N/A")
    with col4:
        avg_30 = df['Premium_30'].mean()
        st.metric("Avg 30-Day Premium", f"${avg_30:.2f}" if pd.notna(avg_30) else "N/A")

    # Main sortable table
    st.markdown("#### 📊 Cash-Secured Puts by Expiration (Click row to expand)")

    event = st.dataframe(
        display_df,
        hide_index=True,
        width='stretch',
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Stock Price": st.column_config.TextColumn("Stock $"),
            "7 DTE": st.column_config.TextColumn("7 Days"),
            "14 DTE": st.column_config.TextColumn("14 Days"),
            "30 DTE": st.column_config.TextColumn("30 Days"),
            "Score": st.column_config.TextColumn("Avg Score")
        }
    )

    # Handle row selection and expansion
    if event.selection and event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        selected_symbol = df.iloc[selected_row_idx]['Symbol']

        # Toggle expansion
        if st.session_state.expanded_symbol == selected_symbol:
            st.session_state.expanded_symbol = None
        else:
            st.session_state.expanded_symbol = selected_symbol
        st.rerun()

    # Display expanded view
    if st.session_state.expanded_symbol:
        symbol = st.session_state.expanded_symbol
        stock_price = df[df['Symbol'] == symbol]['Stock Price'].iloc[0] if symbol in df['Symbol'].values else 0

        st.markdown(f"### 📋 {symbol} - Stock Price: ${stock_price:.2f}")

        col_close1, col_close2 = st.columns([6, 1])
        with col_close2:
            if st.button("✖ Close", key=f"close_{symbol}"):
                st.session_state.expanded_symbol = None
                st.rerun()

        # Fetch expanded data
        expanded_rows = get_expanded_strikes(tv_manager, symbol)

        if expanded_rows:
            exp_df = pd.DataFrame(expanded_rows, columns=[
                'Strike', 'DTE', 'Premium', 'Bid', 'Ask', 'Delta', 'IV',
                'Volume', 'OI', 'Monthly %'
            ])

            # Convert numeric columns
            for col in exp_df.columns:
                exp_df[col] = pd.to_numeric(exp_df[col], errors='coerce')

            # Group by unique DTEs
            unique_dtes = sorted(exp_df['DTE'].unique())

            for dte in unique_dtes:
                dte_df = exp_df[exp_df['DTE'] == dte].copy()

                # Determine label
                if dte <= 9:
                    label = f"📅 ~7 Days ({int(dte)} DTE)"
                elif dte <= 16:
                    label = f"📅 ~14 Days ({int(dte)} DTE)"
                else:
                    label = f"📅 ~30 Days ({int(dte)} DTE)"

                st.markdown(f"#### {label} - {len(dte_df)} strikes available")

                # Format for display
                dte_df['Strike'] = dte_df['Strike'].apply(lambda x: f"${x:.2f}")
                dte_df['Premium'] = dte_df['Premium'].apply(lambda x: f"${x:.2f}")
                dte_df['Bid'] = dte_df['Bid'].apply(lambda x: f"${x:.2f}")
                dte_df['Ask'] = dte_df['Ask'].apply(lambda x: f"${x:.2f}")
                dte_df['Delta'] = dte_df['Delta'].apply(lambda x: f"{x:.3f}")
                dte_df['IV'] = dte_df['IV'].apply(lambda x: f"{x:.1f}%")
                dte_df['Monthly %'] = dte_df['Monthly %'].apply(lambda x: f"{x:.2f}%")

                st.dataframe(
                    dte_df[['Strike', 'Premium', 'Bid', 'Ask', 'Delta', 'IV', 'Volume', 'OI', 'Monthly %']],
                    hide_index=True,
                    width='stretch'
                )
        else:
            st.info(f"No option data found for {symbol}. Try syncing data.")
