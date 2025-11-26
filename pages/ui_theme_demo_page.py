"""
UI Theme System Demonstration Page
===================================

This page demonstrates all features of the unified UI theme system.
Use this as a reference when building or updating pages.
"""

from src.ui_theme import init_page, UIComponents, get_theme
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# ============================================================================
# INITIALIZE PAGE WITH THEME
# ============================================================================

# This single line replaces st.set_page_config() and applies all theme CSS
theme = init_page("UI Theme Demo", "🎨", "wide")
components = UIComponents(theme)


# ============================================================================
# DEMO CONTENT
# ============================================================================

def main():
    """Main demonstration page"""

    # Header with live indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎨 UI Theme System Demo")
        st.caption("Demonstration of all unified theme components and styling")
    with col2:
        st.markdown(
            f"<div style='text-align: right; padding-top: 20px;'>{components.live_indicator('DEMO MODE')}</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================================
    # SECTION 1: STATUS BOXES
    # ========================================================================

    st.header("1️⃣ Status Message Boxes")
    st.markdown("Consistent, themed message boxes for different statuses")

    col1, col2 = st.columns(2)
    with col1:
        components.success_box("This is a success message - operation completed!")
        components.warning_box("This is a warning message - please review settings")

    with col2:
        components.error_box("This is an error message - something went wrong")
        components.info_box("This is an info message - new features available")

    st.divider()

    # ========================================================================
    # SECTION 2: METRICS
    # ========================================================================

    st.header("2️⃣ Themed Metrics")
    st.markdown("Metrics automatically get consistent styling with borders and backgrounds")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Portfolio Value",
            value="$125,432",
            delta="+$1,234 (1.0%)",
            help="Total portfolio value including cash"
        )

    with col2:
        st.metric(
            label="Day P&L",
            value="+$542",
            delta="+0.43%",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="Open Positions",
            value="8",
            delta="-2",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Win Rate",
            value="72.5%",
            delta="+5.2%"
        )

    st.divider()

    # ========================================================================
    # SECTION 3: BADGES
    # ========================================================================

    st.header("3️⃣ Status Badges")
    st.markdown("Inline status indicators with consistent theming")

    badge_col1, badge_col2 = st.columns(2)

    with badge_col1:
        st.markdown("**Trade Status Examples:**")
        st.markdown(
            f"Position: {components.status_badge('success', 'Active')} | "
            f"Risk: {components.status_badge('warning', 'Moderate')} | "
            f"Connection: {components.status_badge('error', 'Failed')}",
            unsafe_allow_html=True
        )

    with badge_col2:
        st.markdown("**System Status Examples:**")
        st.markdown(
            f"Database: {components.status_badge('success', 'Connected')} | "
            f"API: {components.status_badge('warning', 'Slow')} | "
            f"Updates: {components.status_badge('info', 'Available')}",
            unsafe_allow_html=True
        )

    st.divider()

    # ========================================================================
    # SECTION 4: CARDS
    # ========================================================================

    st.header("4️⃣ Card Components")
    st.markdown("Reusable card components with consistent styling and hover effects")

    card_col1, card_col2, card_col3 = st.columns(3)

    with card_col1:
        components.card(
            title="Daily Summary",
            content=f"""
                <strong>Trades:</strong> 12<br>
                <strong>Win Rate:</strong> 75%<br>
                <strong>P&L:</strong> <span style='color: {theme.colors.success}'>+$1,234</span>
            """
        )

    with card_col2:
        components.card(
            title="Market Status",
            content=f"""
                <strong>Status:</strong> {components.status_badge('success', 'Open')}<br>
                <strong>Volatility:</strong> Moderate<br>
                <strong>Volume:</strong> High
            """
        )

    with card_col3:
        components.card(
            title="System Health",
            content=f"""
                <strong>Uptime:</strong> 99.9%<br>
                <strong>Latency:</strong> 45ms<br>
                <strong>Errors:</strong> 0
            """
        )

    st.divider()

    # ========================================================================
    # SECTION 5: BUTTONS
    # ========================================================================

    st.header("5️⃣ Themed Buttons")
    st.markdown("Buttons with hover effects, transitions, and consistent styling")

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("🚀 Execute Trade", type="primary", use_container_width=True):
            components.success_box("Trade executed successfully!")

    with btn_col2:
        if st.button("📊 View Analytics", type="secondary", use_container_width=True):
            components.info_box("Loading analytics dashboard...")

    with btn_col3:
        if st.button("❌ Cancel", use_container_width=True):
            components.warning_box("Operation cancelled")

    st.divider()

    # ========================================================================
    # SECTION 6: DATA TABLES
    # ========================================================================

    st.header("6️⃣ Themed Data Tables")
    st.markdown("DataFrames with automatic styling - headers, alternating rows, hover effects")

    # Sample data
    sample_data = pd.DataFrame({
        'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'Position': ['Long', 'Long', 'Short', 'Long', 'Long'],
        'Quantity': [100, 50, 75, 30, 40],
        'Entry Price': [150.00, 2800.00, 300.00, 3200.00, 250.00],
        'Current Price': [155.50, 2850.00, 295.00, 3250.00, 265.00],
        'P&L': ['+$550', '+$2,500', '+$375', '+$1,500', '+$600'],
        'Status': ['Active', 'Active', 'Active', 'Active', 'Active']
    })

    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    st.divider()

    # ========================================================================
    # SECTION 7: CHARTS
    # ========================================================================

    st.header("7️⃣ Themed Plotly Charts")
    st.markdown("Charts with consistent colors, fonts, and styling")

    chart_col1, chart_col2 = st.columns(2)

    # Get chart template
    template = theme.get_chart_template()

    with chart_col1:
        # Line chart
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        values = np.cumsum(np.random.randn(30)) + 100

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color=theme.colors.chart_primary, width=2)
        ))
        fig1.update_layout(
            **template['layout'],
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        # Bar chart
        categories = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
        values = [35, 25, 20, 12, 8]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=theme.colors.chart_primary
        ))
        fig2.update_layout(
            **template['layout'],
            title='Portfolio Allocation by Sector',
            xaxis_title='Sector',
            yaxis_title='Allocation (%)',
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ========================================================================
    # SECTION 8: INPUTS
    # ========================================================================

    st.header("8️⃣ Themed Input Fields")
    st.markdown("Form inputs with consistent borders and focus states")

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        symbol = st.text_input("Symbol", value="AAPL", placeholder="Enter ticker symbol")

    with input_col2:
        quantity = st.number_input("Quantity", value=100, min_value=1)

    with input_col3:
        action = st.selectbox("Action", ["Buy", "Sell", "Hold"])

    st.divider()

    # ========================================================================
    # SECTION 9: TABS
    # ========================================================================

    st.header("9️⃣ Themed Tabs")
    st.markdown("Tab navigation with consistent styling and active states")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analytics", "⚙️ Settings"])

    with tab1:
        st.markdown("### Overview Tab")
        components.info_box("This tab shows the overview of your portfolio")
        st.metric("Total Value", "$125,432", "+1.0%")

    with tab2:
        st.markdown("### Analytics Tab")
        components.success_box("Analytics data loaded successfully")
        st.write("Charts and detailed analytics would go here...")

    with tab3:
        st.markdown("### Settings Tab")
        st.checkbox("Enable notifications")
        st.checkbox("Auto-refresh data")
        st.slider("Refresh interval (seconds)", 5, 60, 30)

    st.divider()

    # ========================================================================
    # SECTION 10: COLOR PALETTE
    # ========================================================================

    st.header("🎨 Complete Color Palette")
    st.markdown("All available theme colors for custom components")

    color_col1, color_col2, color_col3 = st.columns(3)

    with color_col1:
        st.markdown("**Primary Colors:**")
        for color_name, color_value in [
            ('Primary', theme.colors.primary),
            ('Primary Dark', theme.colors.primary_dark),
            ('Primary Light', theme.colors.primary_light),
            ('Secondary', theme.colors.secondary),
            ('Accent', theme.colors.accent),
        ]:
            st.markdown(
                f'<div style="background-color: {color_value}; color: white; padding: 10px; margin: 5px 0; border-radius: 4px;">'
                f'{color_name}: {color_value}</div>',
                unsafe_allow_html=True
            )

    with color_col2:
        st.markdown("**Status Colors:**")
        for color_name, color_value in [
            ('Success', theme.colors.success),
            ('Warning', theme.colors.warning),
            ('Error', theme.colors.error),
            ('Info', theme.colors.info),
            ('Live', theme.colors.live),
        ]:
            st.markdown(
                f'<div style="background-color: {color_value}; color: white; padding: 10px; margin: 5px 0; border-radius: 4px;">'
                f'{color_name}: {color_value}</div>',
                unsafe_allow_html=True
            )

    with color_col3:
        st.markdown("**Chart Colors:**")
        for color_name, color_value in [
            ('Chart Primary', theme.colors.chart_primary),
            ('Chart Positive', theme.colors.chart_positive),
            ('Chart Negative', theme.colors.chart_negative),
            ('Chart Neutral', theme.colors.chart_neutral),
        ]:
            st.markdown(
                f'<div style="background-color: {color_value}; color: white; padding: 10px; margin: 5px 0; border-radius: 4px;">'
                f'{color_name}: {color_value}</div>',
                unsafe_allow_html=True
            )

    st.divider()

    # ========================================================================
    # CODE EXAMPLES
    # ========================================================================

    st.header("💻 Code Examples")

    with st.expander("View implementation code"):
        st.code("""
# Import the theme system
from src.ui_theme import init_page, UIComponents

# Initialize page (replaces st.set_page_config)
theme = init_page("My Page", "📊", "wide")
components = UIComponents(theme)

# Use components
components.success_box("Operation successful!")
components.warning_box("Warning message")

# Use theme colors
st.markdown(
    f'<div style="color: {theme.colors.primary}">Themed text</div>',
    unsafe_allow_html=True
)

# Use themed charts
template = theme.get_chart_template()
fig = go.Figure(data=[...])
fig.update_layout(**template['layout'])
        """, language="python")

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #6B7280; padding: 20px;'>"
        "✨ UI Theme System v1.0 | Magnus Trading Platform<br>"
        "See <code>docs/UI_THEME_GUIDE.md</code> for complete documentation"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
