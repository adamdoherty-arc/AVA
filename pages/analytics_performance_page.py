"""
Analytics Performance Dashboard for Kalshi Predictions

Comprehensive monitoring dashboard displaying:
- Performance metrics (accuracy, ROI, Sharpe ratio)
- Calibration curves
- Sector analysis
- Time series charts
- Best/worst predictions
- Model comparison

Author: Python Pro Agent
Created: 2025-11-09
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.analytics.performance_tracker import PerformanceTracker
from src.analytics.backtest import BacktestEngine, BacktestConfig


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

def configure_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Analytics Performance",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            color: white;
            margin-bottom: 1rem;
        }
        .positive {
            color: #00FF00;
        }
        .negative {
            color: #FF4444;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 24px;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)
def load_performance_summary(market_type: Optional[str] = None, days: Optional[int] = None) -> Dict:
    """Load overall performance summary"""
    tracker = PerformanceTracker()
    return tracker.get_performance_summary(market_type, days)


@st.cache_data(ttl=300)
def load_performance_by_confidence() -> pd.DataFrame:
    """Load performance grouped by confidence level"""
    tracker = PerformanceTracker()
    return tracker.get_performance_by_confidence()


@st.cache_data(ttl=300)
def load_performance_by_sector(limit: int = 20) -> pd.DataFrame:
    """Load performance by sector"""
    tracker = PerformanceTracker()
    return tracker.get_performance_by_sector(limit)


@st.cache_data(ttl=300)
def load_performance_over_time(days: int = 30) -> pd.DataFrame:
    """Load daily performance metrics"""
    tracker = PerformanceTracker()
    return tracker.get_performance_over_time(days)


@st.cache_data(ttl=300)
def load_calibration_data() -> Dict:
    """Load calibration curve data"""
    tracker = PerformanceTracker()
    return tracker.get_calibration_data()


@st.cache_data(ttl=300)
def load_best_predictions(limit: int = 10) -> pd.DataFrame:
    """Load best predictions"""
    tracker = PerformanceTracker()
    return tracker.get_best_predictions(limit)


@st.cache_data(ttl=300)
def load_worst_predictions(limit: int = 10) -> pd.DataFrame:
    """Load worst predictions"""
    tracker = PerformanceTracker()
    return tracker.get_worst_predictions(limit)


@st.cache_data(ttl=300)
def load_risk_metrics(days: int = 30) -> Dict:
    """Load risk-adjusted metrics"""
    tracker = PerformanceTracker()
    sharpe_sortino = tracker.calculate_sharpe_sortino(days)
    drawdown = tracker.calculate_max_drawdown_from_db(days)
    return {**sharpe_sortino, **drawdown}


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def render_summary_metrics(summary: Dict):
    """Render summary metrics at top of page"""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Predictions",
            f"{summary['total_predictions']:,}",
            f"{summary['settled_predictions']} settled"
        )

    with col2:
        accuracy = summary['accuracy']
        st.metric(
            "Accuracy",
            f"{accuracy:.1f}%",
            delta=f"{accuracy - 50:.1f}% vs random" if accuracy > 0 else None
        )

    with col3:
        total_pnl = summary['total_pnl']
        pnl_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric(
            "Total P&L",
            f"${total_pnl:,.2f}",
            delta=f"{summary['avg_roi']:.2f}% avg ROI",
            delta_color=pnl_color
        )

    with col4:
        st.metric(
            "Brier Score",
            f"{summary['avg_brier_score']:.4f}",
            help="Lower is better. 0 = perfect calibration, 1 = worst"
        )

    with col5:
        st.metric(
            "Log Loss",
            f"{summary['avg_log_loss']:.4f}",
            help="Lower is better. Measures prediction quality"
        )


def render_calibration_chart(calibration_data: Dict):
    """Render calibration curve"""
    if not calibration_data or not calibration_data.get('bin_confidence'):
        st.warning("No calibration data available")
        return

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash', width=2)
    ))

    # Actual calibration
    bin_confidence = calibration_data['bin_confidence']
    bin_accuracy = calibration_data['bin_accuracy']

    # Filter out NaN values
    valid_indices = [i for i, acc in enumerate(bin_accuracy) if not pd.isna(acc)]
    valid_confidence = [bin_confidence[i] for i in valid_indices]
    valid_accuracy = [bin_accuracy[i] for i in valid_indices]

    if valid_confidence:
        fig.add_trace(go.Scatter(
            x=valid_confidence,
            y=valid_accuracy,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        ))

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        hovermode='closest',
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show ECE metric
    ece = calibration_data.get('ece', 0)
    st.info(f"Expected Calibration Error (ECE): {ece:.4f} - Lower is better (well-calibrated < 0.05)")


def render_performance_by_confidence(df: pd.DataFrame):
    """Render performance by confidence bucket"""
    if df.empty:
        st.warning("No confidence data available")
        return

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Accuracy by Confidence", "Total P&L by Confidence"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Accuracy chart
    fig.add_trace(
        go.Bar(
            x=df['confidence_bucket'],
            y=df['accuracy'],
            name='Accuracy',
            marker_color='#667eea',
            text=df['accuracy'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        ),
        row=1, col=1
    )

    # P&L chart
    fig.add_trace(
        go.Bar(
            x=df['confidence_bucket'],
            y=df['total_pnl'],
            name='Total P&L',
            marker_color=df['total_pnl'].apply(lambda x: '#00cc00' if x >= 0 else '#ff4444'),
            text=df['total_pnl'].apply(lambda x: f"${x:.0f}"),
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Confidence Level", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Level", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=2)

    fig.update_layout(height=400, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def render_sector_performance(df: pd.DataFrame):
    """Render sector performance chart"""
    if df.empty:
        st.warning("No sector data available")
        return

    # Top 10 sectors by P&L
    df_top = df.head(10)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Sectors by P&L", "Accuracy by Sector"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # P&L chart
    fig.add_trace(
        go.Bar(
            x=df_top['sector'],
            y=df_top['total_pnl'],
            name='Total P&L',
            marker_color=df_top['total_pnl'].apply(lambda x: '#00cc00' if x >= 0 else '#ff4444'),
            text=df_top['total_pnl'].apply(lambda x: f"${x:.0f}"),
            textposition='outside'
        ),
        row=1, col=1
    )

    # Accuracy chart
    fig.add_trace(
        go.Bar(
            x=df_top['sector'],
            y=df_top['accuracy'],
            name='Accuracy',
            marker_color='#667eea',
            text=df_top['accuracy'].apply(lambda x: f"{x:.1f}%"),
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Sector", row=1, col=1, tickangle=-45)
    fig.update_xaxes(title_text="Sector", row=1, col=2, tickangle=-45)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

    fig.update_layout(height=500, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def render_performance_over_time(df: pd.DataFrame):
    """Render time series performance chart"""
    if df.empty:
        st.warning("No time series data available")
        return

    # Sort by date
    df = df.sort_values('date')
    df['cumulative_pnl'] = df['daily_pnl'].cumsum()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cumulative P&L Over Time", "Daily Accuracy"),
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1
    )

    # Cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ),
        row=1, col=1
    )

    # Accuracy
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['accuracy'],
            mode='lines+markers',
            name='Daily Accuracy',
            line=dict(color='#00cc00', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )

    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1,
                  annotation_text="Random (50%)")

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)

    fig.update_layout(height=600, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)


def render_best_worst_predictions(best_df: pd.DataFrame, worst_df: pd.DataFrame):
    """Render best and worst predictions tables"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Best Predictions (Highest P&L)")
        if not best_df.empty:
            best_display = best_df[['ticker', 'predicted_outcome', 'actual_outcome',
                                   'confidence_score', 'pnl', 'roi_percent']].copy()
            best_display['pnl'] = best_display['pnl'].apply(lambda x: f"${x:.2f}")
            best_display['roi_percent'] = best_display['roi_percent'].apply(lambda x: f"{x:.1f}%")
            best_display.columns = ['Ticker', 'Predicted', 'Actual', 'Confidence', 'P&L', 'ROI']
            st.dataframe(best_display, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions available")

    with col2:
        st.subheader("Worst Predictions (Lowest P&L)")
        if not worst_df.empty:
            worst_display = worst_df[['ticker', 'predicted_outcome', 'actual_outcome',
                                     'confidence_score', 'pnl', 'roi_percent']].copy()
            worst_display['pnl'] = worst_display['pnl'].apply(lambda x: f"${x:.2f}")
            worst_display['roi_percent'] = worst_display['roi_percent'].apply(lambda x: f"{x:.1f}%")
            worst_display.columns = ['Ticker', 'Predicted', 'Actual', 'Confidence', 'P&L', 'ROI']
            st.dataframe(worst_display, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions available")


def render_risk_metrics(metrics: Dict):
    """Render risk-adjusted metrics"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sharpe = metrics.get('sharpe_ratio', 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            help="Risk-adjusted return. >1 is good, >2 is excellent"
        )

    with col2:
        sortino = metrics.get('sortino_ratio', 0)
        st.metric(
            "Sortino Ratio",
            f"{sortino:.2f}",
            help="Return vs downside risk. Higher is better"
        )

    with col3:
        max_dd = metrics.get('max_drawdown_pct', 0)
        st.metric(
            "Max Drawdown",
            f"{max_dd:.2f}%",
            delta=f"${metrics.get('max_drawdown_amount', 0):.2f}",
            delta_color="inverse"
        )

    with col4:
        # Calculate Calmar if we have the data
        st.metric(
            "Risk Level",
            "Moderate" if max_dd < 15 else "High" if max_dd < 30 else "Very High",
            help="Based on maximum drawdown"
        )


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def render_backtest_runner():
    """Render backtesting interface"""
    st.subheader("Strategy Backtesting")

    with st.form("backtest_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            strategy_name = st.text_input("Strategy Name", "My Strategy")
            initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
            position_sizing = st.selectbox("Position Sizing", ["kelly", "fixed", "proportional"])

        with col2:
            kelly_fraction = st.slider("Kelly Fraction", 0.1, 1.0, 0.25, 0.05) if position_sizing == "kelly" else 0.25
            max_position = st.slider("Max Position Size (%)", 1.0, 25.0, 10.0, 1.0)
            max_drawdown_limit = st.slider("Max Drawdown Limit (%)", 10.0, 50.0, 20.0, 5.0)

        with col3:
            min_confidence = st.slider("Min Confidence", 0.0, 100.0, 60.0, 5.0)
            min_edge = st.slider("Min Edge (%)", 0.0, 20.0, 5.0, 1.0)
            market_types = st.multiselect("Market Types", ["nfl", "college"], default=["nfl", "college"])

        submitted = st.form_submit_button("Run Backtest")

    if submitted:
        with st.spinner("Running backtest..."):
            config = BacktestConfig(
                name=strategy_name,
                strategy_name=strategy_name.lower().replace(" ", "_"),
                initial_capital=initial_capital,
                position_sizing=position_sizing,
                kelly_fraction=kelly_fraction,
                max_position_size=max_position,
                max_drawdown_limit=max_drawdown_limit,
                min_confidence=min_confidence,
                min_edge=min_edge,
                market_types=market_types if market_types else None
            )

            engine = BacktestEngine()
            results = engine.run_backtest(config)

            # Display results
            st.success(f"Backtest Complete: {results['total_trades']} trades")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")
            with col2:
                st.metric("Total Return", f"{results['total_return_pct']:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")

            # Equity curve
            if results['equity_curve']:
                equity_df = pd.DataFrame({
                    'Trade': range(len(results['equity_curve'])),
                    'Equity': results['equity_curve']
                })

                fig = px.line(equity_df, x='Trade', y='Equity',
                            title='Equity Curve',
                            labels={'Equity': 'Portfolio Value ($)'})
                fig.update_traces(line_color='#667eea', line_width=3)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    configure_page()

    # Header
    st.title("📊 Analytics Performance Dashboard")
    st.markdown("Comprehensive monitoring and analysis of prediction performance")

    # Sidebar filters
    st.sidebar.header("Filters")
    time_period = st.sidebar.selectbox(
        "Time Period",
        ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )

    days_map = {
        "All Time": None,
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90
    }
    days = days_map[time_period]

    market_type_filter = st.sidebar.selectbox(
        "Market Type",
        ["All", "NFL", "College"]
    )

    market_type = None if market_type_filter == "All" else market_type_filter.lower()

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    with st.spinner("Loading performance data..."):
        summary = load_performance_summary(market_type, days)
        risk_metrics = load_risk_metrics(days or 30)

    # Summary metrics
    render_summary_metrics(summary)

    st.divider()

    # Risk metrics
    st.subheader("Risk-Adjusted Metrics")
    render_risk_metrics(risk_metrics)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Calibration",
        "📊 Performance Analysis",
        "⏰ Time Series",
        "🎯 Best/Worst",
        "🔬 Backtesting"
    ])

    with tab1:
        st.subheader("Model Calibration Analysis")
        calibration_data = load_calibration_data()
        render_calibration_chart(calibration_data)

        st.divider()

        st.subheader("Performance by Confidence Level")
        conf_df = load_performance_by_confidence()
        render_performance_by_confidence(conf_df)

    with tab2:
        st.subheader("Performance by Sector")
        sector_df = load_performance_by_sector(20)
        render_sector_performance(sector_df)

    with tab3:
        st.subheader("Performance Over Time")
        time_df = load_performance_over_time(days or 30)
        render_performance_over_time(time_df)

    with tab4:
        best_df = load_best_predictions(10)
        worst_df = load_worst_predictions(10)
        render_best_worst_predictions(best_df, worst_df)

    with tab5:
        render_backtest_runner()

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
