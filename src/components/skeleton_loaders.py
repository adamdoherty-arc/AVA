"""
Skeleton Loaders for Better Perceived Performance

This module provides skeleton/placeholder components for Streamlit that improve
perceived performance by showing content structure while data is loading.

Skeleton loaders make the application feel faster and more responsive by:
- Showing immediate visual feedback
- Indicating where content will appear
- Preventing layout shifts
- Reducing perceived wait time

Usage:
    from src.components.skeleton_loaders import (
        skeleton_dataframe,
        skeleton_metric_row,
        skeleton_chart,
        skeleton_card,
        with_skeleton
    )

    # Manual usage
    placeholder = st.empty()
    with placeholder:
        skeleton_dataframe(rows=10, cols=5)

    # Load data
    data = expensive_database_query()

    # Replace skeleton with real data
    with placeholder:
        st.dataframe(data)

    # Or use decorator
    @with_skeleton(skeleton_dataframe, rows=10, cols=5)
    def load_positions():
        return get_positions()

Features:
- Pre-built skeletons for common Streamlit components
- Customizable sizes and styles
- Decorator pattern for easy integration
- Responsive and accessible
- Minimal performance overhead
"""

import streamlit as st
import pandas as pd
from typing import Optional, Callable, Any
from functools import wraps
import time


def skeleton_dataframe(
    rows: int = 5,
    cols: int = 4,
    height: Optional[int] = None,
    key: Optional[str] = None
):
    """
    Display a skeleton placeholder for a DataFrame.

    Args:
        rows: Number of skeleton rows to display
        cols: Number of skeleton columns to display
        height: Optional fixed height in pixels
        key: Optional unique key for the component
    """
    # Create dummy data with loading indicators
    dummy_data = {
        f"Column {i+1}": ["⋯ Loading" if j % 2 == 0 else "━━━━━━" for j in range(rows)]
        for i in range(cols)
    }

    df = pd.DataFrame(dummy_data)

    # Apply styling to make it look like a skeleton
    styled_df = df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': '#d0d0d0',
        'text-align': 'center',
        'font-style': 'italic'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#e0e2e6'),
            ('color', '#c0c0c0'),
            ('text-align', 'center')
        ]}
    ])

    if height:
        st.dataframe(styled_df, height=height, key=key)
    else:
        st.dataframe(styled_df, key=key)


def skeleton_metric_row(num_metrics: int = 3, key_prefix: str = "metric"):
    """
    Display a skeleton placeholder for a row of metrics.

    Args:
        num_metrics: Number of metrics to display
        key_prefix: Prefix for unique keys
    """
    cols = st.columns(num_metrics)

    for i, col in enumerate(cols):
        with col:
            st.markdown(
                f"""
                <div style="
                    padding: 1rem;
                    border-radius: 0.5rem;
                    background: linear-gradient(90deg, #f0f2f6 25%, #e0e2e6 50%, #f0f2f6 75%);
                    background-size: 200% 100%;
                    animation: shimmer 1.5s infinite;
                    height: 100px;
                ">
                    <div style="height: 20px; background: #d0d0d0; margin-bottom: 10px; border-radius: 4px;"></div>
                    <div style="height: 40px; background: #c0c0c0; border-radius: 4px;"></div>
                </div>
                <style>
                    @keyframes shimmer {{
                        0% {{ background-position: 200% 0; }}
                        100% {{ background-position: -200% 0; }}
                    }}
                </style>
                """,
                unsafe_allow_html=True
            )


def skeleton_chart(height: int = 400, key: Optional[str] = None):
    """
    Display a skeleton placeholder for a chart.

    Args:
        height: Chart height in pixels
        key: Optional unique key
    """
    st.markdown(
        f"""
        <div style="
            height: {height}px;
            background: linear-gradient(135deg, #f0f2f6 25%, #e0e2e6 50%, #f0f2f6 75%);
            background-size: 200% 200%;
            animation: shimmer 2s infinite;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #a0a0a0;
            font-size: 18px;
            font-style: italic;
        ">
            📊 Loading chart...
        </div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: 200% 200%; }}
                100% {{ background-position: -200% -200%; }}
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def skeleton_card(
    title: bool = True,
    content_lines: int = 3,
    height: Optional[int] = None,
    key: Optional[str] = None
):
    """
    Display a skeleton placeholder for a content card.

    Args:
        title: Whether to show a title skeleton
        content_lines: Number of content line skeletons
        height: Optional fixed height in pixels
        key: Optional unique key
    """
    title_html = """
        <div style="height: 24px; background: #c0c0c0; margin-bottom: 12px; border-radius: 4px; width: 60%;"></div>
    """ if title else ""

    content_html = "\n".join([
        f'<div style="height: 16px; background: #d0d0d0; margin-bottom: 8px; border-radius: 4px; width: {90 - i*10}%;"></div>'
        for i in range(content_lines)
    ])

    height_style = f"height: {height}px;" if height else ""

    st.markdown(
        f"""
        <div style="
            padding: 1.5rem;
            border-radius: 0.5rem;
            background: #f0f2f6;
            {height_style}
        ">
            {title_html}
            {content_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def skeleton_table(rows: int = 5, cols: int = 4, key: Optional[str] = None):
    """
    Display a skeleton placeholder for a table (alternative to skeleton_dataframe).

    Uses pure HTML/CSS for more customization.

    Args:
        rows: Number of skeleton rows
        cols: Number of skeleton columns
        key: Optional unique key
    """
    header_html = "\n".join([
        f'<th style="padding: 0.5rem; background: #e0e2e6;"><div style="height: 20px; background: #c0c0c0; border-radius: 4px;"></div></th>'
        for _ in range(cols)
    ])

    rows_html = "\n".join([
        f"""
        <tr>
            {"".join([
                f'<td style="padding: 0.5rem;"><div style="height: 16px; background: #d0d0d0; border-radius: 4px;"></div></td>'
                for _ in range(cols)
            ])}
        </tr>
        """
        for _ in range(rows)
    ])

    st.markdown(
        f"""
        <table style="width: 100%; border-collapse: collapse; background: #f0f2f6; border-radius: 0.5rem; overflow: hidden;">
            <thead>
                <tr>{header_html}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )


def skeleton_sidebar_filters(num_filters: int = 3):
    """
    Display skeleton placeholders for sidebar filters.

    Args:
        num_filters: Number of filter skeletons to display
    """
    for i in range(num_filters):
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <div style="height: 16px; background: #d0d0d0; margin-bottom: 8px; border-radius: 4px; width: 50%;"></div>
                <div style="height: 36px; background: #e0e2e6; border-radius: 4px;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )


def skeleton_tabs(num_tabs: int = 3, content_height: int = 300):
    """
    Display skeleton placeholder for tabbed content.

    Args:
        num_tabs: Number of tab skeletons
        content_height: Height of content area in pixels
    """
    tabs_html = "\n".join([
        f'<div style="padding: 0.5rem 1rem; background: #e0e2e6; border-radius: 0.5rem 0.5rem 0 0; margin-right: 0.25rem; display: inline-block;"><div style="height: 16px; width: 60px; background: #c0c0c0; border-radius: 4px;"></div></div>'
        for _ in range(num_tabs)
    ])

    st.markdown(
        f"""
        <div>
            <div style="margin-bottom: -1px;">
                {tabs_html}
            </div>
            <div style="
                height: {content_height}px;
                background: #f0f2f6;
                border-radius: 0 0.5rem 0.5rem 0.5rem;
                padding: 1rem;
            ">
                <div style="height: 24px; background: #d0d0d0; margin-bottom: 12px; border-radius: 4px; width: 40%;"></div>
                <div style="height: 200px; background: #e0e2e6; border-radius: 4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def with_skeleton(
    skeleton_func: Callable,
    min_display_time: float = 0.3,
    **skeleton_kwargs
):
    """
    Decorator to show skeleton while function executes.

    Usage:
        @with_skeleton(skeleton_dataframe, rows=10, cols=5)
        def load_positions():
            return get_positions_from_db()

        # In Streamlit page:
        positions_df = load_positions()
        st.dataframe(positions_df)

    Args:
        skeleton_func: Skeleton function to call (e.g., skeleton_dataframe)
        min_display_time: Minimum time to show skeleton (prevents flicker)
        **skeleton_kwargs: Arguments to pass to skeleton function

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create placeholder
            placeholder = st.empty()

            # Show skeleton
            start_time = time.time()
            with placeholder.container():
                skeleton_func(**skeleton_kwargs)

            # Execute function
            result = func(*args, **kwargs)

            # Ensure skeleton shows for minimum time (prevent flicker)
            elapsed = time.time() - start_time
            if elapsed < min_display_time:
                time.sleep(min_display_time - elapsed)

            # Clear skeleton (will be replaced by real content outside decorator)
            placeholder.empty()

            return result

        return wrapper
    return decorator


class SkeletonContext:
    """
    Context manager for skeleton loading states.

    Usage:
        with SkeletonContext(skeleton_dataframe, rows=10, cols=5) as skeleton:
            data = expensive_query()

        # Skeleton automatically cleared
        st.dataframe(data)
    """

    def __init__(self, skeleton_func: Callable, min_display_time: float = 0.3, **skeleton_kwargs):
        self.skeleton_func = skeleton_func
        self.skeleton_kwargs = skeleton_kwargs
        self.min_display_time = min_display_time
        self.placeholder = None
        self.start_time = None

    def __enter__(self) -> None:
        self.placeholder = st.empty()
        self.start_time = time.time()

        with self.placeholder.container():
            self.skeleton_func(**self.skeleton_kwargs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure skeleton shows for minimum time
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed < self.min_display_time:
                time.sleep(self.min_display_time - elapsed)

        # Clear skeleton
        if self.placeholder:
            self.placeholder.empty()

        return False  # Don't suppress exceptions


def progressive_skeleton_load(sections: list):
    """
    Load multiple sections progressively with skeletons.

    Usage:
        sections = [
            ("Metrics", skeleton_metric_row, load_metrics, {"num_metrics": 4}),
            ("Chart", skeleton_chart, load_chart, {"height": 400}),
            ("Table", skeleton_dataframe, load_table, {"rows": 10, "cols": 5}),
        ]

        results = progressive_skeleton_load(sections)

    Args:
        sections: List of tuples (label, skeleton_func, data_func, skeleton_kwargs)

    Returns:
        List of results from data functions
    """
    results = []
    placeholders = []

    # Create all placeholders and show skeletons
    for label, skeleton_func, data_func, skeleton_kwargs in sections:
        placeholder = st.empty()
        placeholders.append(placeholder)

        with placeholder.container():
            if label:
                st.subheader(label)
            skeleton_func(**skeleton_kwargs)

    # Load data progressively
    for i, (label, skeleton_func, data_func, skeleton_kwargs) in enumerate(sections):
        # Load data
        result = data_func()
        results.append(result)

        # Replace skeleton with real content
        # (Note: Actual rendering must be done by caller)
        placeholders[i].empty()

    return results


# Pre-configured skeleton presets for common use cases
class SkeletonPresets:
    """Common skeleton configurations for typical scenarios."""

    @staticmethod
    def positions_table():
        """Skeleton for positions/trades table."""
        skeleton_dataframe(rows=8, cols=8, height=400)

    @staticmethod
    def portfolio_metrics():
        """Skeleton for portfolio metrics row."""
        skeleton_metric_row(num_metrics=4)

    @staticmethod
    def options_chain():
        """Skeleton for options chain table."""
        skeleton_dataframe(rows=15, cols=10, height=600)

    @staticmethod
    def performance_chart():
        """Skeleton for performance chart."""
        skeleton_chart(height=400)

    @staticmethod
    def market_overview():
        """Skeleton for market overview section."""
        skeleton_metric_row(num_metrics=3)
        st.markdown("<br>", unsafe_allow_html=True)
        skeleton_chart(height=300)
        st.markdown("<br>", unsafe_allow_html=True)
        skeleton_dataframe(rows=5, cols=4)

    @staticmethod
    def trade_entry_form():
        """Skeleton for trade entry form."""
        skeleton_card(title=True, content_lines=5, height=250)


# Convenience exports
__all__ = [
    'skeleton_dataframe',
    'skeleton_metric_row',
    'skeleton_chart',
    'skeleton_card',
    'skeleton_table',
    'skeleton_sidebar_filters',
    'skeleton_tabs',
    'with_skeleton',
    'SkeletonContext',
    'progressive_skeleton_load',
    'SkeletonPresets',
]
