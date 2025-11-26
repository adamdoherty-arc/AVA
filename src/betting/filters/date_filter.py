"""
Date Range Filter
Filter for game date ranges with preset options
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from src.betting.filters.base_filter import BaseBettingFilter


class DateRangeFilter(BaseBettingFilter):
    """
    Filter for date ranges with preset options.

    Used to filter games/events by date range.
    Consolidates date filtering from sports betting pages.
    """

    def __init__(self,
                 label: str = "Date Range",
                 presets: list = None,
                 help_text: str = None):
        """
        Initialize date range filter.

        Args:
            label: Display label
            presets: List of preset options (default: ["All", "Today", "Tomorrow", "This Week", "Custom"])
            help_text: Optional help text
        """
        super().__init__(label)
        self.presets = presets or ["All", "Today", "Tomorrow", "This Week", "Custom"]
        self.help_text = help_text or "Filter by game date range"

    def render(self, key_prefix: str) -> dict:
        """
        Render date range filter.

        Args:
            key_prefix: Unique key prefix

        Returns:
            Dictionary with 'preset' and optional 'start_date'/'end_date'
        """
        preset = st.selectbox(
            self.label,
            options=self.presets,
            key=f"{key_prefix}_date_preset",
            help=self.help_text
        )

        result = {"preset": preset}

        # Show custom date pickers if "Custom" is selected
        if preset == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date(),
                    key=f"{key_prefix}_start_date"
                )
                result["start_date"] = start_date
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=(datetime.now() + timedelta(days=7)).date(),
                    key=f"{key_prefix}_end_date"
                )
                result["end_date"] = end_date

        return result

    def apply(self, df: pd.DataFrame, value: dict) -> pd.DataFrame:
        """
        Apply date range filter.

        Args:
            df: Input DataFrame
            value: Dictionary with preset and optional date range

        Returns:
            Filtered DataFrame
        """
        preset = value.get("preset", "All")

        if preset == "All":
            return df

        # Try multiple common column names for date
        date_cols = ['game_date', 'date', 'event_date', 'game_time',
                     'start_time', 'datetime', 'timestamp']

        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return df  # No date column found

        # Convert to datetime if needed
        try:
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
                df_copy[date_col] = pd.to_datetime(df_copy[date_col])

            now = pd.Timestamp.now()
            today = now.normalize()

            if preset == "Today":
                start = today
                end = today + pd.Timedelta(days=1)
            elif preset == "Tomorrow":
                start = today + pd.Timedelta(days=1)
                end = today + pd.Timedelta(days=2)
            elif preset == "This Week":
                start = today
                end = today + pd.Timedelta(days=7)
            elif preset == "This Month":
                start = today
                end = today + pd.Timedelta(days=30)
            elif preset == "Custom":
                start = pd.Timestamp(value.get("start_date", today.date()))
                end = pd.Timestamp(value.get("end_date", (today + pd.Timedelta(days=7)).date()))
                end = end + pd.Timedelta(days=1)  # Include end date
            else:
                return df_copy

            return df_copy[(df_copy[date_col] >= start) & (df_copy[date_col] < end)]

        except Exception:
            # If date conversion fails, return unfiltered
            return df
