"""
Betting Filter Panel
Composite component that combines multiple filters
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.betting.filters.base_filter import BaseBettingFilter


class BettingFilterPanel:
    """
    Composite filter panel that combines multiple filters.

    Provides a unified interface for rendering and applying multiple filters
    to betting data. Simplifies filter management in sports betting pages.
    """

    def __init__(self, filters: List[BaseBettingFilter] = None):
        """
        Initialize filter panel.

        Args:
            filters: List of filter instances to include in the panel
        """
        self.filters = filters or []

    def add_filter(self, filter_instance: BaseBettingFilter):
        """
        Add a filter to the panel.

        Args:
            filter_instance: Filter instance to add
        """
        self.filters.append(filter_instance)

    def render(self, key_prefix: str, layout: str = "columns") -> Dict[str, Any]:
        """
        Render all filters in the panel.

        Args:
            key_prefix: Unique key prefix for widgets
            layout: Layout style ("columns", "rows", or "sidebar")
                   - "columns": Arrange filters in columns (2-3 per row)
                   - "rows": Stack filters vertically
                   - "sidebar": Render in Streamlit sidebar

        Returns:
            Dictionary mapping filter names to their selected values
        """
        if not self.filters:
            return {}

        values = {}

        if layout == "sidebar":
            # Render in sidebar
            for filter_instance in self.filters:
                filter_name = filter_instance.get_filter_name()
                values[filter_name] = filter_instance.render(key_prefix)

        elif layout == "columns":
            # Render in columns (2-3 filters per row)
            num_filters = len(self.filters)
            cols_per_row = min(3, num_filters)

            for i in range(0, num_filters, cols_per_row):
                batch = self.filters[i:i + cols_per_row]
                cols = st.columns(len(batch))

                for col, filter_instance in zip(cols, batch):
                    with col:
                        filter_name = filter_instance.get_filter_name()
                        values[filter_name] = filter_instance.render(key_prefix)

        else:  # layout == "rows"
            # Render in rows (stacked vertically)
            for filter_instance in self.filters:
                filter_name = filter_instance.get_filter_name()
                values[filter_name] = filter_instance.render(key_prefix)

        return values

    def apply(self, df: pd.DataFrame, filter_values: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply all filters to DataFrame.

        Args:
            df: Input DataFrame
            filter_values: Dictionary of filter values from render()

        Returns:
            Filtered DataFrame
        """
        result = df.copy()

        for filter_instance in self.filters:
            filter_name = filter_instance.get_filter_name()
            if filter_name in filter_values:
                value = filter_values[filter_name]
                result = filter_instance.apply(result, value)

        return result

    def render_and_apply(self, df: pd.DataFrame, key_prefix: str,
                        layout: str = "columns") -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convenience method to render filters and apply them in one call.

        Args:
            df: Input DataFrame
            key_prefix: Unique key prefix for widgets
            layout: Layout style ("columns", "rows", or "sidebar")

        Returns:
            Tuple of (filtered_df, filter_values)
        """
        filter_values = self.render(key_prefix, layout)
        filtered_df = self.apply(df, filter_values)
        return filtered_df, filter_values

    def get_filter_count(self) -> int:
        """Get number of filters in the panel."""
        return len(self.filters)

    def clear_filters(self) -> None:
        """Remove all filters from the panel."""
        self.filters = []


def create_standard_betting_panel() -> BettingFilterPanel:
    """
    Create a standard betting filter panel with common filters.

    Returns:
        BettingFilterPanel with confidence, EV, date, status, sport, and sort filters
    """
    from src.betting.filters.confidence_filter import ConfidenceFilter
    from src.betting.filters.ev_filter import ExpectedValueFilter
    from src.betting.filters.date_filter import DateRangeFilter
    from src.betting.filters.status_filter import GameStatusFilter
    from src.betting.filters.sport_filter import SportFilter
    from src.betting.filters.sort_filter import SortFilter

    panel = BettingFilterPanel()
    panel.add_filter(ConfidenceFilter())
    panel.add_filter(ExpectedValueFilter())
    panel.add_filter(DateRangeFilter())
    panel.add_filter(GameStatusFilter())
    panel.add_filter(SportFilter())
    panel.add_filter(SortFilter())

    return panel


def create_minimal_betting_panel() -> BettingFilterPanel:
    """
    Create a minimal betting filter panel with essential filters only.

    Returns:
        BettingFilterPanel with confidence and EV filters
    """
    from src.betting.filters.confidence_filter import ConfidenceFilter
    from src.betting.filters.ev_filter import ExpectedValueFilter

    panel = BettingFilterPanel()
    panel.add_filter(ConfidenceFilter())
    panel.add_filter(ExpectedValueFilter())

    return panel
