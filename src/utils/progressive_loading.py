"""
Progressive Loading Utilities
Load critical data first, then secondary data progressively
"""
import streamlit as st
from typing import Callable, Any, List, Tuple
import time
import logging

logger = logging.getLogger(__name__)

def load_progressively(
    sections: List[Tuple[str, Callable[[], Any], bool]]
):
    """
    Load page sections progressively for better UX.

    Args:
        sections: List of (label, loader_function, is_critical) tuples

    Example:
        load_progressively([
            ("Positions", load_positions, True),   # Load first
            ("Charts", load_charts, False),        # Load after
            ("Stats", load_stats, False)           # Load last
        ])
    """
    # Load critical sections first (immediate display)
    critical_sections = [s for s in sections if s[2]]
    for label, loader, _ in critical_sections:
        with st.spinner(f"Loading {label}..."):
            try:
                result = loader()
                if result is not None:
                    # Loader is responsible for displaying
                    pass
            except Exception as e:
                logger.error(f"Error loading critical section {label}: {e}")
                st.error(f"Error loading {label}")

    # Load non-critical sections progressively
    non_critical = [s for s in sections if not s[2]]
    placeholders = []

    # Create placeholders
    for label, _, _ in non_critical:
        placeholders.append(st.empty())

    # Load data into placeholders
    for i, (label, loader, _) in enumerate(non_critical):
        with placeholders[i]:
            with st.spinner(f"Loading {label}..."):
                try:
                    result = loader()
                    if result is not None:
                        pass  # Loader displays in placeholder
                except Exception as e:
                    logger.error(f"Error loading section {label}: {e}")
                    st.warning(f"Could not load {label}")

class ProgressiveLoader:
    """
    Class-based progressive loader with better control
    """
    def __init__(self) -> None:
        self.sections = []
        self.results = {}

    def add_section(self, name: str, loader: Callable, critical: bool = False):
        """Add a section to load"""
        self.sections.append((name, loader, critical))
        return self

    def load_all(self) -> None:
        """Load all sections progressively"""
        load_progressively(self.sections)
        return self.results

def progressive_load_with_placeholder(
    label: str,
    loader: Callable,
    show_spinner: bool = True
) -> Any:
    """
    Load a single section with a placeholder and optional spinner.

    Args:
        label: Display label for loading indicator
        loader: Function to load the data
        show_spinner: Whether to show loading spinner

    Returns:
        Result from loader function
    """
    placeholder = st.empty()

    with placeholder:
        if show_spinner:
            with st.spinner(f"Loading {label}..."):
                try:
                    result = loader()
                    return result
                except Exception as e:
                    logger.error(f"Error in progressive load {label}: {e}")
                    st.error(f"Failed to load {label}")
                    return None
        else:
            try:
                result = loader()
                return result
            except Exception as e:
                logger.error(f"Error in progressive load {label}: {e}")
                st.error(f"Failed to load {label}")
                return None

def load_with_timeout(
    loader: Callable,
    timeout: float = 5.0,
    fallback: Any = None
) -> Any:
    """
    Load data with a timeout, returning fallback on timeout or error.

    Args:
        loader: Function to load the data
        timeout: Timeout in seconds
        fallback: Value to return on timeout/error

    Returns:
        Result from loader or fallback
    """
    import threading

    result = [fallback]
    exception = [None]

    def target():
        try:
            result[0] = loader()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        logger.warning(f"Loader timed out after {timeout}s")
        return fallback

    if exception[0]:
        logger.error(f"Loader failed: {exception[0]}")
        return fallback

    return result[0]

def progressive_dataframe_load(
    label: str,
    data_loader: Callable,
    page_size: int = 50,
    show_initial_count: int = 10
) -> Any:
    """
    Progressive loading for large DataFrames.
    Show first N rows immediately, then load the rest.

    Args:
        label: Display label
        data_loader: Function that returns a DataFrame
        page_size: Pagination page size
        show_initial_count: Number of rows to show immediately

    Returns:
        Complete DataFrame
    """
    import pandas as pd

    try:
        # Load data
        with st.spinner(f"Loading {label}..."):
            df = data_loader()

        if df is None or len(df) == 0:
            st.info(f"No {label} data available")
            return df

        # Show subset immediately if large
        if len(df) > show_initial_count:
            st.caption(f"Showing first {show_initial_count} of {len(df)} rows...")
            return df

        return df

    except Exception as e:
        logger.error(f"Error loading {label}: {e}")
        st.error(f"Failed to load {label}")
        return pd.DataFrame()
