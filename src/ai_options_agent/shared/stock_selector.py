"""
Shared Stock Selector Component
Provides 3 modes: Manual Input, TradingView Watchlist, Database Stocks
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from src.tradingview_db_manager import TradingViewDBManager


class StockSelector:
    """Reusable stock selection component"""

    def __init__(self) -> None:
        self.tv_manager = TradingViewDBManager()

    def render_single_stock_selector(
        self,
        modes: List[str] = None,
        default_symbol: str = "AAPL",
        show_quick_info: bool = True
    ) -> Optional[str]:
        """
        Render stock selector with multiple input modes

        Args:
            modes: List of modes to show. Options:
                   ["manual", "tradingview", "database"]
                   Default: all three
            default_symbol: Default symbol for manual input
            show_quick_info: Show quick info panel

        Returns:
            Selected stock symbol or None
        """
        if modes is None:
            modes = ["manual", "tradingview", "database"]

        # Build mode options
        mode_options = []
        if "manual" in modes:
            mode_options.append("✏️ Manual Input")
        if "tradingview" in modes:
            mode_options.append("📺 TradingView Watchlist")
        if "database" in modes:
            mode_options.append("💾 Database Stocks")

        # Data source selector
        data_source = st.radio(
            "Select Data Source",
            mode_options,
            horizontal=True,
            help="Choose where to select your stock from"
        )

        # Create columns for selector + info
        if show_quick_info:
            col_select, col_info = st.columns([2, 1])
        else:
            col_select = st.container()
            col_info = None

        selected_symbol = None

        with col_select:
            if "TradingView" in data_source:
                selected_symbol = self._render_tradingview_mode()
            elif "Database" in data_source:
                selected_symbol = self._render_database_mode()
            else:
                selected_symbol = self._render_manual_mode(default_symbol)

        # Show quick info if enabled
        if show_quick_info and col_info and selected_symbol:
            with col_info:
                self._render_quick_info(selected_symbol)

        return selected_symbol

    def render_watchlist_selector(self) -> tuple[Optional[str], Optional[List[str]]]:
        """
        Render watchlist selector (for batch analysis)

        Returns:
            (watchlist_name, symbols_list) or (None, None)
        """
        watchlists = self.tv_manager.get_all_symbols()

        if not watchlists:
            st.warning("No watchlists found. Please sync watchlists first.")
            return None, None

        watchlist_name = st.selectbox(
            "Select Watchlist:",
            list(watchlists.keys()),
            format_func=lambda x: f"📁 {x} ({len(watchlists[x])} stocks)",
            help="Choose a TradingView watchlist to analyze"
        )

        symbols = watchlists.get(watchlist_name, [])

        return watchlist_name, symbols

    def _render_tradingview_mode(self) -> Optional[str]:
        """Render TradingView watchlist mode"""
        with st.spinner("Loading watchlists..."):
            watchlists = self.tv_manager.get_all_symbols()

        if not watchlists:
            st.warning("⚠️ No watchlists found. Please sync watchlists from the TradingView Watchlists page.")
            return None

        selected_watchlist = st.selectbox(
            "Select Watchlist",
            list(watchlists.keys()),
            format_func=lambda x: f"📁 {x} ({len(watchlists[x])} stocks)",
            key="watchlist_selector"
        )

        if selected_watchlist:
            symbols = watchlists[selected_watchlist]
            if symbols:
                return st.selectbox(
                    "Select Stock",
                    symbols,
                    key="symbol_selector_tv"
                )
            else:
                st.warning(f"⚠️ Watchlist '{selected_watchlist}' is empty.")

        return None

    def _render_database_mode(self) -> Optional[str]:
        """Render database stocks mode"""
        from .data_fetchers import fetch_database_stocks

        with st.spinner("Loading stocks from database..."):
            stocks = fetch_database_stocks()

        if not stocks:
            st.warning("⚠️ No stocks found in database. Please run database sync first.")
            return None

        return st.selectbox(
            "Select Stock",
            [s['symbol'] for s in stocks],
            format_func=lambda x: f"{x} - {next((s['company_name'] for s in stocks if s['symbol'] == x), x)}",
            key="symbol_selector_db"
        )

    def _render_manual_mode(self, default_symbol: str) -> str:
        """Render manual input mode"""
        return st.text_input(
            "Enter Stock Symbol",
            value=default_symbol,
            help="Type any valid stock ticker symbol",
            key="symbol_manual"
        ).upper()

    def _render_quick_info(self, symbol: str):
        """Render quick info panel for selected symbol"""
        from .data_fetchers import fetch_stock_info

        st.markdown("**📊 Quick Info**")

        with st.spinner("Loading..."):
            info = fetch_stock_info(symbol)

        if info:
            st.metric("Price", f"${info['current_price']:.2f}")
            st.caption(f"**Sector:** {info['sector']}")
            if info.get('market_cap', 0) > 0:
                st.caption(f"**Market Cap:** ${info['market_cap']/1e9:.1f}B")
        else:
            st.warning("Could not fetch stock info")
