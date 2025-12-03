"""
CSP Opportunities Finder - Find next 30-day CSP trades for current positions

For each current CSP position symbol, finds the next optimal 30-day put option
with delta close to 0.30 for rolling into or opening new positions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.tradingview_db_manager import TradingViewDBManager
import logging

logger = logging.getLogger(__name__)


class CSPOpportunitiesFinder:
    """Finds next CSP opportunities for symbols in current positions"""

    def __init__(self) -> None:
        self.tv_manager = TradingViewDBManager()
        self.target_dte = 30
        self.dte_range = (20, 40)  # 20-40 days (wider range)
        self.target_delta = -0.30  # 30 delta puts
        self.delta_range = (-0.45, -0.15)  # 15-45 delta range (wider to catch actual data)

    def find_opportunities_for_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """
        Find next 30-day CSP opportunities for given symbols

        Args:
            symbols: List of stock symbols from current CSP positions

        Returns:
            DataFrame with opportunity details, styled like TradingView watchlist
        """
        if not symbols:
            return pd.DataFrame()

        try:
            conn = self.tv_manager.get_connection()
            cur = conn.cursor()

            # Query for best 30-day put option per symbol
            # Select option with delta closest to -0.30
            # NOTE: Accepts both '30_delta' (old) and '30_dte' (new) strike types
            # Also filters by negative delta to ensure puts only
            query = """
                SELECT DISTINCT ON (sp.symbol)
                    sp.symbol,
                    sd.current_price as stock_price,
                    sp.strike_price,
                    sp.expiration_date,
                    sp.dte,
                    sp.premium,
                    sp.delta,
                    sp.monthly_return,
                    sp.annual_return,
                    sp.implied_volatility as iv,
                    sp.bid,
                    sp.ask,
                    sp.volume,
                    sp.open_interest as oi,
                    (sp.strike_price - (sp.premium / 100)) as breakeven
                FROM stock_premiums sp
                LEFT JOIN stock_data sd ON sp.symbol = sd.symbol
                WHERE sp.symbol = ANY(%s)
                    AND sp.dte BETWEEN %s AND %s
                    AND sp.delta BETWEEN %s AND %s
                    AND sp.delta < 0
                    AND sp.premium > 0
                    AND sp.strike_type IN ('30_delta', '30_dte')
                    AND sp.strike_price < sd.current_price
                ORDER BY sp.symbol, ABS(sp.delta - %s) ASC
            """

            cur.execute(query, (
                symbols,
                self.dte_range[0],
                self.dte_range[1],
                self.delta_range[0],
                self.delta_range[1],
                self.target_delta
            ))

            rows = cur.fetchall()
            cur.close()
            conn.close()

            if not rows:
                logger.warning(f"No 30-day CSP opportunities found for symbols: {symbols}")
                return pd.DataFrame()

            # Create DataFrame with TradingView-style columns
            df = pd.DataFrame(rows, columns=[
                'Symbol', 'Stock Price', 'Strike', 'Expiration', 'DTE',
                'Premium', 'Delta', 'Monthly %', 'Annual %', 'IV',
                'Bid', 'Ask', 'Volume', 'OI', 'Breakeven'
            ])

            # Convert to numeric types
            numeric_cols = [
                'Stock Price', 'Strike', 'DTE', 'Premium', 'Delta',
                'Monthly %', 'Annual %', 'IV', 'Bid', 'Ask', 'Volume', 'OI', 'Breakeven'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Calculate additional metrics
            df['Premium %'] = (df['Premium'] / df['Strike']) * 100
            df['Distance %'] = ((df['Stock Price'] - df['Strike']) / df['Stock Price']) * 100

            # Sort by monthly return descending
            df = df.sort_values('Monthly %', ascending=False)

            return df

        except Exception as e:
            logger.error(f"Error finding CSP opportunities: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def find_opportunities_for_current_positions(self, csp_positions: List[Dict]) -> pd.DataFrame:
        """
        Find next CSP opportunities based on current CSP position symbols

        Args:
            csp_positions: List of current CSP positions from Robinhood

        Returns:
            DataFrame with next trade opportunities
        """
        if not csp_positions:
            return pd.DataFrame()

        # Extract unique symbols from current positions
        symbols = list(set([
            pos.get('Symbol', pos.get('symbol', pos.get('symbol_raw', '')))
            for pos in csp_positions
        ]))

        # Remove empty strings
        symbols = [s for s in symbols if s]

        if not symbols:
            return pd.DataFrame()

        return self.find_opportunities_for_symbols(symbols)

    def get_summary_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary metrics for opportunities

        Args:
            df: Opportunities DataFrame

        Returns:
            Dict with summary statistics
        """
        if df.empty:
            return {
                'total_opportunities': 0,
                'avg_premium': 0,
                'avg_monthly_return': 0,
                'avg_annual_return': 0,
                'avg_delta': 0
            }

        return {
            'total_opportunities': len(df),
            'avg_premium': df['Premium'].mean(),
            'avg_monthly_return': df['Monthly %'].mean(),
            'avg_annual_return': df['Annual %'].mean(),
            'avg_delta': df['Delta'].mean(),
            'total_premium': df['Premium'].sum()
        }


# Example usage
if __name__ == "__main__":
    finder = CSPOpportunitiesFinder()

    # Test with sample symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    opportunities = finder.find_opportunities_for_symbols(test_symbols)

    if not opportunities.empty:
        print("\nCSP Opportunities Found:")
        print(opportunities[['Symbol', 'Strike', 'DTE', 'Premium', 'Delta', 'Monthly %']])
        print("\nSummary:")
        print(finder.get_summary_metrics(opportunities))
    else:
        print("No opportunities found. Make sure stock_premiums table has data.")
