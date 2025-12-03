"""
Sector ETF Manager
Manages sector ETF data, holdings, and performance metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SectorETFManager:
    """
    Manages sector ETF data for the 11 GICS sectors
    Provides holdings, performance metrics, and sector rotation data
    """

    def __init__(self) -> None:
        """Initialize the Sector ETF Manager"""

        # 11 Select Sector SPDR ETFs
        self.sector_etfs = {
            'XLK': {
                'name': 'Technology Select Sector SPDR Fund',
                'sector': 'Information Technology',
                'expense_ratio': 0.0010,
                'description': 'Software, hardware, semiconductors, and tech services',
                'major_holdings': [
                    {'symbol': 'NVDA', 'name': 'NVIDIA', 'weight': 14.91},
                    {'symbol': 'AAPL', 'name': 'Apple', 'weight': 13.37},
                    {'symbol': 'MSFT', 'name': 'Microsoft', 'weight': 12.03},
                    {'symbol': 'AVGO', 'name': 'Broadcom', 'weight': 5.52},
                    {'symbol': 'PLTR', 'name': 'Palantir', 'weight': 3.34}
                ]
            },
            'XLV': {
                'name': 'Health Care Select Sector SPDR Fund',
                'sector': 'Health Care',
                'expense_ratio': 0.0010,
                'description': 'Pharmaceuticals, biotech, medical devices, healthcare services',
                'major_holdings': [
                    {'symbol': 'UNH', 'name': 'UnitedHealth Group', 'weight': 10.5},
                    {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'weight': 7.8},
                    {'symbol': 'LLY', 'name': 'Eli Lilly', 'weight': 7.2},
                    {'symbol': 'ABBV', 'name': 'AbbVie', 'weight': 5.1},
                    {'symbol': 'PFE', 'name': 'Pfizer', 'weight': 3.9}
                ]
            },
            'XLF': {
                'name': 'Financial Select Sector SPDR Fund',
                'sector': 'Financials',
                'expense_ratio': 0.0010,
                'description': 'Banks, insurance, investment firms, REITs',
                'major_holdings': [
                    {'symbol': 'BRK.B', 'name': 'Berkshire Hathaway', 'weight': 12.8},
                    {'symbol': 'JPM', 'name': 'JPMorgan Chase', 'weight': 9.7},
                    {'symbol': 'V', 'name': 'Visa', 'weight': 7.2},
                    {'symbol': 'MA', 'name': 'Mastercard', 'weight': 6.5},
                    {'symbol': 'BAC', 'name': 'Bank of America', 'weight': 4.9}
                ]
            },
            'XLC': {
                'name': 'Communication Services Select Sector SPDR Fund',
                'sector': 'Communication Services',
                'expense_ratio': 0.0010,
                'description': 'Media, entertainment, interactive services, telecom',
                'major_holdings': [
                    {'symbol': 'META', 'name': 'Meta Platforms', 'weight': 22.5},
                    {'symbol': 'GOOGL', 'name': 'Alphabet Class A', 'weight': 12.3},
                    {'symbol': 'GOOG', 'name': 'Alphabet Class C', 'weight': 10.8},
                    {'symbol': 'NFLX', 'name': 'Netflix', 'weight': 5.9},
                    {'symbol': 'DIS', 'name': 'Disney', 'weight': 4.2}
                ]
            },
            'XLY': {
                'name': 'Consumer Discretionary Select Sector SPDR Fund',
                'sector': 'Consumer Discretionary',
                'expense_ratio': 0.0010,
                'description': 'Retail, automotive, leisure, hotels, restaurants',
                'major_holdings': [
                    {'symbol': 'AMZN', 'name': 'Amazon', 'weight': 23.1},
                    {'symbol': 'TSLA', 'name': 'Tesla', 'weight': 14.7},
                    {'symbol': 'HD', 'name': 'Home Depot', 'weight': 8.2},
                    {'symbol': 'MCD', 'name': "McDonald's", 'weight': 4.3},
                    {'symbol': 'NKE', 'name': 'Nike', 'weight': 3.1}
                ]
            },
            'XLI': {
                'name': 'Industrial Select Sector SPDR Fund',
                'sector': 'Industrials',
                'expense_ratio': 0.0010,
                'description': 'Aerospace, defense, construction, machinery, transportation',
                'major_holdings': [
                    {'symbol': 'GE', 'name': 'General Electric', 'weight': 6.8},
                    {'symbol': 'CAT', 'name': 'Caterpillar', 'weight': 5.9},
                    {'symbol': 'RTX', 'name': 'RTX Corporation', 'weight': 4.7},
                    {'symbol': 'UNP', 'name': 'Union Pacific', 'weight': 4.3},
                    {'symbol': 'HON', 'name': 'Honeywell', 'weight': 4.1}
                ]
            },
            'XLP': {
                'name': 'Consumer Staples Select Sector SPDR Fund',
                'sector': 'Consumer Staples',
                'expense_ratio': 0.0010,
                'description': 'Food, beverage, household products, personal care',
                'major_holdings': [
                    {'symbol': 'PG', 'name': 'Procter & Gamble', 'weight': 14.2},
                    {'symbol': 'COST', 'name': 'Costco', 'weight': 12.1},
                    {'symbol': 'WMT', 'name': 'Walmart', 'weight': 11.8},
                    {'symbol': 'KO', 'name': 'Coca-Cola', 'weight': 9.3},
                    {'symbol': 'PEP', 'name': 'PepsiCo', 'weight': 8.7}
                ]
            },
            'XLE': {
                'name': 'Energy Select Sector SPDR Fund',
                'sector': 'Energy',
                'expense_ratio': 0.0010,
                'description': 'Oil, gas, coal, consumable fuels, energy equipment',
                'major_holdings': [
                    {'symbol': 'XOM', 'name': 'Exxon Mobil', 'weight': 23.81},
                    {'symbol': 'CVX', 'name': 'Chevron', 'weight': 17.64},
                    {'symbol': 'COP', 'name': 'ConocoPhillips', 'weight': 6.66},
                    {'symbol': 'WMB', 'name': 'Williams Companies', 'weight': 4.49},
                    {'symbol': 'MPC', 'name': 'Marathon Petroleum', 'weight': 4.04}
                ]
            },
            'XLB': {
                'name': 'Materials Select Sector SPDR Fund',
                'sector': 'Materials',
                'expense_ratio': 0.0010,
                'description': 'Chemicals, metals, mining, construction materials',
                'major_holdings': [
                    {'symbol': 'LIN', 'name': 'Linde', 'weight': 20.5},
                    {'symbol': 'APD', 'name': 'Air Products', 'weight': 8.3},
                    {'symbol': 'SHW', 'name': 'Sherwin-Williams', 'weight': 6.9},
                    {'symbol': 'ECL', 'name': 'Ecolab', 'weight': 5.7},
                    {'symbol': 'FCX', 'name': 'Freeport-McMoRan', 'weight': 4.2}
                ]
            },
            'XLRE': {
                'name': 'Real Estate Select Sector SPDR Fund',
                'sector': 'Real Estate',
                'expense_ratio': 0.0010,
                'description': 'REITs, real estate management and development',
                'major_holdings': [
                    {'symbol': 'PLD', 'name': 'Prologis', 'weight': 13.2},
                    {'symbol': 'AMT', 'name': 'American Tower', 'weight': 11.8},
                    {'symbol': 'EQIX', 'name': 'Equinix', 'weight': 7.9},
                    {'symbol': 'CCI', 'name': 'Crown Castle', 'weight': 5.4},
                    {'symbol': 'SPG', 'name': 'Simon Property Group', 'weight': 4.8}
                ]
            },
            'XLU': {
                'name': 'Utilities Select Sector SPDR Fund',
                'sector': 'Utilities',
                'expense_ratio': 0.0010,
                'description': 'Electric, gas, water utilities, renewable energy',
                'major_holdings': [
                    {'symbol': 'NEE', 'name': 'NextEra Energy', 'weight': 14.7},
                    {'symbol': 'SO', 'name': 'Southern Company', 'weight': 6.8},
                    {'symbol': 'DUK', 'name': 'Duke Energy', 'weight': 6.5},
                    {'symbol': 'CEG', 'name': 'Constellation Energy', 'weight': 5.9},
                    {'symbol': 'AEP', 'name': 'American Electric Power', 'weight': 4.3}
                ]
            }
        }

    def get_all_etfs(self) -> List[str]:
        """Get list of all sector ETF tickers"""
        return list(self.sector_etfs.keys())

    def get_etf_info(self, ticker: str) -> Optional[Dict]:
        """Get detailed info for a sector ETF"""
        return self.sector_etfs.get(ticker)

    def get_etf_by_sector(self, sector: str) -> Optional[Tuple[str, Dict]]:
        """
        Get ETF ticker and info for a given sector

        Args:
            sector: GICS sector name

        Returns:
            Tuple of (ticker, info_dict) or None
        """
        for ticker, info in self.sector_etfs.items():
            if info['sector'] == sector:
                return ticker, info
        return None

    def fetch_etf_price_data(
        self,
        ticker: str,
        period: str = '1y'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for sector ETF

        Args:
            ticker: ETF ticker (e.g., 'XLK')
            period: Period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            etf = yf.Ticker(ticker)
            data = etf.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching {ticker} data: {e}")
            return None

    def calculate_etf_returns(
        self,
        ticker: str,
        periods: List[int] = [21, 63, 126, 252]
    ) -> Dict[str, float]:
        """
        Calculate returns for multiple periods

        Args:
            ticker: ETF ticker
            periods: List of periods in trading days (21=1M, 63=3M, 126=6M, 252=1Y)

        Returns:
            Dictionary with returns for each period
        """
        try:
            data = self.fetch_etf_price_data(ticker, period='2y')
            if data is None or len(data) < max(periods):
                return {}

            returns = {}
            current_price = data['Close'].iloc[-1]

            for period in periods:
                if len(data) > period:
                    past_price = data['Close'].iloc[-period-1]
                    period_return = ((current_price - past_price) / past_price) * 100

                    # Map period to label
                    period_labels = {
                        21: '1M',
                        63: '3M',
                        126: '6M',
                        252: '1Y'
                    }
                    label = period_labels.get(period, f'{period}D')
                    returns[label] = round(period_return, 2)

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns for {ticker}: {e}")
            return {}

    def get_etf_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for sector ETF"""
        try:
            etf = yf.Ticker(ticker)
            data = etf.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting price for {ticker}: {e}")
            return None

    def get_all_etf_performance(self) -> pd.DataFrame:
        """
        Get performance metrics for all sector ETFs

        Returns:
            DataFrame with sector performance data
        """
        performance_data = []

        for ticker, info in self.sector_etfs.items():
            try:
                # Fetch data
                current_price = self.get_etf_current_price(ticker)
                returns = self.calculate_etf_returns(ticker)

                performance_data.append({
                    'Ticker': ticker,
                    'Sector': info['sector'],
                    'Name': info['name'],
                    'Current Price': current_price,
                    '1M Return': returns.get('1M', 0.0),
                    '3M Return': returns.get('3M', 0.0),
                    '6M Return': returns.get('6M', 0.0),
                    '1Y Return': returns.get('1Y', 0.0),
                    'Expense Ratio': info['expense_ratio']
                })

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        df = pd.DataFrame(performance_data)
        return df

    def get_top_holdings_df(self, ticker: str) -> pd.DataFrame:
        """
        Get top holdings as DataFrame

        Args:
            ticker: ETF ticker

        Returns:
            DataFrame with holdings data
        """
        info = self.get_etf_info(ticker)
        if not info or 'major_holdings' not in info:
            return pd.DataFrame()

        df = pd.DataFrame(info['major_holdings'])
        df['etf_ticker'] = ticker
        df['sector'] = info['sector']

        return df

    def get_all_holdings(self) -> pd.DataFrame:
        """
        Get holdings for all sector ETFs

        Returns:
            DataFrame with all holdings
        """
        all_holdings = []

        for ticker in self.sector_etfs.keys():
            holdings_df = self.get_top_holdings_df(ticker)
            if not holdings_df.empty:
                all_holdings.append(holdings_df)

        if all_holdings:
            return pd.concat(all_holdings, ignore_index=True)
        else:
            return pd.DataFrame()

    def calculate_sector_momentum_rank(self) -> pd.DataFrame:
        """
        Calculate momentum ranking for all sectors

        Returns:
            DataFrame with momentum scores and rankings
        """
        performance_df = self.get_all_etf_performance()

        if performance_df.empty:
            return pd.DataFrame()

        # Calculate momentum score (weighted average of returns)
        performance_df['Momentum Score'] = (
            (performance_df['1M Return'] * 0.5) +
            (performance_df['3M Return'] * 0.3) +
            (performance_df['6M Return'] * 0.2)
        )

        # Rank by momentum
        performance_df['Momentum Rank'] = performance_df['Momentum Score'].rank(
            ascending=False,
            method='dense'
        ).astype(int)

        # Sort by rank
        performance_df = performance_df.sort_values('Momentum Rank')

        return performance_df

    def get_sector_rotation_signals(self) -> List[Dict]:
        """
        Generate sector rotation buy/sell signals based on momentum

        Returns:
            List of signal dictionaries
        """
        momentum_df = self.calculate_sector_momentum_rank()

        if momentum_df.empty:
            return []

        signals = []
        total_sectors = len(momentum_df)

        for _, row in momentum_df.iterrows():
            rank = row['Momentum Rank']
            momentum = row['Momentum Score']

            # Top 3 sectors: BUY
            if rank <= 3:
                signal = 'BUY'
                strength = 'Strong'
            # Ranks 4-6: HOLD
            elif rank <= 6:
                signal = 'HOLD'
                strength = 'Moderate'
            # Bottom 3 sectors: SELL/AVOID
            elif rank > total_sectors - 3:
                signal = 'SELL'
                strength = 'Weak'
            # Middle: NEUTRAL
            else:
                signal = 'NEUTRAL'
                strength = 'Neutral'

            signals.append({
                'Ticker': row['Ticker'],
                'Sector': row['Sector'],
                'Signal': signal,
                'Strength': strength,
                'Momentum Score': round(momentum, 2),
                'Rank': rank,
                '1M Return': row['1M Return'],
                '3M Return': row['3M Return'],
                '6M Return': row['6M Return']
            })

        return signals

    def export_to_database(self, db_manager) -> bool:
        """
        Export sector ETF data to database

        Args:
            db_manager: Database manager instance

        Returns:
            Success boolean
        """
        try:
            conn = db_manager.get_connection()
            cur = conn.cursor()

            # Insert sector ETFs
            for ticker, info in self.sector_etfs.items():
                cur.execute("""
                    INSERT INTO sector_etfs (
                        etf_symbol, etf_name, sector, expense_ratio, description
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (etf_symbol)
                    DO UPDATE SET
                        etf_name = EXCLUDED.etf_name,
                        expense_ratio = EXCLUDED.expense_ratio,
                        description = EXCLUDED.description
                """, (
                    ticker,
                    info['name'],
                    info['sector'],
                    info['expense_ratio'],
                    info['description']
                ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info("Successfully exported sector ETF data to database")
            return True

        except Exception as e:
            logger.error(f"Error exporting to database: {e}")
            return False
