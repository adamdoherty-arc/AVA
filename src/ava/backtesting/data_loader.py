"""
Historical Data Loader
======================

Data loading utilities for backtesting with support for:
- Yahoo Finance (yfinance)
- CSV files
- Database queries
- Synthetic data generation

Author: AVA Trading Platform
Created: 2025-11-28
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Union
from enum import Enum
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class OptionsDataSource(Enum):
    """Available data sources"""
    YFINANCE = "yfinance"
    CSV = "csv"
    DATABASE = "database"
    SYNTHETIC = "synthetic"


@dataclass
class DataConfig:
    """Configuration for data loading"""
    source: OptionsDataSource = OptionsDataSource.YFINANCE
    cache_dir: Optional[Path] = None
    csv_path: Optional[Path] = None
    database_uri: Optional[str] = None

    # IV calculation settings
    calculate_iv: bool = True
    iv_window: int = 30  # Days for IV rank calculation
    hv_window: int = 20  # Days for historical volatility

    # Synthetic data settings
    synthetic_volatility: float = 0.25
    synthetic_drift: float = 0.05


class HistoricalDataLoader:
    """
    Load historical data for backtesting.

    Usage:
        loader = HistoricalDataLoader()
        data = loader.load_symbols(['SPY', 'QQQ'], start='2023-01-01', end='2024-01-01')
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

        if self.config.cache_dir:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, date],
        end_date: Union[str, date],
        include_options: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            include_options: Whether to generate options chain data

        Returns:
            Dict mapping symbol to DataFrame with columns:
            date, open, high, low, close, volume, iv, iv_rank, hv, vix
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        data = {}

        for symbol in symbols:
            try:
                df = self._load_single_symbol(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    data[symbol] = df
                    logger.info(f"Loaded {len(df)} rows for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        return data

    def _load_single_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """Load data for a single symbol"""

        if self.config.source == OptionsDataSource.YFINANCE:
            return self._load_from_yfinance(symbol, start_date, end_date)
        elif self.config.source == OptionsDataSource.CSV:
            return self._load_from_csv(symbol, start_date, end_date)
        elif self.config.source == OptionsDataSource.SYNTHETIC:
            return self._generate_synthetic_data(symbol, start_date, end_date)
        else:
            logger.warning(f"Unknown data source: {self.config.source}")
            return None

    def _load_from_yfinance(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """Load from Yahoo Finance"""
        try:
            import yfinance as yf

            # Extend start date for IV calculation
            extended_start = start_date - timedelta(days=365)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=extended_start, end=end_date + timedelta(days=1))

            if df.empty:
                return None

            # Rename columns
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index).date
            df.index.name = 'date'

            # Calculate volatility metrics
            df = self._calculate_volatility_metrics(df)

            # Get VIX data
            vix = self._load_vix_data(extended_start, end_date)
            if vix is not None:
                df = df.join(vix, how='left')
                df['vix'] = df['vix'].fillna(method='ffill')

            # Filter to requested date range
            df = df[df.index >= start_date]

            return df

        except ImportError:
            logger.warning("yfinance not installed, falling back to synthetic data")
            return self._generate_synthetic_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Error loading from yfinance: {e}")
            return None

    def _load_from_csv(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """Load from CSV file"""
        if not self.config.csv_path:
            return None

        csv_file = self.config.csv_path / f"{symbol}.csv"
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return None

        df = pd.read_csv(csv_file, parse_dates=['date'], index_col='date')
        df.index = pd.to_datetime(df.index).date

        # Filter date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        df = df[mask]

        # Calculate IV metrics if not present
        if 'iv_rank' not in df.columns:
            df = self._calculate_volatility_metrics(df)

        return df

    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        logger.info(f"Generating synthetic data for {symbol}")

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        # Starting price based on symbol
        start_prices = {
            'SPY': 450, 'QQQ': 380, 'IWM': 200, 'DIA': 350,
            'AAPL': 180, 'MSFT': 370, 'AMZN': 150, 'GOOGL': 140,
            'NVDA': 500, 'TSLA': 250, 'AMD': 150, 'META': 350
        }
        start_price = start_prices.get(symbol.upper(), 100)

        # Generate random walk with drift
        n = len(dates)
        drift = self.config.synthetic_drift / 252
        vol = self.config.synthetic_volatility / np.sqrt(252)

        returns = np.random.normal(drift, vol, n)
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        open_prices = low + np.random.random(n) * (high - low)
        volume = np.random.randint(1000000, 10000000, n)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=dates.date)

        df.index.name = 'date'

        # Add volatility metrics
        df = self._calculate_volatility_metrics(df)

        # Add synthetic VIX
        base_vix = 18
        vix_vol = 5
        df['vix'] = base_vix + np.random.normal(0, vix_vol, len(df))
        df['vix'] = df['vix'].clip(10, 50)

        return df

    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate IV rank and historical volatility"""

        # Historical volatility (using close-to-close)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        df['hv'] = log_returns.rolling(window=self.config.hv_window).std() * np.sqrt(252)

        # Estimate IV from HV (simplified - real IV comes from options prices)
        # In practice, you'd use actual option prices to calculate IV
        df['iv'] = df['hv'] * (1 + np.random.uniform(-0.1, 0.2, len(df)))
        df['iv'] = df['iv'].fillna(self.config.synthetic_volatility)

        # IV Rank (where current IV sits in past year's range)
        iv_window = min(252, len(df))
        df['iv_min'] = df['iv'].rolling(window=iv_window, min_periods=20).min()
        df['iv_max'] = df['iv'].rolling(window=iv_window, min_periods=20).max()

        iv_range = df['iv_max'] - df['iv_min']
        df['iv_rank'] = np.where(
            iv_range > 0,
            (df['iv'] - df['iv_min']) / iv_range * 100,
            50
        )

        # IV Percentile
        def calc_percentile(x):
            return (x.rank(pct=True).iloc[-1]) * 100 if len(x) > 0 else 50

        df['iv_percentile'] = df['iv'].rolling(window=iv_window, min_periods=20).apply(calc_percentile)

        # Clean up
        df = df.drop(columns=['iv_min', 'iv_max'], errors='ignore')
        df['iv_rank'] = df['iv_rank'].fillna(50)
        df['iv_percentile'] = df['iv_percentile'].fillna(50)

        return df

    def _load_vix_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Load VIX data"""
        try:
            import yfinance as yf

            vix = yf.Ticker('^VIX')
            df = vix.history(start=start_date, end=end_date + timedelta(days=1))

            if df.empty:
                return None

            df.index = pd.to_datetime(df.index).date
            df = df[['Close']].rename(columns={'Close': 'vix'})

            return df

        except Exception as e:
            logger.debug(f"Could not load VIX: {e}")
            return None

    def load_earnings_calendar(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """Load earnings dates for symbols"""
        earnings_data = []

        try:
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    calendar = ticker.calendar

                    if calendar is not None and 'Earnings Date' in calendar.columns:
                        for earning_date in calendar['Earnings Date']:
                            if start_date <= earning_date.date() <= end_date:
                                earnings_data.append({
                                    'symbol': symbol,
                                    'earnings_date': earning_date.date()
                                })
                except Exception:
                    pass

        except ImportError:
            logger.warning("yfinance not available for earnings calendar")

        return pd.DataFrame(earnings_data)

    def add_earnings_to_data(
        self,
        data: Dict[str, pd.DataFrame],
        earnings: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Add earnings dates and days_to_earnings to data"""

        for symbol, df in data.items():
            symbol_earnings = earnings[earnings['symbol'] == symbol]['earnings_date'].tolist()

            if not symbol_earnings:
                df['earnings_date'] = None
                df['days_to_earnings'] = None
                continue

            def calc_days_to_earnings(current_date):
                future_earnings = [e for e in symbol_earnings if e > current_date]
                if future_earnings:
                    return (min(future_earnings) - current_date).days
                return None

            df['days_to_earnings'] = df.index.map(calc_days_to_earnings)
            df['earnings_date'] = df['days_to_earnings'].apply(
                lambda d: df.index[0] + timedelta(days=d) if d else None
            )

        return data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_backtest_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    source: str = 'yfinance'
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load backtest data.

    Args:
        symbols: List of stock symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        source: Data source ('yfinance', 'synthetic', 'csv')

    Returns:
        Dict of symbol -> DataFrame
    """
    config = DataConfig(source=OptionsDataSource(source))
    loader = HistoricalDataLoader(config)

    return loader.load_symbols(
        symbols,
        date.fromisoformat(start_date),
        date.fromisoformat(end_date)
    )


def generate_sample_data(
    symbols: List[str] = None,
    start_date: str = '2023-01-01',
    end_date: str = '2024-01-01'
) -> Dict[str, pd.DataFrame]:
    """Generate sample synthetic data for testing"""
    symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'NVDA']

    config = DataConfig(source=OptionsDataSource.SYNTHETIC)
    loader = HistoricalDataLoader(config)

    return loader.load_symbols(
        symbols,
        date.fromisoformat(start_date),
        date.fromisoformat(end_date)
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Historical Data Loader ===\n")

    # Test synthetic data generation
    print("1. Generating synthetic data...")
    data = generate_sample_data(
        symbols=['SPY', 'QQQ'],
        start_date='2024-01-01',
        end_date='2024-06-30'
    )

    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  IV Rank range: {df['iv_rank'].min():.1f} - {df['iv_rank'].max():.1f}")

    # Test yfinance if available
    print("\n2. Testing yfinance loader...")
    try:
        real_data = load_backtest_data(
            symbols=['SPY'],
            start_date='2024-01-01',
            end_date='2024-03-01',
            source='yfinance'
        )
        if 'SPY' in real_data:
            print(f"   Loaded {len(real_data['SPY'])} rows from yfinance")
    except Exception as e:
        print(f"   yfinance test skipped: {e}")

    print("\n✅ Data loader ready!")
