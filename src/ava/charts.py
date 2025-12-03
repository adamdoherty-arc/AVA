"""
AVA Chart Generator
===================

Generate beautiful charts for portfolio, positions, and stock analysis.

Features:
- Portfolio balance charts (line, area)
- Stock price charts with indicators
- Position P&L charts
- Options Greeks visualization
- Heatmaps for portfolio composition
- Clean, professional styling

Usage:
    from src.ava.charts import ChartGenerator

    gen = ChartGenerator()

    # Generate portfolio chart
    chart_path = gen.generate_portfolio_chart(balances_data)

    # Send via Telegram (use context manager to avoid resource leak!)
    with open(chart_path, 'rb') as photo_file:
        await update.message.reply_photo(photo=photo_file)
"""

import os
import io
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generate charts for AVA Telegram bot.

    Uses matplotlib to create professional-looking charts
    that can be sent as images via Telegram.
    """

    # Chart styling
    FIGURE_SIZE = (12, 6)
    DPI = 100
    BACKGROUND_COLOR = '#1e1e1e'  # Dark theme
    TEXT_COLOR = '#ffffff'
    GRID_COLOR = '#404040'
    POSITIVE_COLOR = '#00ff00'  # Green
    NEGATIVE_COLOR = '#ff0000'  # Red
    PRIMARY_COLOR = '#00bfff'   # Sky blue
    SECONDARY_COLOR = '#ffa500'  # Orange

    def __init__(self) -> None:
        """Initialize chart generator"""
        # Set default matplotlib style
        plt.style.use('dark_background')

    def _setup_figure(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "",
        xlabel: str = "",
        ylabel: str = ""
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Setup a matplotlib figure with consistent styling.

        Args:
            figsize: Figure size (width, height)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize or self.FIGURE_SIZE, dpi=self.DPI)

        # Set colors
        fig.patch.set_facecolor(self.BACKGROUND_COLOR)
        ax.set_facecolor(self.BACKGROUND_COLOR)
        ax.tick_params(colors=self.TEXT_COLOR)
        ax.xaxis.label.set_color(self.TEXT_COLOR)
        ax.yaxis.label.set_color(self.TEXT_COLOR)
        ax.title.set_color(self.TEXT_COLOR)

        # Set labels
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)

        # Grid
        ax.grid(True, alpha=0.3, color=self.GRID_COLOR)

        return fig, ax

    def _save_figure(self, fig: plt.Figure, filename: str = None) -> str:
        """
        Save figure to file and return path.

        Args:
            fig: Matplotlib figure
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ava_chart_{timestamp}.png"

        # Use temp directory
        filepath = os.path.join(os.path.dirname(__file__), "..", "..", "temp", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        fig.tight_layout()
        fig.savefig(filepath, dpi=self.DPI, facecolor=self.BACKGROUND_COLOR, edgecolor='none')
        plt.close(fig)

        return filepath

    def generate_portfolio_chart(
        self,
        balances: List[Dict[str, Any]],
        days: int = 30
    ) -> str:
        """
        Generate portfolio balance chart.

        Args:
            balances: List of balance records with 'timestamp' and 'balance' keys
            days: Number of days to show (default 30)

        Returns:
            Path to saved chart image
        """
        try:
            if not balances:
                raise ValueError("No balance data provided")

            # Extract data
            timestamps = [b['timestamp'] for b in balances]
            values = [float(b['balance']) for b in balances]

            # Filter to last N days
            cutoff = datetime.now() - timedelta(days=days)
            filtered_data = [(t, v) for t, v in zip(timestamps, values) if t >= cutoff]

            if not filtered_data:
                raise ValueError(f"No data in last {days} days")

            timestamps, values = zip(*filtered_data)

            # Create chart
            fig, ax = self._setup_figure(
                title=f"Portfolio Balance - Last {days} Days",
                xlabel="Date",
                ylabel="Balance ($)"
            )

            # Plot line
            ax.plot(timestamps, values, color=self.PRIMARY_COLOR, linewidth=2, label="Balance")

            # Fill area under curve
            ax.fill_between(timestamps, values, alpha=0.3, color=self.PRIMARY_COLOR)

            # Add current value annotation
            current_value = values[-1]
            ax.annotate(
                f'${current_value:,.2f}',
                xy=(timestamps[-1], current_value),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color=self.POSITIVE_COLOR if current_value > values[0] else self.NEGATIVE_COLOR,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
            )

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
            plt.xticks(rotation=45)

            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            # Add legend
            ax.legend(loc='upper left')

            return self._save_figure(fig, f"portfolio_{days}d.png")

        except Exception as e:
            logger.error(f"Error generating portfolio chart: {e}")
            raise

    def generate_stock_chart(
        self,
        ticker: str,
        price_data: List[Dict[str, Any]],
        indicators: Optional[Dict[str, List[float]]] = None
    ) -> str:
        """
        Generate stock price chart with optional indicators.

        Args:
            ticker: Stock ticker symbol
            price_data: List of price records with 'date', 'open', 'high', 'low', 'close', 'volume'
            indicators: Optional dict of indicator data (e.g., {'SMA20': [...], 'SMA50': [...]})

        Returns:
            Path to saved chart image
        """
        try:
            if not price_data:
                raise ValueError("No price data provided")

            # Extract data
            dates = [d['date'] for d in price_data]
            closes = [float(d['close']) for d in price_data]
            volumes = [float(d.get('volume', 0)) for d in price_data]

            # Create figure with two subplots (price and volume)
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=self.FIGURE_SIZE,
                dpi=self.DPI,
                gridspec_kw={'height_ratios': [3, 1]}
            )

            # Style both axes
            for ax in (ax1, ax2):
                ax.set_facecolor(self.BACKGROUND_COLOR)
                ax.tick_params(colors=self.TEXT_COLOR)
                ax.grid(True, alpha=0.3, color=self.GRID_COLOR)

            # Price chart
            ax1.plot(dates, closes, color=self.PRIMARY_COLOR, linewidth=2, label=ticker)
            ax1.set_title(f"{ticker} - Price Chart", fontsize=16, fontweight='bold', color=self.TEXT_COLOR)
            ax1.set_ylabel("Price ($)", fontsize=12, color=self.TEXT_COLOR)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))

            # Add indicators if provided
            if indicators:
                colors = [self.SECONDARY_COLOR, '#ff00ff', '#00ff00']
                for i, (name, values) in enumerate(indicators.items()):
                    if len(values) == len(dates):
                        ax1.plot(dates, values, label=name, linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])

            ax1.legend(loc='upper left')

            # Volume chart
            colors = [self.POSITIVE_COLOR if closes[i] >= closes[i-1] else self.NEGATIVE_COLOR
                     for i in range(len(closes))]
            colors[0] = self.POSITIVE_COLOR  # First bar
            ax2.bar(dates, volumes, color=colors, alpha=0.5)
            ax2.set_xlabel("Date", fontsize=12, color=self.TEXT_COLOR)
            ax2.set_ylabel("Volume", fontsize=12, color=self.TEXT_COLOR)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))

            # Format x-axis
            for ax in (ax1, ax2):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))

            plt.xticks(rotation=45)

            return self._save_figure(fig, f"stock_{ticker}.png")

        except Exception as e:
            logger.error(f"Error generating stock chart: {e}")
            raise

    def generate_position_pnl_chart(
        self,
        positions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate position P&L bar chart.

        Args:
            positions: List of position dictionaries with 'ticker' and 'pnl' keys

        Returns:
            Path to saved chart image
        """
        try:
            if not positions:
                raise ValueError("No positions provided")

            # Extract data
            tickers = [p['ticker'] for p in positions]
            pnls = [float(p.get('pnl', 0)) for p in positions]

            # Sort by P&L
            sorted_data = sorted(zip(tickers, pnls), key=lambda x: x[1], reverse=True)
            tickers, pnls = zip(*sorted_data)

            # Create chart
            fig, ax = self._setup_figure(
                title="Position P&L",
                xlabel="Position",
                ylabel="P&L ($)"
            )

            # Color bars based on positive/negative
            colors = [self.POSITIVE_COLOR if p >= 0 else self.NEGATIVE_COLOR for p in pnls]

            # Create bars
            bars = ax.bar(range(len(tickers)), pnls, color=colors, alpha=0.7)

            # Add value labels on bars
            for i, (bar, pnl) in enumerate(zip(bars, pnls)):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'${pnl:,.0f}',
                    ha='center',
                    va='bottom' if pnl >= 0 else 'top',
                    fontsize=10,
                    color=self.TEXT_COLOR
                )

            # Set x-axis labels
            ax.set_xticks(range(len(tickers)))
            ax.set_xticklabels(tickers, rotation=45, ha='right')

            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            # Add zero line
            ax.axhline(y=0, color=self.TEXT_COLOR, linestyle='-', linewidth=0.5)

            return self._save_figure(fig, "position_pnl.png")

        except Exception as e:
            logger.error(f"Error generating P&L chart: {e}")
            raise

    def generate_portfolio_composition_chart(
        self,
        positions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate portfolio composition pie chart.

        Args:
            positions: List of position dictionaries with 'ticker' and 'market_value' keys

        Returns:
            Path to saved chart image
        """
        try:
            if not positions:
                raise ValueError("No positions provided")

            # Extract data
            tickers = [p['ticker'] for p in positions]
            values = [float(p.get('market_value', 0)) for p in positions]

            # Filter out zero/negative values
            filtered = [(t, v) for t, v in zip(tickers, values) if v > 0]
            if not filtered:
                raise ValueError("No positive position values")

            tickers, values = zip(*filtered)

            # Combine small positions into "Other"
            total_value = sum(values)
            threshold = total_value * 0.05  # 5% threshold
            main_positions = []
            other_value = 0

            for ticker, value in zip(tickers, values):
                if value >= threshold:
                    main_positions.append((ticker, value))
                else:
                    other_value += value

            if other_value > 0:
                main_positions.append(("Other", other_value))

            tickers, values = zip(*main_positions)

            # Create chart
            fig, ax = plt.subplots(figsize=(10, 10), dpi=self.DPI)
            fig.patch.set_facecolor(self.BACKGROUND_COLOR)

            # Generate colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))

            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                values,
                labels=tickers,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'color': self.TEXT_COLOR, 'fontsize': 12}
            )

            # Enhance autopct text
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontweight('bold')

            ax.set_title(
                "Portfolio Composition",
                fontsize=16,
                fontweight='bold',
                color=self.TEXT_COLOR,
                pad=20
            )

            return self._save_figure(fig, "portfolio_composition.png")

        except Exception as e:
            logger.error(f"Error generating composition chart: {e}")
            raise

    def generate_greeks_chart(
        self,
        position_data: Dict[str, Any]
    ) -> str:
        """
        Generate options Greeks visualization.

        Args:
            position_data: Dictionary with 'ticker' and Greeks ('delta', 'gamma', 'theta', 'vega')

        Returns:
            Path to saved chart image
        """
        try:
            ticker = position_data.get('ticker', 'Position')
            greeks = {
                'Delta': position_data.get('delta', 0),
                'Gamma': position_data.get('gamma', 0),
                'Theta': position_data.get('theta', 0),
                'Vega': position_data.get('vega', 0),
            }

            # Create chart
            fig, ax = self._setup_figure(
                title=f"{ticker} - Options Greeks",
                xlabel="Greek",
                ylabel="Value"
            )

            # Create bars
            greek_names = list(greeks.keys())
            greek_values = list(greeks.values())
            colors = [self.PRIMARY_COLOR, self.SECONDARY_COLOR, '#ff00ff', '#00ff00']

            bars = ax.bar(greek_names, greek_values, color=colors, alpha=0.7)

            # Add value labels
            for bar, value in zip(bars, greek_values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{value:.4f}',
                    ha='center',
                    va='bottom' if value >= 0 else 'top',
                    fontsize=12,
                    fontweight='bold',
                    color=self.TEXT_COLOR
                )

            # Add zero line
            ax.axhline(y=0, color=self.TEXT_COLOR, linestyle='-', linewidth=0.5)

            return self._save_figure(fig, f"greeks_{ticker}.png")

        except Exception as e:
            logger.error(f"Error generating Greeks chart: {e}")
            raise


# Example usage
if __name__ == "__main__":
    import sys
    from datetime import datetime, timedelta

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing AVA Chart Generator...\n")

    gen = ChartGenerator()

    try:
        # Test 1: Portfolio chart
        print("Test 1: Portfolio balance chart")
        balances = [
            {
                'timestamp': datetime.now() - timedelta(days=i),
                'balance': 100000 + (i * 100) + (i ** 2)
            }
            for i in range(30, 0, -1)
        ]
        chart_path = gen.generate_portfolio_chart(balances, days=30)
        print(f"✅ Created: {chart_path}")

        # Test 2: Stock chart
        print("\nTest 2: Stock price chart")
        price_data = [
            {
                'date': datetime.now() - timedelta(days=i),
                'close': 150 + np.sin(i/5) * 10,
                'volume': 1000000 + np.random.randint(-200000, 200000)
            }
            for i in range(30, 0, -1)
        ]
        chart_path = gen.generate_stock_chart("NVDA", price_data)
        print(f"✅ Created: {chart_path}")

        # Test 3: Position P&L chart
        print("\nTest 3: Position P&L chart")
        positions = [
            {'ticker': 'AAPL', 'pnl': 1500},
            {'ticker': 'NVDA', 'pnl': 2000},
            {'ticker': 'MSFT', 'pnl': -500},
            {'ticker': 'TSLA', 'pnl': -200},
        ]
        chart_path = gen.generate_position_pnl_chart(positions)
        print(f"✅ Created: {chart_path}")

        # Test 4: Portfolio composition chart
        print("\nTest 4: Portfolio composition chart")
        positions = [
            {'ticker': 'AAPL', 'market_value': 10000},
            {'ticker': 'NVDA', 'market_value': 15000},
            {'ticker': 'MSFT', 'market_value': 8000},
            {'ticker': 'TSLA', 'market_value': 5000},
            {'ticker': 'GOOGL', 'market_value': 2000},
        ]
        chart_path = gen.generate_portfolio_composition_chart(positions)
        print(f"✅ Created: {chart_path}")

        # Test 5: Greeks chart
        print("\nTest 5: Greeks chart")
        position_data = {
            'ticker': 'AAPL',
            'delta': 0.65,
            'gamma': 0.03,
            'theta': -0.05,
            'vega': 0.15
        }
        chart_path = gen.generate_greeks_chart(position_data)
        print(f"✅ Created: {chart_path}")

        print("\n✅ All chart tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
