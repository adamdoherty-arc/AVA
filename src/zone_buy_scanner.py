"""
Buy Zone Scanner - Enhanced scanner for finding best stocks in demand zones

Incorporates best practices from GitHub and Reddit:
- Multi-factor rating system
- Watchlist integration
- Database stock scanning
- Visual ranking and filtering
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from src.zone_database_manager import ZoneDatabaseManager
from src.zone_analyzer import ZoneAnalyzer
from src.tradingview_db_manager import TradingViewDBManager
import os

logger = logging.getLogger(__name__)


class BuyZoneScanner:
    """
    Enhanced scanner for finding stocks in demand (buy) zones
    
    Features:
    - Pulls from TradingView watchlists
    - Pulls from database stocks
    - Multi-factor rating system
    - Distance from zone calculation
    - Strength score weighting
    - Visual ranking
    """
    
    def __init__(self) -> None:
        self.zone_db = ZoneDatabaseManager()
        self.zone_analyzer = ZoneAnalyzer()
        self.tv_manager = TradingViewDBManager()
    
    def get_stocks_from_sources(
        self,
        watchlist_names: Optional[List[str]] = None,
        use_database_stocks: bool = True,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get stocks from watchlists and/or database
        
        Args:
            watchlist_names: List of watchlist names to include (None = all)
            use_database_stocks: Include stocks from database
            limit: Maximum number of stocks to return
        
        Returns:
            List of unique stock symbols
        """
        symbols = set()
        
        # Get from watchlists
        if watchlist_names is None:
            # Get all watchlists
            watchlists = self.tv_manager.get_all_symbols()
            for watchlist_name, watchlist_symbols in watchlists.items():
                symbols.update(watchlist_symbols)
        else:
            # Get specific watchlists
            for watchlist_name in watchlist_names:
                watchlist_symbols = self.tv_manager.get_watchlist_symbols(watchlist_name)
                symbols.update(watchlist_symbols)
        
        # Get from database stocks
        if use_database_stocks:
            db_symbols = self._get_database_stocks()
            symbols.update(db_symbols)
        
        symbols_list = sorted(list(symbols))
        
        if limit:
            symbols_list = symbols_list[:limit]
        
        logger.info(f"Found {len(symbols_list)} unique stocks from sources")
        return symbols_list
    
    def _get_database_stocks(self) -> List[str]:
        """Get stocks from database stocks table"""
        try:
            conn = self.zone_db.get_connection()
            cur = conn.cursor()
            
            # Get from stocks table
            cur.execute("""
                SELECT DISTINCT ticker
                FROM stocks
                WHERE ticker IS NOT NULL
                  AND ticker != ''
                ORDER BY ticker
            """)
            
            results = cur.fetchall()
            symbols = [row[0] for row in results]
            
            cur.close()
            conn.close()
            
            return symbols
        except Exception as e:
            logger.error(f"Error getting database stocks: {e}")
            # Fallback: try stock_data table
            try:
                conn = self.zone_db.get_connection()
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT symbol FROM stock_data WHERE symbol IS NOT NULL")
                results = cur.fetchall()
                symbols = [row[0] for row in results]
                cur.close()
                conn.close()
                return symbols
            except:
                return []
    
    def scan_for_buy_zones(
        self,
        symbols: Optional[List[str]] = None,
        watchlist_names: Optional[List[str]] = None,
        use_database_stocks: bool = True,
        max_distance_pct: float = 5.0,
        min_strength: int = 50,
        min_rating: float = 60.0
    ) -> pd.DataFrame:
        """
        Scan stocks for demand (buy) zones and calculate ratings
        
        Args:
            symbols: Specific symbols to scan (None = auto-detect from sources)
            watchlist_names: Watchlist names to include
            use_database_stocks: Include database stocks
            max_distance_pct: Maximum distance from zone (percentage)
            min_strength: Minimum zone strength score
            min_rating: Minimum overall rating
        
        Returns:
            DataFrame with columns:
            - Symbol
            - Current Price
            - Zone Type (DEMAND)
            - Zone Bottom
            - Zone Top
            - Zone Midpoint
            - Distance from Zone (%)
            - Zone Strength
            - Zone Status
            - Distance Score (0-100)
            - Strength Score (0-100)
            - Freshness Score (0-100)
            - Overall Rating (0-100)
            - Recommendation
        """
        # Get symbols if not provided
        if symbols is None:
            symbols = self.get_stocks_from_sources(
                watchlist_names=watchlist_names,
                use_database_stocks=use_database_stocks
            )
        
        results = []
        
        logger.info(f"Scanning {len(symbols)} symbols for buy zones...")
        
        for symbol in symbols:
            try:
                # Get active demand zones for this symbol
                zones = self.zone_db.get_active_zones(
                    symbol=symbol,
                    zone_type='DEMAND',
                    min_strength=min_strength
                )
                
                if not zones:
                    continue
                
                # Get current price
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1m')
                    if data.empty:
                        continue
                    current_price = float(data['Close'].iloc[-1])
                except:
                    continue
                
                # Analyze each zone
                for zone in zones:
                    # Calculate distance from zone
                    zone_midpoint = float(zone['zone_midpoint'])
                    distance_pct = abs((current_price - zone_midpoint) / zone_midpoint) * 100
                    
                    # Skip if too far
                    if distance_pct > max_distance_pct:
                        continue
                    
                    # Calculate rating components
                    rating_data = self._calculate_zone_rating(
                        zone=zone,
                        current_price=current_price,
                        distance_pct=distance_pct
                    )
                    
                    # Only include if meets minimum rating
                    if rating_data['overall_rating'] < min_rating:
                        continue
                    
                    # Add to results
                    result = {
                        'Symbol': symbol,
                        'Current Price': current_price,
                        'Zone Type': 'DEMAND',
                        'Zone Bottom': float(zone['zone_bottom']),
                        'Zone Top': float(zone['zone_top']),
                        'Zone Midpoint': zone_midpoint,
                        'Distance from Zone (%)': round(distance_pct, 2),
                        'Zone Strength': float(zone['strength_score']),
                        'Zone Status': zone['status'],
                        'Test Count': zone.get('test_count', 0),
                        'Formed Date': zone.get('formed_date'),
                        'Distance Score': rating_data['distance_score'],
                        'Strength Score': rating_data['strength_score'],
                        'Freshness Score': rating_data['freshness_score'],
                        'Overall Rating': rating_data['overall_rating'],
                        'Recommendation': rating_data['recommendation']
                    }
                    
                    results.append(result)
            
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Create DataFrame
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Sort by overall rating (descending)
        df = df.sort_values('Overall Rating', ascending=False)
        
        return df
    
    def _calculate_zone_rating(
        self,
        zone: Dict,
        current_price: float,
        distance_pct: float
    ) -> Dict:
        """
        Calculate comprehensive rating for a buy zone
        
        Rating factors (0-100 scale):
        1. Distance Score (40% weight): Closer = better
        2. Strength Score (35% weight): Zone strength from database
        3. Freshness Score (25% weight): Untested zones = better
        
        Returns:
            Dictionary with scores and recommendation
        """
        # 1. Distance Score (0-100): Closer to zone = higher score
        # 0% distance = 100, 5% distance = 0
        distance_score = max(0, 100 - (distance_pct / 5.0) * 100)
        
        # 2. Strength Score (0-100): From zone strength_score
        strength_score = min(100, float(zone.get('strength_score', 0)))
        
        # 3. Freshness Score (0-100): Based on status and test count
        status = zone.get('status', 'FRESH')
        test_count = zone.get('test_count', 0)
        
        if status == 'FRESH':
            freshness_score = 100
        elif status == 'TESTED' and test_count == 1:
            freshness_score = 80
        elif status == 'TESTED' and test_count == 2:
            freshness_score = 60
        elif status == 'WEAK':
            freshness_score = 40
        else:
            freshness_score = 20
        
        # Calculate weighted overall rating
        overall_rating = (
            distance_score * 0.40 +
            strength_score * 0.35 +
            freshness_score * 0.25
        )
        
        # Generate recommendation
        if overall_rating >= 85:
            recommendation = "🔥 Excellent - Strong buy zone"
        elif overall_rating >= 75:
            recommendation = "✅ Very Good - Good buy opportunity"
        elif overall_rating >= 65:
            recommendation = "👍 Good - Consider buying"
        elif overall_rating >= 55:
            recommendation = "⚠️ Fair - Monitor closely"
        else:
            recommendation = "❌ Weak - Low priority"
        
        return {
            'distance_score': round(distance_score, 1),
            'strength_score': round(strength_score, 1),
            'freshness_score': round(freshness_score, 1),
            'overall_rating': round(overall_rating, 1),
            'recommendation': recommendation
        }
    
    def get_zone_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics for buy zones"""
        if df.empty:
            return {
                'total_opportunities': 0,
                'avg_rating': 0,
                'avg_distance': 0,
                'avg_strength': 0,
                'excellent_count': 0,
                'very_good_count': 0,
                'good_count': 0
            }
        
        return {
            'total_opportunities': len(df),
            'avg_rating': round(df['Overall Rating'].mean(), 1),
            'avg_distance': round(df['Distance from Zone (%)'].mean(), 2),
            'avg_strength': round(df['Zone Strength'].mean(), 1),
            'excellent_count': len(df[df['Overall Rating'] >= 85]),
            'very_good_count': len(df[(df['Overall Rating'] >= 75) & (df['Overall Rating'] < 85)]),
            'good_count': len(df[(df['Overall Rating'] >= 65) & (df['Overall Rating'] < 75)])
        }

