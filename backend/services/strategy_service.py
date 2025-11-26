import logging
from typing import List, Dict, Any, Optional
from backend.config import get_settings
from src.watchlist_strategy_analyzer import WatchlistStrategyAnalyzer, StrategyAnalysis

logger = logging.getLogger(__name__)

class StrategyService:
    def __init__(self):
        self.settings = get_settings()
        self.analyzer = WatchlistStrategyAnalyzer()

    async def analyze_watchlist(
        self, 
        watchlist_name: str, 
        min_score: float = 60.0,
        strategies: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a watchlist for trading strategies.
        """
        try:
            # Run analysis (this is synchronous in the original code, might block)
            # In a real prod app, run in a thread pool
            results: List[StrategyAnalysis] = self.analyzer.analyze_watchlist(
                watchlist_name=watchlist_name,
                min_score=min_score,
                strategies=strategies
            )
            
            # Convert to dict
            return [self._analysis_to_dict(r) for r in results]
            
        except Exception as e:
            logger.error(f"Error in StrategyService.analyze_watchlist: {str(e)}")
            raise

    def _analysis_to_dict(self, analysis: StrategyAnalysis) -> Dict[str, Any]:
        """Convert StrategyAnalysis dataclass to dict."""
        from dataclasses import asdict
        return asdict(analysis)

# Singleton
_strategy_service = None

def get_strategy_service() -> StrategyService:
    global _strategy_service
    if _strategy_service is None:
        _strategy_service = StrategyService()
    return _strategy_service
