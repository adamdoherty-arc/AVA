"""
Sector Configurations

Defines configuration for each prediction market sector (sports, politics, economics, crypto).
Each sector has specific features, data sources, and model parameters.

Author: Python Pro
Created: 2025-11-09
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MarketSector(str, Enum):
    """Market sector enumeration"""
    SPORTS = "sports"
    POLITICS = "politics"
    ECONOMICS = "economics"
    CRYPTO = "crypto"


@dataclass
class SectorConfig:
    """Configuration for a specific market sector"""

    sector: MarketSector
    display_name: str

    # Feature groups to use
    feature_groups: List[str]

    # Model parameters
    xgboost_params: Dict

    # Data source configuration
    data_sources: List[str]

    # Feature importance weights
    feature_weights: Dict[str, float] = field(default_factory=dict)

    # Calibration parameters
    calibration_method: str = "isotonic"  # "isotonic" or "platt"
    min_samples_calibration: int = 100

    # Prediction thresholds
    min_confidence_threshold: float = 0.55
    max_confidence_threshold: float = 0.95

    # Market-specific keywords for detection
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration"""
        if not 0 <= self.min_confidence_threshold <= 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")
        if not 0 <= self.max_confidence_threshold <= 1:
            raise ValueError("max_confidence_threshold must be between 0 and 1")
        if self.min_confidence_threshold >= self.max_confidence_threshold:
            raise ValueError("min_confidence_threshold must be less than max_confidence_threshold")


# ============================================================================
# SPORTS CONFIGURATION
# ============================================================================

SPORTS_CONFIG = SectorConfig(
    sector=MarketSector.SPORTS,
    display_name="Sports",
    feature_groups=[
        "price_history",
        "volume_features",
        "time_features",
        "team_stats",
        "weather",
        "matchup_features"
    ],
    xgboost_params={
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "max_features": 0.8,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42
    },
    data_sources=[
        "kalshi_markets",
        "reddit_sports",
        "team_stats_api"  # Placeholder for free sports stats API
    ],
    feature_weights={
        "team_strength": 0.30,
        "recent_form": 0.25,
        "market_sentiment": 0.20,
        "weather_impact": 0.10,
        "price_momentum": 0.15
    },
    calibration_method="isotonic",
    min_samples_calibration=150,
    min_confidence_threshold=0.55,
    max_confidence_threshold=0.92,
    keywords=[
        "nfl", "nba", "mlb", "nhl", "ncaa", "football", "basketball",
        "baseball", "hockey", "super bowl", "world series", "playoffs",
        "chiefs", "bills", "lakers", "yankees", "game", "match", "win"
    ]
)

# ============================================================================
# POLITICS CONFIGURATION
# ============================================================================

POLITICS_CONFIG = SectorConfig(
    sector=MarketSector.POLITICS,
    display_name="Politics",
    feature_groups=[
        "price_history",
        "volume_features",
        "time_features",
        "poll_data",
        "sentiment_features",
        "event_features"
    ],
    xgboost_params={
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 250,
        "subsample": 0.75,
        "max_features": 0.75,
        "min_samples_split": 12,
        "min_samples_leaf": 6,
        "random_state": 42
    },
    data_sources=[
        "kalshi_markets",
        "reddit_politics",
        "poll_aggregators"  # Placeholder for poll data
    ],
    feature_weights={
        "poll_average": 0.35,
        "sentiment_score": 0.25,
        "market_momentum": 0.20,
        "time_to_event": 0.10,
        "historical_accuracy": 0.10
    },
    calibration_method="isotonic",
    min_samples_calibration=100,
    min_confidence_threshold=0.52,
    max_confidence_threshold=0.95,
    keywords=[
        "election", "president", "senate", "congress", "vote", "democrat",
        "republican", "primary", "campaign", "poll", "debate", "gubernatorial",
        "governor", "mayor", "referendum", "ballot", "biden", "trump",
        "candidate", "nominee"
    ]
)

# ============================================================================
# ECONOMICS CONFIGURATION
# ============================================================================

ECONOMICS_CONFIG = SectorConfig(
    sector=MarketSector.ECONOMICS,
    display_name="Economics",
    feature_groups=[
        "price_history",
        "volume_features",
        "time_features",
        "economic_indicators",
        "market_features",
        "sentiment_features"
    ],
    xgboost_params={
        "max_depth": 7,
        "learning_rate": 0.04,
        "n_estimators": 300,
        "subsample": 0.8,
        "max_features": 0.7,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42
    },
    data_sources=[
        "kalshi_markets",
        "fred_api",  # Federal Reserve Economic Data
        "yfinance",
        "reddit_economics"
    ],
    feature_weights={
        "indicator_trends": 0.35,
        "market_correlation": 0.25,
        "expert_consensus": 0.20,
        "historical_patterns": 0.15,
        "volatility": 0.05
    },
    calibration_method="platt",
    min_samples_calibration=120,
    min_confidence_threshold=0.54,
    max_confidence_threshold=0.93,
    keywords=[
        "gdp", "inflation", "unemployment", "fed", "interest rate", "recession",
        "cpi", "pce", "jobs report", "fomc", "treasury", "yield", "economy",
        "economic", "fiscal", "monetary", "downturn", "growth"
    ]
)

# ============================================================================
# CRYPTO CONFIGURATION
# ============================================================================

CRYPTO_CONFIG = SectorConfig(
    sector=MarketSector.CRYPTO,
    display_name="Crypto",
    feature_groups=[
        "price_history",
        "volume_features",
        "time_features",
        "crypto_momentum",
        "volatility_features",
        "market_correlation"
    ],
    xgboost_params={
        "max_depth": 6,
        "learning_rate": 0.06,
        "n_estimators": 200,
        "subsample": 0.85,
        "max_features": 0.85,
        "min_samples_split": 8,
        "min_samples_leaf": 4,
        "random_state": 42
    },
    data_sources=[
        "kalshi_markets",
        "coingecko_api",
        "reddit_crypto"
    ],
    feature_weights={
        "price_momentum": 0.30,
        "volume_profile": 0.25,
        "volatility": 0.20,
        "market_sentiment": 0.15,
        "correlation_btc": 0.10
    },
    calibration_method="isotonic",
    min_samples_calibration=100,
    min_confidence_threshold=0.53,
    max_confidence_threshold=0.90,
    keywords=[
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "blockchain", "altcoin", "defi", "nft", "token", "coin", "price",
        "halving", "hash rate", "mining"
    ]
)

# ============================================================================
# SECTOR REGISTRY
# ============================================================================

SECTOR_CONFIGS: Dict[MarketSector, SectorConfig] = {
    MarketSector.SPORTS: SPORTS_CONFIG,
    MarketSector.POLITICS: POLITICS_CONFIG,
    MarketSector.ECONOMICS: ECONOMICS_CONFIG,
    MarketSector.CRYPTO: CRYPTO_CONFIG,
}


def detect_sector(market_text: str) -> Optional[MarketSector]:
    """
    Detect market sector based on text content

    Args:
        market_text: Market title, description, or ticker

    Returns:
        Detected MarketSector or None
    """
    market_text_lower = market_text.lower()

    # Score each sector based on keyword matches
    sector_scores: Dict[MarketSector, int] = {}

    for sector, config in SECTOR_CONFIGS.items():
        score = sum(1 for keyword in config.keywords if keyword in market_text_lower)
        if score > 0:
            sector_scores[sector] = score

    if not sector_scores:
        return None

    # Return sector with highest score
    return max(sector_scores.items(), key=lambda x: x[1])[0]


def get_sector_config(sector: MarketSector) -> SectorConfig:
    """
    Get configuration for a specific sector

    Args:
        sector: Market sector

    Returns:
        SectorConfig for the sector

    Raises:
        ValueError: If sector not found
    """
    if sector not in SECTOR_CONFIGS:
        raise ValueError(f"Unknown sector: {sector}")

    return SECTOR_CONFIGS[sector]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Test sector detection
    test_cases = [
        "Will the Chiefs win the Super Bowl?",
        "Will Biden win the 2024 election?",
        "Will US GDP grow by 2% in Q4?",
        "Will Bitcoin reach $100,000 by year end?",
    ]

    print("\n" + "="*80)
    print("SECTOR DETECTION TEST")
    print("="*80)

    for test_text in test_cases:
        sector = detect_sector(test_text)
        print(f"\nText: {test_text}")
        print(f"Detected Sector: {sector.value if sector else 'Unknown'}")

        if sector:
            config = get_sector_config(sector)
            print(f"Display Name: {config.display_name}")
            print(f"Feature Groups: {', '.join(config.feature_groups[:3])}...")
            print(f"Data Sources: {', '.join(config.data_sources)}")

    print("\n" + "="*80)
