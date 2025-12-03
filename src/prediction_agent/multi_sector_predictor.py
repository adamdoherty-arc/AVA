"""
Multi-Sector Predictor

Main prediction engine for sports, politics, economics, and crypto markets.
Combines XGBoost ensemble with existing LLM predictions.

Author: Python Pro
Created: 2025-11-09
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from .ensemble import EnsembleModel
from .features import FeatureEngineering
from .sector_configs import MarketSector, detect_sector, get_sector_config, SECTOR_CONFIGS
from .data_sources import DataSourceManager

logger = logging.getLogger(__name__)


class MultiSectorPredictor:
    """
    Multi-sector prediction system

    Handles prediction for sports, politics, economics, and crypto markets.
    Integrates XGBoost models with existing LLM predictions.
    """

    VERSION = "1.0.0"

    def __init__(self,
                 models_dir: Optional[str] = None,
                 data_source_manager: Optional[DataSourceManager] = None):
        """
        Initialize multi-sector predictor

        Args:
            models_dir: Directory to store trained models
            data_source_manager: DataSourceManager instance (creates new if None)
        """
        self.models_dir = Path(models_dir or os.path.join(os.getcwd(), "models", "prediction_agent"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.data_source_manager = data_source_manager or DataSourceManager()
        self.feature_engineering = FeatureEngineering(self.data_source_manager)

        # Sector-specific models
        self.models: Dict[MarketSector, EnsembleModel] = {}

        # Load existing models if available
        self._load_models()

    def _load_models(self) -> None:
        """Load trained models from disk"""
        for sector in MarketSector:
            model_path = self.models_dir / f"{sector.value}_model.pkl"

            if model_path.exists():
                try:
                    config = get_sector_config(sector)
                    model = EnsembleModel(
                        xgboost_params=config.xgboost_params,
                        calibration_method=config.calibration_method
                    )
                    model.load_model(str(model_path))
                    self.models[sector] = model
                    logger.info(f"Loaded {sector.value} model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {sector.value} model: {e}")

    def _get_or_create_model(self, sector: MarketSector) -> EnsembleModel:
        """Get existing model or create new one for sector"""
        if sector in self.models:
            return self.models[sector]

        # Create new model with sector-specific config
        config = get_sector_config(sector)
        model = EnsembleModel(
            xgboost_params=config.xgboost_params,
            calibration_method=config.calibration_method
        )

        self.models[sector] = model
        return model

    def predict_market(self,
                      market: Dict,
                      llm_prediction: Optional[Dict] = None,
                      sector: Optional[MarketSector] = None) -> Dict:
        """
        Predict outcome for a single market

        Args:
            market: Market dictionary with pricing and metadata
            llm_prediction: Optional existing LLM prediction
            sector: Market sector (auto-detected if None)

        Returns:
            Dictionary with prediction results
        """
        # Detect sector if not provided
        if sector is None:
            market_text = f"{market.get('title', '')} {market.get('ticker', '')}"
            sector = detect_sector(market_text)

            if sector is None:
                logger.warning(f"Could not detect sector for market: {market.get('ticker')}")
                return self._create_fallback_prediction(market, llm_prediction)

        # Get sector config
        config = get_sector_config(sector)

        # Build context for feature extraction
        context = self._build_context(market, sector)

        # Extract features
        features = self.feature_engineering.extract_features(market, sector, context)

        # Create feature DataFrame
        feature_df = pd.DataFrame([features])

        # Prepare LLM predictions if available
        llm_df = None
        if llm_prediction:
            llm_df = pd.DataFrame([{
                'ticker': market.get('ticker'),
                'predicted_prob': self._get_llm_probability(llm_prediction),
                'confidence_score': llm_prediction.get('confidence_score', 50.0)
            }])

        # Get or create model for sector
        model = self._get_or_create_model(sector)

        if not model.is_trained:
            logger.warning(f"No trained model for {sector.value}. Using LLM prediction only.")
            return self._create_fallback_prediction(market, llm_prediction)

        # Prepare features for prediction
        X = model.prepare_features(feature_df, llm_df)

        # Make prediction
        _, probabilities = model.predict(X)
        probability = float(probabilities[0])

        # Calculate confidence
        confidence = float(model.calculate_confidence(probabilities)[0])

        # Determine predicted outcome
        predicted_outcome = 'yes' if probability >= 0.5 else 'no'

        # Calculate edge
        market_price = float(market.get('yes_price', 0.5))
        if predicted_outcome == 'yes':
            edge = ((probability - market_price) / market_price * 100) if market_price > 0 else 0
        else:
            no_price = float(market.get('no_price', 0.5))
            edge = (((1 - probability) - no_price) / no_price * 100) if no_price > 0 else 0

        # Cap edge at reasonable bounds
        edge = max(-500, min(edge, 500))

        # Generate recommendation
        recommendation = self._generate_recommendation(
            probability, confidence, edge, config
        )

        # Create prediction result
        result = {
            'ticker': market.get('ticker'),
            'sector': sector.value,
            'predicted_outcome': predicted_outcome,
            'confidence_score': round(confidence, 2),
            'calibrated_probability': round(probability, 4),
            'edge_percentage': round(edge, 2),
            'recommended_action': recommendation['action'],
            'recommended_stake_pct': recommendation['stake_pct'],
            'prediction_agent_version': self.VERSION,
            'model_trained_samples': model.n_samples_trained,
            'timestamp': datetime.now().isoformat()
        }

        # Add LLM comparison if available
        if llm_prediction:
            result['llm_confidence'] = llm_prediction.get('confidence_score', 0)
            result['llm_predicted_outcome'] = llm_prediction.get('predicted_outcome', 'unknown')
            result['ensemble_weight_ml'] = 0.7  # ML gets 70% weight
            result['ensemble_weight_llm'] = 0.3  # LLM gets 30% weight

        return result

    def predict_markets(self,
                       markets: List[Dict],
                       llm_predictions: Optional[List[Dict]] = None,
                       sector: Optional[MarketSector] = None) -> List[Dict]:
        """
        Predict outcomes for multiple markets

        Args:
            markets: List of market dictionaries
            llm_predictions: Optional list of existing LLM predictions
            sector: Market sector (auto-detected if None)

        Returns:
            List of prediction dictionaries
        """
        if not markets:
            return []

        # Create LLM lookup dictionary
        llm_lookup = {}
        if llm_predictions:
            llm_lookup = {pred.get('ticker'): pred for pred in llm_predictions}

        # Predict each market
        predictions = []
        for market in markets:
            ticker = market.get('ticker')
            llm_pred = llm_lookup.get(ticker)

            try:
                prediction = self.predict_market(market, llm_pred, sector)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting market {ticker}: {e}")
                # Add fallback prediction
                predictions.append(self._create_fallback_prediction(market, llm_pred))

        # Rank predictions
        predictions = self._rank_predictions(predictions)

        return predictions

    def train_sector_model(self,
                          sector: MarketSector,
                          training_markets: List[Dict],
                          outcomes: List[int],
                          llm_predictions: Optional[List[Dict]] = None) -> Dict:
        """
        Train model for a specific sector

        Args:
            sector: Market sector
            training_markets: List of historical markets
            outcomes: List of actual outcomes (0 or 1)
            llm_predictions: Optional list of LLM predictions for training markets

        Returns:
            Dictionary of training metrics
        """
        if len(training_markets) != len(outcomes):
            raise ValueError("training_markets and outcomes must have same length")

        if len(training_markets) < 100:
            logger.warning(f"Only {len(training_markets)} samples for training. "
                         f"Recommend at least 100 samples.")

        logger.info(f"Training {sector.value} model on {len(training_markets)} samples...")

        # Build contexts
        contexts = [self._build_context(m, sector) for m in training_markets]

        # Extract features
        feature_df = self.feature_engineering.create_feature_dataframe(
            training_markets, sector, contexts
        )

        # Prepare LLM predictions
        llm_df = None
        if llm_predictions:
            llm_df = pd.DataFrame([
                {
                    'ticker': pred.get('ticker'),
                    'predicted_prob': self._get_llm_probability(pred),
                    'confidence_score': pred.get('confidence_score', 50.0)
                }
                for pred in llm_predictions
            ])

        # Get or create model
        model = self._get_or_create_model(sector)

        # Prepare features
        X = model.prepare_features(feature_df, llm_df)

        # Train model
        metrics = model.train(X, np.array(outcomes))

        # Save model
        model_path = self.models_dir / f"{sector.value}_model.pkl"
        model.save_model(str(model_path))

        logger.info(f"Model saved to {model_path}")

        return metrics

    def _build_context(self, market: Dict, sector: MarketSector) -> Dict:
        """Build context dictionary for feature extraction"""
        context = {}

        title = market.get('title', '').lower()
        ticker = market.get('ticker', '').lower()
        combined_text = f"{title} {ticker}"

        if sector == MarketSector.SPORTS:
            # Determine sport type
            if 'nfl' in combined_text or 'football' in combined_text:
                context['sport'] = 'nfl'
            elif 'nba' in combined_text or 'basketball' in combined_text:
                context['sport'] = 'nba'
            elif 'mlb' in combined_text or 'baseball' in combined_text:
                context['sport'] = 'mlb'
            else:
                context['sport'] = 'sports'

            # Extract keywords
            keywords = [market.get('home_team', ''), market.get('away_team', '')]
            keywords = [k.lower() for k in keywords if k]
            context['keywords'] = keywords

        elif sector == MarketSector.POLITICS:
            # Extract keywords from title
            political_terms = ['election', 'president', 'senate', 'congress', 'vote']
            keywords = [term for term in political_terms if term in combined_text]
            context['keywords'] = keywords or ['politics']

        elif sector == MarketSector.ECONOMICS:
            # Relevant economic indicators
            context['indicators'] = ['UNRATE', 'CPIAUCSL']  # Unemployment, Inflation
            context['indices'] = ['SPY', '^VIX']  # S&P 500, VIX

        elif sector == MarketSector.CRYPTO:
            # Determine which crypto
            if 'bitcoin' in combined_text or 'btc' in combined_text:
                context['coins'] = ['bitcoin']
                context['keywords'] = ['bitcoin']
            elif 'ethereum' in combined_text or 'eth' in combined_text:
                context['coins'] = ['ethereum']
                context['keywords'] = ['ethereum']
            else:
                context['coins'] = ['bitcoin']  # Default
                context['keywords'] = ['crypto']

        return context

    def _get_llm_probability(self, llm_prediction: Dict) -> float:
        """Extract probability from LLM prediction"""
        # Check if probability is directly provided
        if 'predicted_probability' in llm_prediction:
            return float(llm_prediction['predicted_probability'])

        # Otherwise infer from outcome and confidence
        outcome = llm_prediction.get('predicted_outcome', 'yes')
        confidence = float(llm_prediction.get('confidence_score', 50.0)) / 100

        if outcome == 'yes':
            return 0.5 + confidence * 0.5  # Map to 0.5-1.0
        else:
            return 0.5 - confidence * 0.5  # Map to 0.0-0.5

    def _generate_recommendation(self,
                                probability: float,
                                confidence: float,
                                edge: float,
                                config) -> Dict:
        """Generate trading recommendation"""
        recommendation = {
            'action': 'pass',
            'stake_pct': 0.0
        }

        # Check minimum confidence threshold
        if confidence < config.min_confidence_threshold * 100:
            return recommendation

        # Determine action based on edge and confidence
        if edge > 15 and confidence > 80:
            recommendation['action'] = 'strong_buy'
            recommendation['stake_pct'] = min(confidence / 10, 10.0)
        elif edge > 8 and confidence > 70:
            recommendation['action'] = 'buy'
            recommendation['stake_pct'] = min(confidence / 15, 5.0)
        elif edge > 3 and confidence > 60:
            recommendation['action'] = 'hold'
            recommendation['stake_pct'] = min(confidence / 20, 2.0)

        return recommendation

    def _create_fallback_prediction(self,
                                   market: Dict,
                                   llm_prediction: Optional[Dict] = None) -> Dict:
        """Create fallback prediction when model not available"""
        if llm_prediction:
            # Use LLM prediction as fallback
            return {
                'ticker': market.get('ticker'),
                'sector': 'unknown',
                'predicted_outcome': llm_prediction.get('predicted_outcome', 'yes'),
                'confidence_score': llm_prediction.get('confidence_score', 50.0),
                'calibrated_probability': self._get_llm_probability(llm_prediction),
                'edge_percentage': llm_prediction.get('edge_percentage', 0.0),
                'recommended_action': llm_prediction.get('recommended_action', 'pass'),
                'recommended_stake_pct': llm_prediction.get('recommended_stake_pct', 0.0),
                'prediction_agent_version': self.VERSION,
                'model_trained_samples': 0,
                'timestamp': datetime.now().isoformat(),
                'fallback_mode': True,
                'llm_confidence': llm_prediction.get('confidence_score', 50.0),
                'llm_predicted_outcome': llm_prediction.get('predicted_outcome', 'yes'),
                'ensemble_weight_ml': 0.0,
                'ensemble_weight_llm': 1.0
            }
        else:
            # Pure fallback - neutral prediction
            return {
                'ticker': market.get('ticker'),
                'sector': 'unknown',
                'predicted_outcome': 'yes',
                'confidence_score': 50.0,
                'calibrated_probability': 0.5,
                'edge_percentage': 0.0,
                'recommended_action': 'pass',
                'recommended_stake_pct': 0.0,
                'prediction_agent_version': self.VERSION,
                'model_trained_samples': 0,
                'timestamp': datetime.now().isoformat(),
                'fallback_mode': True
            }

    def _rank_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Rank predictions by opportunity score"""
        # Sort by edge, then confidence
        ranked = sorted(
            predictions,
            key=lambda x: (
                x.get('edge_percentage', 0) if x.get('edge_percentage', 0) > 0 else -999,
                x.get('confidence_score', 0)
            ),
            reverse=True
        )

        # Add rank
        for i, pred in enumerate(ranked, 1):
            pred['overall_rank'] = i

        return ranked

    def get_sector_statistics(self) -> Dict[str, Dict]:
        """Get statistics about trained models"""
        stats = {}

        for sector in MarketSector:
            if sector in self.models:
                model = self.models[sector]
                stats[sector.value] = {
                    'is_trained': model.is_trained,
                    'n_samples_trained': model.n_samples_trained,
                    'n_features': len(model.feature_names) if model.feature_names else 0,
                    'metrics': model.training_metrics
                }
            else:
                stats[sector.value] = {
                    'is_trained': False,
                    'n_samples_trained': 0,
                    'n_features': 0,
                    'metrics': {}
                }

        return stats


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictor = MultiSectorPredictor()

    print("\n" + "="*80)
    print("MULTI-SECTOR PREDICTOR TEST")
    print("="*80)

    # Test markets from different sectors
    test_markets = [
        {
            'ticker': 'NFL-CHIEFS-001',
            'title': 'Will the Chiefs win the Super Bowl?',
            'yes_price': 0.35,
            'no_price': 0.65,
            'volume': 150000,
            'open_interest': 12000,
            'close_time': datetime.now().isoformat(),
            'home_team': 'Chiefs',
            'away_team': 'Bills'
        },
        {
            'ticker': 'PRES-2024-001',
            'title': 'Will Biden win the 2024 presidential election?',
            'yes_price': 0.48,
            'no_price': 0.52,
            'volume': 500000,
            'open_interest': 50000,
            'close_time': datetime.now().isoformat()
        },
        {
            'ticker': 'GDP-Q4-001',
            'title': 'Will US GDP grow by more than 2% in Q4?',
            'yes_price': 0.62,
            'no_price': 0.38,
            'volume': 75000,
            'open_interest': 8000,
            'close_time': datetime.now().isoformat()
        },
        {
            'ticker': 'BTC-100K-001',
            'title': 'Will Bitcoin reach $100,000 by year end?',
            'yes_price': 0.28,
            'no_price': 0.72,
            'volume': 200000,
            'open_interest': 15000,
            'close_time': datetime.now().isoformat()
        }
    ]

    print("\nPredicting markets (fallback mode - no trained models)...")
    predictions = predictor.predict_markets(test_markets)

    print(f"\nGenerated {len(predictions)} predictions:\n")
    for pred in predictions:
        print(f"{pred['ticker']} ({pred['sector']}):")
        print(f"  Predicted: {pred['predicted_outcome']} @ {pred['calibrated_probability']:.2%}")
        print(f"  Confidence: {pred['confidence_score']:.1f}%")
        print(f"  Edge: {pred['edge_percentage']:.2f}%")
        print(f"  Action: {pred['recommended_action']}")
        print()

    # Show model statistics
    print("\nModel Statistics:")
    stats = predictor.get_sector_statistics()
    for sector, stat in stats.items():
        print(f"  {sector}: Trained={stat['is_trained']}, "
              f"Samples={stat['n_samples_trained']}")

    print("\n" + "="*80)
