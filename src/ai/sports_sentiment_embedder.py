"""
Sports Sentiment Analyzer using HuggingFace Embeddings
Analyzes news headlines and social sentiment for sports predictions

Uses: sentence-transformers/all-MiniLM-L6-v2 (fast, 5x faster than mpnet)
Performance: 14k sentences/sec on CPU, 84-85% STS-B accuracy

Author: AI Engineer
Created: 2025-11-15
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SportsSentimentAnalyzer:
    """
    Analyze sports news and social sentiment using semantic embeddings

    Uses pre-trained sentence-transformers model for:
    - News headline analysis
    - Social media sentiment
    - Team momentum detection
    - Injury impact assessment
    """

    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize sentiment analyzer

        Args:
            model_name: HuggingFace model to use
                - all-MiniLM-L6-v2: Fast, 14k sent/sec (RECOMMENDED)
                - all-mpnet-base-v2: Slower, higher quality
        """
        self.model_name = model_name
        self.model = None
        self.positive_anchor = None
        self.negative_anchor = None

        # Lazy loading - only load when needed
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load model and create sentiment anchors"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

            # Create sentiment anchors (one-time computation)
            logger.info("Creating sentiment anchors...")

            positive_phrases = [
                "team playing exceptionally well",
                "dominant performance expected",
                "strong momentum and confidence",
                "injury-free healthy roster",
                "playoff contender ready to win",
                "offense clicking on all cylinders",
                "defense shutting down opponents",
                "winning streak continues",
                "home field advantage strong",
                "coach confident in game plan"
            ]

            negative_phrases = [
                "team struggling badly",
                "key injuries hurt chances",
                "losing streak continues",
                "defensive collapse ongoing",
                "playoff hopes fading fast",
                "offense unable to score",
                "defense allowing too many points",
                "quarterback uncertainty",
                "locker room issues",
                "coach on hot seat"
            ]

            self.positive_anchor = self.model.encode(positive_phrases).mean(axis=0)
            self.negative_anchor = self.model.encode(negative_phrases).mean(axis=0)

            logger.info(f"✓ Model loaded: {self.model_name}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def analyze_headlines(self, headlines: List[str], team: str) -> Dict:
        """
        Analyze news headlines for team sentiment

        Args:
            headlines: List of news headlines/tweets
            team: Team name to filter sentiment for

        Returns:
            {
                'sentiment_score': 0.72,  # -1 (very negative) to 1 (very positive)
                'confidence': 0.85,       # 0-1, based on consistency
                'positive_headlines': [...],
                'negative_headlines': [...],
                'headline_count': 15
            }
        """
        if not headlines:
            return self._empty_result()

        # Filter headlines mentioning the team
        team_lower = team.lower()
        team_headlines = [
            h for h in headlines
            if team_lower in h.lower()
        ]

        if not team_headlines:
            logger.warning(f"No headlines found for team: {team}")
            return self._empty_result()

        # Encode headlines
        embeddings = self.model.encode(team_headlines, show_progress_bar=False)

        # Calculate sentiment scores
        sentiment_scores = []

        for emb in embeddings:
            pos_sim = self._cosine_similarity(emb, self.positive_anchor)
            neg_sim = self._cosine_similarity(emb, self.negative_anchor)

            # Normalize to -1 to 1
            sentiment = (pos_sim - neg_sim) / 2
            sentiment_scores.append(sentiment)

        # Aggregate
        avg_sentiment = float(np.mean(sentiment_scores))
        std_sentiment = float(np.std(sentiment_scores))

        # Confidence: lower std = higher confidence
        confidence = 1 - min(std_sentiment, 1.0)

        # Categorize headlines
        positive = [
            h for h, s in zip(team_headlines, sentiment_scores)
            if s > 0.2
        ]
        negative = [
            h for h, s in zip(team_headlines, sentiment_scores)
            if s < -0.2
        ]

        return {
            'sentiment_score': avg_sentiment,
            'confidence': confidence,
            'positive_headlines': positive[:5],  # Top 5
            'negative_headlines': negative[:5],
            'headline_count': len(team_headlines),
            'raw_scores': sentiment_scores
        }

    def compare_teams(
        self,
        team1: str,
        team2: str,
        news_data: List[str]
    ) -> Dict:
        """
        Compare sentiment between two teams

        Args:
            team1: First team name
            team2: Second team name
            news_data: List of news headlines/articles

        Returns:
            {
                'team1_sentiment': 0.45,
                'team2_sentiment': -0.12,
                'advantage': 0.57,  # Positive favors team1
                'confidence': 0.78,
                'winner': 'team1'
            }
        """
        team1_analysis = self.analyze_headlines(news_data, team1)
        team2_analysis = self.analyze_headlines(news_data, team2)

        sentiment_diff = (
            team1_analysis['sentiment_score'] -
            team2_analysis['sentiment_score']
        )

        # Average confidence
        avg_confidence = (
            team1_analysis['confidence'] +
            team2_analysis['confidence']
        ) / 2

        # Determine winner
        if abs(sentiment_diff) < 0.1:
            winner = 'neutral'
        else:
            winner = team1 if sentiment_diff > 0 else team2

        return {
            'team1_sentiment': team1_analysis['sentiment_score'],
            'team2_sentiment': team2_analysis['sentiment_score'],
            'advantage': sentiment_diff,
            'confidence': avg_confidence,
            'winner': winner,
            'team1_headline_count': team1_analysis['headline_count'],
            'team2_headline_count': team2_analysis['headline_count']
        }

    def analyze_momentum(
        self,
        team: str,
        headlines_by_date: Dict[str, List[str]]
    ) -> Dict:
        """
        Analyze sentiment momentum over time

        Args:
            team: Team name
            headlines_by_date: Dict mapping date -> headlines
                Example: {'2025-11-10': [...], '2025-11-11': [...]}

        Returns:
            {
                'current_sentiment': 0.65,
                'momentum': 'positive',  # positive/negative/neutral
                'velocity': 0.12,        # Rate of change
                'trend': [0.45, 0.52, 0.65]  # Historical trend
            }
        """
        if not headlines_by_date:
            return {
                'current_sentiment': 0.0,
                'momentum': 'neutral',
                'velocity': 0.0,
                'trend': []
            }

        # Sort by date
        sorted_dates = sorted(headlines_by_date.keys())

        # Analyze each day
        daily_sentiments = []

        for date in sorted_dates:
            headlines = headlines_by_date[date]
            analysis = self.analyze_headlines(headlines, team)
            daily_sentiments.append(analysis['sentiment_score'])

        # Calculate momentum
        if len(daily_sentiments) < 2:
            momentum = 'neutral'
            velocity = 0.0
        else:
            # Compare recent vs older sentiment
            recent_avg = np.mean(daily_sentiments[-3:])
            older_avg = np.mean(daily_sentiments[:-3]) if len(daily_sentiments) > 3 else recent_avg

            velocity = float(recent_avg - older_avg)

            if velocity > 0.1:
                momentum = 'positive'
            elif velocity < -0.1:
                momentum = 'negative'
            else:
                momentum = 'neutral'

        return {
            'current_sentiment': daily_sentiments[-1] if daily_sentiments else 0.0,
            'momentum': momentum,
            'velocity': velocity,
            'trend': daily_sentiments
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def _empty_result(self) -> Dict:
        """Return empty result when no data available"""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'positive_headlines': [],
            'negative_headlines': [],
            'headline_count': 0,
            'raw_scores': []
        }

    def save_anchors(self, filepath: str):
        """Save sentiment anchors to disk for faster loading"""
        anchor_data = {
            'positive_anchor': self.positive_anchor,
            'negative_anchor': self.negative_anchor,
            'model_name': self.model_name
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(anchor_data, f)

        logger.info(f"Sentiment anchors saved to {filepath}")

    def load_anchors(self, filepath: str):
        """Load pre-computed sentiment anchors"""
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"Anchor file not found: {filepath}")
            return

        with open(filepath, 'rb') as f:
            anchor_data = pickle.load(f)

        if anchor_data['model_name'] != self.model_name:
            logger.warning(
                f"Anchor model mismatch: {anchor_data['model_name']} vs {self.model_name}"
            )
            return

        self.positive_anchor = anchor_data['positive_anchor']
        self.negative_anchor = anchor_data['negative_anchor']

        logger.info(f"Sentiment anchors loaded from {filepath}")


# ============================================================================
# TESTING
# ============================================================================

def test_sentiment_analyzer():
    """Test the sentiment analyzer"""

    print("\n" + "="*80)
    print("SPORTS SENTIMENT ANALYZER TEST")
    print("="*80)

    # Initialize
    analyzer = SportsSentimentAnalyzer()

    # Sample headlines
    headlines = [
        "Chiefs offense looks unstoppable with Mahomes leading the way",
        "Bills defense dominates in recent games, shutting down opponents",
        "Chiefs tight end Kelce questionable with ankle injury",
        "Bills running back out for season with torn ACL",
        "Kansas City Chiefs favored to win Super Bowl this year",
        "Buffalo Bills struggling with injuries on offensive line",
        "Patrick Mahomes named MVP frontrunner after stellar performance",
        "Josh Allen throws 3 interceptions in disappointing loss",
        "Chiefs home record best in AFC this season",
        "Bills face tough road ahead with difficult schedule"
    ]

    # Test 1: Analyze Chiefs sentiment
    print("\n1. Analyzing Chiefs Headlines:")
    chiefs_result = analyzer.analyze_headlines(headlines, "Chiefs")
    print(f"   Sentiment Score: {chiefs_result['sentiment_score']:.3f}")
    print(f"   Confidence: {chiefs_result['confidence']:.3f}")
    print(f"   Headlines Analyzed: {chiefs_result['headline_count']}")
    print(f"   Positive Headlines: {len(chiefs_result['positive_headlines'])}")
    print(f"   Negative Headlines: {len(chiefs_result['negative_headlines'])}")

    # Test 2: Analyze Bills sentiment
    print("\n2. Analyzing Bills Headlines:")
    bills_result = analyzer.analyze_headlines(headlines, "Bills")
    print(f"   Sentiment Score: {bills_result['sentiment_score']:.3f}")
    print(f"   Confidence: {bills_result['confidence']:.3f}")
    print(f"   Headlines Analyzed: {bills_result['headline_count']}")

    # Test 3: Compare teams
    print("\n3. Comparing Teams:")
    comparison = analyzer.compare_teams("Chiefs", "Bills", headlines)
    print(f"   Chiefs Sentiment: {comparison['team1_sentiment']:.3f}")
    print(f"   Bills Sentiment: {comparison['team2_sentiment']:.3f}")
    print(f"   Advantage: {comparison['advantage']:.3f} (favors {comparison['winner']})")
    print(f"   Confidence: {comparison['confidence']:.3f}")

    # Test 4: Momentum analysis
    print("\n4. Momentum Analysis (Chiefs):")
    headlines_by_date = {
        '2025-11-10': [
            "Chiefs lose close game in overtime",
            "Mahomes struggles with accuracy"
        ],
        '2025-11-11': [
            "Chiefs practice hard after tough loss",
            "Defense looks better in practice"
        ],
        '2025-11-12': [
            "Chiefs offense back on track",
            "Mahomes confident heading into next game"
        ]
    }

    momentum = analyzer.analyze_momentum("Chiefs", headlines_by_date)
    print(f"   Current Sentiment: {momentum['current_sentiment']:.3f}")
    print(f"   Momentum: {momentum['momentum']}")
    print(f"   Velocity: {momentum['velocity']:.3f}")
    print(f"   Trend: {[f'{s:.2f}' for s in momentum['trend']]}")

    print("\n" + "="*80)
    print("✓ Test Complete!")
    print("="*80)


if __name__ == "__main__":
    test_sentiment_analyzer()
