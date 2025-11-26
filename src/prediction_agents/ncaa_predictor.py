"""
NCAA Football Prediction Agent
===============================

NCAA College Football-specific prediction agent with advanced features:
- Conference-aware modeling (Power 5 vs. Group of 5)
- Recruiting rankings integration
- Coaching experience factors
- Home field advantage (larger than NFL due to crowds)
- Rivalry game adjustments
- Talent disparity detection

Based on research showing 74-77% accuracy achievable (higher than NFL due to talent gaps).
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import math
import json
import os
from thefuzz import fuzz, process

from .base_predictor import BaseSportsPredictor
from .llm_explanation_enhancer import get_llm_enhancer


class NCAAPredictor(BaseSportsPredictor):
    """
    NCAA Football-specific prediction agent using ensemble machine learning.

    Features:
    - Conference-aware predictions (Power 5 vs. Group of 5)
    - Recruiting talent ratings
    - Home crowd impact (larger than NFL)
    - Rivalry detection and adjustments
    - Coaching experience factors
    """

    # NCAA-specific constants
    HOME_FIELD_ADVANTAGE = 3.5  # Points (higher than NFL due to crowds)
    ELO_K_FACTOR = 25  # Higher than NFL due to more variance
    ELO_BASE = 1500  # Starting Elo rating
    SPREAD_FACTOR = 28.0  # For probability to spread conversion (higher scoring games)

    # Conference tiers (power rankings)
    CONFERENCE_POWER = {
        # Power 5
        'SEC': 1.0,
        'Big Ten': 0.98,
        'ACC': 0.92,
        'Big 12': 0.90,
        'Pac-12': 0.88,  # Historical (conference realignment pending)

        # Group of 5
        'American': 0.70,
        'Mountain West': 0.68,
        'Sun Belt': 0.65,
        'Conference USA': 0.63,
        'MAC': 0.60,

        # FCS/Independent
        'Independent': 0.75,
        'FCS': 0.40
    }

    # Rivalry games (sample - would need comprehensive list)
    RIVALRIES = {
        ('Alabama', 'Auburn'): 'Iron Bowl',
        ('Michigan', 'Ohio State'): 'The Game',
        ('Texas', 'Oklahoma'): 'Red River Rivalry',
        ('USC', 'Notre Dame'): 'Historic Rivalry',
        ('Florida', 'Georgia'): 'World\'s Largest Outdoor Cocktail Party',
        ('Army', 'Navy'): 'Army-Navy Game',
    }

    # Team strength modifiers (offense/defense rankings, recent form)
    TEAM_STRENGTHS = {
        # Elite programs
        'Georgia': {'offense': 2, 'defense': 1, 'form': 5},
        'Alabama': {'offense': 1, 'defense': 3, 'form': 4},
        'Ohio State': {'offense': 3, 'defense': 2, 'form': 5},
        'Michigan': {'offense': 5, 'defense': 4, 'form': 4},
        'Texas': {'offense': 4, 'defense': 5, 'form': 4},
        'Florida State': {'offense': 8, 'defense': 7, 'form': 3},
        'Penn State': {'offense': 10, 'defense': 6, 'form': 3},
        'LSU': {'offense': 6, 'defense': 9, 'form': 3},
        'Clemson': {'offense': 12, 'defense': 8, 'form': 3},
        'Notre Dame': {'offense': 7, 'defense': 10, 'form': 3},
        'Oregon': {'offense': 9, 'defense': 12, 'form': 3},
        'USC': {'offense': 11, 'defense': 15, 'form': 2},
        'Oklahoma': {'offense': 13, 'defense': 18, 'form': 2},
        'Tennessee': {'offense': 14, 'defense': 11, 'form': 3},
        'Auburn': {'offense': 18, 'defense': 14, 'form': 2},
        'Florida': {'offense': 22, 'defense': 13, 'form': 2},
        'Texas A&M': {'offense': 20, 'defense': 16, 'form': 2},
        # Add more teams as needed
    }

    # Common mascots to remove from team names
    MASCOTS = [
        'seminoles', 'wolfpack', 'buckeyes', 'wolverines', 'broncos',
        'bulldogs', 'tigers', 'bears', 'wildcats', 'eagles', 'hawks',
        'panthers', 'lions', 'aggies', 'cowboys', 'knights', 'trojans',
        'spartans', 'huskies', 'crimson', 'tide', 'gators', 'gamecocks',
        'volunteers', 'rebels', 'commodores', 'razorbacks', 'sooners',
        'longhorns', 'horns', 'hurricanes', 'hokies', 'cardinals', 'rams',
        'ducks', 'beavers', 'cougars', 'utes', 'scarlet knights', 'nittany lions',
        '49ers', 'golden eagles', 'blue devils', 'demon deacons', 'yellow jackets',
        'fighting irish', 'black knights', 'midshipmen', 'red raiders', 'mountaineers',
        'jayhawks', 'cyclones', 'horned frogs'
    ]

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize NCAA predictor.

        Args:
            db_config: Database configuration for loading stats
        """
        super().__init__("NCAA Football")

        self.db_config = db_config
        self.elo_ratings = {}  # Team -> current Elo rating
        self.team_stats = {}   # Team -> current season stats
        self.recruiting_rankings = {}  # Team -> recruiting score
        self.coaching_data = {}  # Team -> coach info
        self.conference_map = {}  # Team -> conference
        self.team_name_cache = {}  # Cache for fuzzy-matched team names

        # Load initial data
        self._load_team_data()
        self._load_elo_ratings()
        self._load_recruiting_data()

    def _find_best_team_match(self, team_name: str, search_dict: Dict[str, Any], threshold: int = 60) -> Optional[str]:
        """
        Use fuzzy matching to find the best team name match in a dictionary.

        Args:
            team_name: Team name from ESPN (e.g., "Florida State Seminoles")
            search_dict: Dictionary to search (e.g., self.elo_ratings)
            threshold: Minimum similarity score (0-100)

        Returns:
            Best matching key from search_dict, or None if no good match
        """
        if not team_name or not search_dict:
            return None

        # Check cache first
        cache_key = f"{team_name}:{id(search_dict)}"
        if cache_key in self.team_name_cache:
            return self.team_name_cache[cache_key]

        # Try exact match first (fast path)
        if team_name in search_dict:
            self.team_name_cache[cache_key] = team_name
            return team_name

        # Use fuzzy matching to find best match
        # process.extractOne returns (match, score, key)
        result = process.extractOne(
            team_name,
            search_dict.keys(),
            scorer=fuzz.token_sort_ratio  # Handles word order differences
        )

        if result and result[1] >= threshold:
            best_match = result[0]
            self.logger.debug(f"Fuzzy matched '{team_name}' -> '{best_match}' (score: {result[1]})")
            self.team_name_cache[cache_key] = best_match
            return best_match

        # No good match found
        self.logger.warning(f"No fuzzy match found for '{team_name}' (threshold: {threshold})")
        return None

    def _load_team_data(self):
        """Load NCAA team data (conferences, divisions)."""
        # This is a sample - would need complete database
        self.conference_map = {
            # SEC
            'Alabama': 'SEC',
            'Georgia': 'SEC',
            'LSU': 'SEC',
            'Florida': 'SEC',
            'Tennessee': 'SEC',
            'Auburn': 'SEC',
            'Texas A&M': 'SEC',
            'Arkansas': 'SEC',
            'Ole Miss': 'SEC',
            'Mississippi State': 'SEC',
            'Kentucky': 'SEC',
            'South Carolina': 'SEC',
            'Missouri': 'SEC',
            'Vanderbilt': 'SEC',

            # Big Ten
            'Ohio State': 'Big Ten',
            'Michigan': 'Big Ten',
            'Penn State': 'Big Ten',
            'Wisconsin': 'Big Ten',
            'Iowa': 'Big Ten',
            'Michigan State': 'Big Ten',
            'Minnesota': 'Big Ten',
            'Nebraska': 'Big Ten',
            'Northwestern': 'Big Ten',
            'Illinois': 'Big Ten',
            'Indiana': 'Big Ten',
            'Purdue': 'Big Ten',
            'Maryland': 'Big Ten',
            'Rutgers': 'Big Ten',

            # ACC
            'Clemson': 'ACC',
            'Florida State': 'ACC',
            'Miami': 'ACC',
            'North Carolina': 'ACC',
            'Virginia Tech': 'ACC',
            'Pittsburgh': 'ACC',
            'Louisville': 'ACC',
            'NC State': 'ACC',
            'Wake Forest': 'ACC',
            'Virginia': 'ACC',
            'Duke': 'ACC',
            'Georgia Tech': 'ACC',
            'Boston College': 'ACC',
            'Syracuse': 'ACC',

            # Big 12
            'Oklahoma': 'Big 12',
            'Texas': 'Big 12',
            'Oklahoma State': 'Big 12',
            'Baylor': 'Big 12',
            'TCU': 'Big 12',
            'Kansas State': 'Big 12',
            'Texas Tech': 'Big 12',
            'West Virginia': 'Big 12',
            'Iowa State': 'Big 12',
            'Kansas': 'Big 12',

            # Notre Dame (Independent)
            'Notre Dame': 'Independent',
        }

    def _load_elo_ratings(self):
        """Load or initialize Elo ratings for all teams."""
        ratings_file = 'src/data/ncaa_elo_ratings.json'

        if os.path.exists(ratings_file):
            try:
                with open(ratings_file, 'r') as f:
                    self.elo_ratings = json.load(f)
                self.logger.info(f"Loaded Elo ratings from {ratings_file}")
                return
            except Exception as e:
                self.logger.warning(f"Could not load Elo ratings: {e}")

        # Initialize all teams to base rating
        for team in self.conference_map.keys():
            self.elo_ratings[team] = self.ELO_BASE

        self.logger.info("Initialized Elo ratings to base value")

    def _save_elo_ratings(self):
        """Save current Elo ratings to file."""
        ratings_file = 'src/data/ncaa_elo_ratings.json'

        try:
            os.makedirs(os.path.dirname(ratings_file), exist_ok=True)
            with open(ratings_file, 'w') as f:
                json.dump(self.elo_ratings, f, indent=2)
            self.logger.info(f"Saved Elo ratings to {ratings_file}")
        except Exception as e:
            self.logger.error(f"Could not save Elo ratings: {e}")

    def _load_recruiting_data(self):
        """Load recruiting rankings (247Sports composite)."""
        # Sample data - would need real recruiting API
        # Higher score = better recruiting
        self.recruiting_rankings = {
            'Alabama': 95,
            'Georgia': 94,
            'Ohio State': 93,
            'Texas': 92,
            'LSU': 91,
            'Clemson': 90,
            'Michigan': 89,
            'Oklahoma': 88,
            'Texas A&M': 87,
            'Florida': 86,
            # ... etc.
        }

    def get_conference_power(self, team: str) -> float:
        """
        Get conference power rating for a team (with fuzzy matching).

        Args:
            team: Team name

        Returns:
            Conference power multiplier (0.4-1.0)
        """
        # Try fuzzy match if exact match fails
        matched_team = self._find_best_team_match(team, self.conference_map, threshold=60)
        team_key = matched_team if matched_team else team

        conference = self.conference_map.get(team_key, 'FCS')
        return self.CONFERENCE_POWER.get(conference, 0.50)

    def get_recruiting_score(self, team: str) -> float:
        """
        Get recruiting ranking score for a team (with fuzzy matching).

        Args:
            team: Team name

        Returns:
            Recruiting score (0-100, higher is better)
        """
        # Try fuzzy match if exact match fails
        matched_team = self._find_best_team_match(team, self.recruiting_rankings, threshold=60)
        team_key = matched_team if matched_team else team

        return self.recruiting_rankings.get(team_key, 50.0)  # Default to average

    def get_team_strength(self, team: str) -> Dict[str, Any]:
        """
        Get team strength metrics (with fuzzy matching).

        Args:
            team: Team name

        Returns:
            Dictionary with offensive rank, defensive rank, recent form
        """
        # Try fuzzy match if exact match fails
        matched_team = self._find_best_team_match(team, self.TEAM_STRENGTHS, threshold=60)
        team_key = matched_team if matched_team else team

        return self.TEAM_STRENGTHS.get(team_key, {
            'offense': 50,  # Average for FBS
            'defense': 50,
            'form': 2
        })

    def _calculate_momentum_adjustment(self, home_team: str, away_team: str) -> float:
        """
        Calculate momentum adjustment based on recent form.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Probability adjustment (-0.12 to +0.12)
        """
        home_strength = self.get_team_strength(home_team)
        away_strength = self.get_team_strength(away_team)

        form_diff = home_strength.get('form', 2) - away_strength.get('form', 2)

        # Higher variance in college football
        impact = (form_diff / 5.0) * 0.10
        return max(-0.12, min(0.12, impact))

    def is_rivalry_game(self, team1: str, team2: str) -> Optional[str]:
        """
        Check if game is a rivalry matchup.

        Args:
            team1: First team name
            team2: Second team name

        Returns:
            Rivalry name if it exists, None otherwise
        """
        # Check both orderings
        return (
            self.RIVALRIES.get((team1, team2)) or
            self.RIVALRIES.get((team2, team1))
        )

    def predict_winner(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict winner of NCAA football game.

        Args:
            home_team: Home team name (may include mascot)
            away_team: Away team name (may include mascot)
            game_date: Date/time of game
            **kwargs: Additional parameters (crowd_size, bowl_game, etc.)

        Returns:
            Prediction dictionary with winner, probability, confidence, etc.
        """
        # Use fuzzy matching to find best match for team names
        # ESPN sends "Florida State Seminoles", we need to match to "Florida State"
        home_team_matched = self._find_best_team_match(home_team, self.elo_ratings)
        away_team_matched = self._find_best_team_match(away_team, self.elo_ratings)

        # Use original names for display, matched names for lookups
        home_team_display = home_team
        away_team_display = away_team

        # Use matched names if found, otherwise use original
        home_team_lookup = home_team_matched if home_team_matched else home_team
        away_team_lookup = away_team_matched if away_team_matched else away_team

        # Check cache first (use matched names)
        cache_key = self.create_cache_key(home_team_lookup, away_team_lookup, game_date)
        cached = self.get_cached_prediction(cache_key)
        if cached:
            # Update cached prediction with display names
            cached['winner'] = home_team_display if cached.get('winner') == home_team_lookup else away_team_display
            return cached

        # Calculate features (use matched names)
        features = self.calculate_features(home_team_lookup, away_team_lookup, game_date)

        # Get Elo-based probability (use matched names)
        home_elo = self.elo_ratings.get(home_team_lookup, self.ELO_BASE)
        away_elo = self.elo_ratings.get(away_team_lookup, self.ELO_BASE)

        # Log if using default Elo (indicates no match found)
        if home_elo == self.ELO_BASE and home_team_matched is None:
            self.logger.warning(f"Using default Elo for home team '{home_team}' (no match found)")
        if away_elo == self.ELO_BASE and away_team_matched is None:
            self.logger.warning(f"Using default Elo for away team '{away_team}' (no match found)")

        # Adjust for conference strength (use matched names)
        home_conf_power = self.get_conference_power(home_team_lookup)
        away_conf_power = self.get_conference_power(away_team_lookup)

        # Conference adjustment to Elo
        conf_diff = (home_conf_power - away_conf_power) * 100  # ~100 Elo points max
        adjusted_home_elo = home_elo + conf_diff

        # Add home field advantage (larger in college due to crowds)
        home_crowd_size = kwargs.get('crowd_size', 60000)  # Default average
        crowd_factor = min(home_crowd_size / 100000, 1.5)  # Cap at 1.5x
        effective_hfa = self.HOME_FIELD_ADVANTAGE * crowd_factor

        adjusted_home_elo += (effective_hfa * 25)  # Convert points to Elo

        # Calculate base probability
        base_prob = self._calculate_elo_win_prob(adjusted_home_elo, away_elo)

        # Adjust for recruiting (talent gap) - use matched names
        recruiting_adjustment = self._calculate_recruiting_impact(home_team_lookup, away_team_lookup)

        # Adjust for momentum - use matched names
        momentum_adjustment = self._calculate_momentum_adjustment(home_team_lookup, away_team_lookup)

        adjusted_prob = base_prob + recruiting_adjustment + momentum_adjustment

        # Adjust for rivalry games (closer, more intense) - use matched names
        if self.is_rivalry_game(home_team_lookup, away_team_lookup):
            # Regress toward 50%
            adjusted_prob = 0.5 + (adjusted_prob - 0.5) * 0.75
            is_rivalry = True
        else:
            is_rivalry = False

        # Bowl game adjustment (neutral site)
        if kwargs.get('is_bowl_game', False):
            # Remove home field advantage
            adjusted_prob -= (effective_hfa / 100.0)  # Reverse HFA points
            is_bowl = True
        else:
            is_bowl = False

        # Ensure probability is in valid range
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))

        # Determine winner (use display names for output)
        winner = home_team_display if adjusted_prob >= 0.5 else away_team_display
        win_prob = adjusted_prob if adjusted_prob >= 0.5 else (1 - adjusted_prob)

        # Calculate confidence
        confidence = self.get_confidence(adjusted_prob)

        # Calculate predicted spread
        spread = self.get_spread(adjusted_prob, self.SPREAD_FACTOR)

        # Build prediction result
        prediction = {
            'winner': winner,
            'probability': win_prob,
            'confidence': confidence,
            'spread': spread,
            'method': 'ncaa_ensemble',
            'features': features,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_prob': adjusted_prob,
            'away_prob': 1 - adjusted_prob,
            'adjustments': {
                'home_field': effective_hfa,
                'conference_diff': conf_diff,
                'recruiting_impact': recruiting_adjustment,
                'is_rivalry': is_rivalry,
                'is_bowl_game': is_bowl,
                'crowd_size': home_crowd_size
            },
            'explanation': self._generate_explanation(
                winner,
                win_prob,
                home_team_display,
                away_team_display,
                features,
                is_rivalry
            )
        }

        # Cache the prediction
        self.cache_prediction(cache_key, prediction)

        return prediction

    def calculate_features(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate prediction features for NCAA matchup.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date/time

        Returns:
            Dictionary of calculated features
        """
        features = {
            'home_elo': self.elo_ratings.get(home_team, self.ELO_BASE),
            'away_elo': self.elo_ratings.get(away_team, self.ELO_BASE),
            'elo_diff': self.elo_ratings.get(home_team, self.ELO_BASE) - self.elo_ratings.get(away_team, self.ELO_BASE),
            'home_field_advantage': self.HOME_FIELD_ADVANTAGE,
            'home_conf_power': self.get_conference_power(home_team),
            'away_conf_power': self.get_conference_power(away_team),
            'home_recruiting': self.get_recruiting_score(home_team),
            'away_recruiting': self.get_recruiting_score(away_team),
            'recruiting_diff': self.get_recruiting_score(home_team) - self.get_recruiting_score(away_team),
            'is_rivalry': 1.0 if self.is_rivalry_game(home_team, away_team) else 0.0,
        }

        # Add team stats if available
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        if home_stats:
            features['home_points_per_game'] = home_stats.get('points_per_game', 0)
            features['home_points_allowed'] = home_stats.get('points_allowed', 0)

        if away_stats:
            features['away_points_per_game'] = away_stats.get('points_per_game', 0)
            features['away_points_allowed'] = away_stats.get('points_allowed', 0)

        return features

    def _calculate_elo_win_prob(self, elo_a: float, elo_b: float) -> float:
        """
        Calculate win probability from Elo ratings.

        Args:
            elo_a: Team A's Elo rating
            elo_b: Team B's Elo rating

        Returns:
            Probability that Team A wins (0.0-1.0)
        """
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

    def _calculate_recruiting_impact(self, home_team: str, away_team: str) -> float:
        """
        Calculate impact of recruiting talent gap.

        Recruiting makes a bigger difference in college than NFL.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Probability adjustment (-0.15 to +0.15)
        """
        home_recruiting = self.get_recruiting_score(home_team)
        away_recruiting = self.get_recruiting_score(away_team)

        # Difference in recruiting scores
        diff = home_recruiting - away_recruiting

        # Convert to probability impact (max ±15%)
        # A 20-point recruiting gap = ~10% probability swing
        impact = (diff / 20.0) * 0.10

        # Cap at ±0.15
        return max(-0.15, min(0.15, impact))

    def _generate_explanation(
        self,
        winner: str,
        probability: float,
        home_team: str,
        away_team: str,
        features: Dict,
        is_rivalry: bool
    ) -> str:
        """
        Generate human-readable explanation of prediction.

        Args:
            winner: Predicted winner
            probability: Win probability
            home_team: Home team name
            away_team: Away team name
            features: Features dictionary
            is_rivalry: Whether this is a rivalry game

        Returns:
            Explanation string
        """
        explanation = f"{winner} predicted to win with {self.format_probability(probability)} probability. "

        # Explain conference strength
        home_conf_power = features.get('home_conf_power', 0.5)
        away_conf_power = features.get('away_conf_power', 0.5)

        if abs(home_conf_power - away_conf_power) > 0.15:
            stronger_conf = home_team if home_conf_power > away_conf_power else away_team
            explanation += f"{stronger_conf} plays in a stronger conference. "

        # Explain recruiting gap
        recruiting_diff = features.get('recruiting_diff', 0)
        if abs(recruiting_diff) > 10:
            better_recruiting = home_team if recruiting_diff > 0 else away_team
            explanation += f"{better_recruiting} has superior recruiting (talent advantage). "

        # Rivalry note
        if is_rivalry:
            explanation += "Rivalry game typically more competitive. "

        # Home field advantage
        if winner == home_team:
            explanation += f"Home crowd advantage worth ~{self.HOME_FIELD_ADVANTAGE} points."
        else:
            explanation += f"{away_team} expected to overcome home crowd advantage."

        # Generate base template explanation
        template_explanation = explanation
        
        # Enhance with LLM if available
        try:
            enhancer = get_llm_enhancer()
            loser = away_team if winner == home_team else home_team
            
            # Prepare context for LLM
            context = {
                'home_team': home_team,
                'away_team': away_team,
                'is_rivalry': is_rivalry,
                'home_conf_power': features.get('home_conf_power'),
                'away_conf_power': features.get('away_conf_power'),
                'recruiting_diff': features.get('recruiting_diff'),
                'home_field_advantage': self.HOME_FIELD_ADVANTAGE
            }
            
            return enhancer.enhance_explanation(
                sport="NCAA Football",
                winner=winner,
                loser=loser,
                probability=probability,
                features=features,
                context=context,
                template_explanation=template_explanation
            )
        except Exception as e:
            self.logger.warning(f"Failed to enhance explanation with LLM: {e}")
            return template_explanation

    def update_elo_ratings(
        self,
        winner: str,
        loser: str,
        winner_score: int,
        loser_score: int,
        home_team: str
    ):
        """
        Update Elo ratings after a game result.

        Args:
            winner: Winning team name
            loser: Losing team name
            winner_score: Winner's score
            loser_score: Loser's score
            home_team: Which team was home
        """
        # Get current ratings
        winner_elo = self.elo_ratings.get(winner, self.ELO_BASE)
        loser_elo = self.elo_ratings.get(loser, self.ELO_BASE)

        # Calculate expected win probability
        if winner == home_team:
            winner_expected = self._calculate_elo_win_prob(winner_elo + 88, loser_elo)  # ~3.5 point HFA
        else:
            winner_expected = self._calculate_elo_win_prob(winner_elo, loser_elo + 88)

        # Calculate margin of victory multiplier (higher in college due to score variance)
        point_diff = abs(winner_score - loser_score)
        mov_multiplier = math.log(point_diff + 1) * (2.5 / ((winner_elo - loser_elo) * 0.001 + 2.5))

        # Update ratings
        rating_change = self.ELO_K_FACTOR * mov_multiplier * (1 - winner_expected)

        self.elo_ratings[winner] = winner_elo + rating_change
        self.elo_ratings[loser] = loser_elo - rating_change

        self.logger.info(f"Updated Elo: {winner} +{rating_change:.1f}, {loser} -{rating_change:.1f}")

        # Save updated ratings
        self._save_elo_ratings()

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """
        Get current season statistics for a team.

        Args:
            team_name: Name of the team

        Returns:
            Dictionary of team statistics
        """
        # Return cached stats if available
        if team_name in self.team_stats:
            return self.team_stats[team_name]

        # TODO: Load from database
        return {}

    def get_historical_matchups(
        self,
        team1: str,
        team2: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical head-to-head matchups.

        Args:
            team1: First team name
            team2: Second team name
            limit: Maximum number of games to return

        Returns:
            List of historical game dictionaries
        """
        # TODO: Load from database
        return []

    def get_top_25_rankings(self) -> List[Dict[str, Any]]:
        """
        Get current Top 25 rankings based on Elo.

        Returns:
            List of teams ranked by Elo rating
        """
        ranked_teams = sorted(
            self.elo_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            {
                'rank': idx + 1,
                'team': team,
                'elo': elo,
                'conference': self.conference_map.get(team, 'Unknown'),
                'recruiting': self.get_recruiting_score(team)
            }
            for idx, (team, elo) in enumerate(ranked_teams[:25])
        ]

    def predict_playoff_chances(self, team: str) -> float:
        """
        Estimate playoff/bowl eligibility chances.

        Args:
            team: Team name

        Returns:
            Probability of making playoffs/bowl (0.0-1.0)
        """
        # Simple heuristic based on Elo
        team_elo = self.elo_ratings.get(team, self.ELO_BASE)

        if team_elo > 1700:
            return 0.95  # Elite team
        elif team_elo > 1600:
            return 0.75  # Very good team
        elif team_elo > 1500:
            return 0.50  # Average team
        elif team_elo > 1400:
            return 0.25  # Below average
        else:
            return 0.10  # Struggling team
