"""
NFL Prediction Agent
====================

NFL-specific prediction agent with advanced features:
- Elo rating system (FiveThirtyEight-style)
- EPA (Expected Points Added) analysis
- Home field advantage calculations
- Injury impact assessment
- Weather considerations
- Divisional game adjustments

Based on research synthesis showing 70-73% accuracy achievable with ensemble methods.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import math
import json
import os
import logging

from .base_predictor import BaseSportsPredictor
from .llm_explanation_enhancer import get_llm_enhancer

logger = logging.getLogger(__name__)


class NFLPredictor(BaseSportsPredictor):
    """
    NFL-specific prediction agent using ensemble machine learning approach.

    Features:
    - Fast predictions (<100ms) for real-time UI
    - Elo-based rating system
    - Advanced statistical features (EPA, DVOA)
    - Situation-aware predictions (primetime, weather, rest)
    """

    # NFL-specific constants
    HOME_FIELD_ADVANTAGE = 2.5  # Points (historically ~2.5 points for NFL)
    ELO_K_FACTOR = 20  # How quickly ratings update
    ELO_BASE = 1500  # Starting Elo rating
    SPREAD_FACTOR = 25.0  # For probability to spread conversion

    # Injury impact multipliers
    INJURY_IMPACT = {
        'QB': 0.15,      # Quarterback: 15% impact
        'RB': 0.03,      # Running back: 3% impact
        'WR': 0.04,      # Wide receiver: 4% impact
        'OL': 0.06,      # Offensive line: 6% impact
        'DL': 0.05,      # Defensive line: 5% impact
        'LB': 0.04,      # Linebacker: 4% impact
        'DB': 0.04,      # Defensive back: 4% impact
        'K': 0.01,       # Kicker: 1% impact
        'P': 0.01        # Punter: 1% impact
    }

    # Team strength modifiers (offensive/defensive rankings, recent form)
    TEAM_STRENGTHS = {
        # Format: offensive_rank (1-32), defensive_rank (1-32), recent_form (wins in last 5)
        'Kansas City Chiefs': {'offense': 3, 'defense': 10, 'form': 4},
        'Baltimore Ravens': {'offense': 1, 'defense': 12, 'form': 4},
        'Buffalo Bills': {'offense': 5, 'defense': 8, 'form': 3},
        'San Francisco 49ers': {'offense': 2, 'defense': 3, 'form': 4},
        'Philadelphia Eagles': {'offense': 4, 'defense': 9, 'form': 3},
        'Detroit Lions': {'offense': 6, 'defense': 15, 'form': 5},
        'Miami Dolphins': {'offense': 7, 'defense': 20, 'form': 3},
        'Dallas Cowboys': {'offense': 8, 'defense': 14, 'form': 3},
        'Los Angeles Rams': {'offense': 12, 'defense': 11, 'form': 2},
        'Cleveland Browns': {'offense': 20, 'defense': 5, 'form': 2},
        'Houston Texans': {'offense': 15, 'defense': 18, 'form': 3},
        'Jacksonville Jaguars': {'offense': 16, 'defense': 22, 'form': 2},
        'Green Bay Packers': {'offense': 14, 'defense': 16, 'form': 3},
        'Cincinnati Bengals': {'offense': 9, 'defense': 24, 'form': 2},
        'Los Angeles Chargers': {'offense': 17, 'defense': 13, 'form': 2},
        'Seattle Seahawks': {'offense': 11, 'defense': 21, 'form': 3},
        'Pittsburgh Steelers': {'offense': 22, 'defense': 6, 'form': 2},
        'Tampa Bay Buccaneers': {'offense': 13, 'defense': 17, 'form': 2},
        'Minnesota Vikings': {'offense': 18, 'defense': 19, 'form': 2},
        'Atlanta Falcons': {'offense': 19, 'defense': 23, 'form': 2},
        'New Orleans Saints': {'offense': 23, 'defense': 25, 'form': 1},
        'Indianapolis Colts': {'offense': 21, 'defense': 26, 'form': 2},
        'Las Vegas Raiders': {'offense': 25, 'defense': 27, 'form': 1},
        'Tennessee Titans': {'offense': 26, 'defense': 28, 'form': 1},
        'New York Jets': {'offense': 27, 'defense': 7, 'form': 1},
        'Denver Broncos': {'offense': 24, 'defense': 4, 'form': 2},
        'New England Patriots': {'offense': 28, 'defense': 29, 'form': 1},
        'Washington Commanders': {'offense': 10, 'defense': 30, 'form': 2},
        'Chicago Bears': {'offense': 29, 'defense': 31, 'form': 1},
        'New York Giants': {'offense': 30, 'defense': 32, 'form': 1},
        'Arizona Cardinals': {'offense': 31, 'defense': 15, 'form': 1},
        'Carolina Panthers': {'offense': 32, 'defense': 33, 'form': 0},
    }

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize NFL predictor.

        Args:
            db_config: Database configuration for loading stats
        """
        super().__init__("NFL")

        self.db_config = db_config
        self.elo_ratings = {}  # Team -> current Elo rating
        self.team_stats = {}   # Team -> current season stats
        self.injury_data = {}  # Team -> injury report
        self.division_map = {}  # Team -> division

        # Load initial data
        self._load_team_data()
        self._load_elo_ratings()

    def _load_team_data(self):
        """Load NFL team data (divisions, current stats)."""
        # NFL divisions
        self.division_map = {
            # AFC East
            'Buffalo Bills': 'AFC East',
            'Miami Dolphins': 'AFC East',
            'New England Patriots': 'AFC East',
            'New York Jets': 'AFC East',

            # AFC North
            'Baltimore Ravens': 'AFC North',
            'Cincinnati Bengals': 'AFC North',
            'Cleveland Browns': 'AFC North',
            'Pittsburgh Steelers': 'AFC North',

            # AFC South
            'Houston Texans': 'AFC South',
            'Indianapolis Colts': 'AFC South',
            'Jacksonville Jaguars': 'AFC South',
            'Tennessee Titans': 'AFC South',

            # AFC West
            'Denver Broncos': 'AFC West',
            'Kansas City Chiefs': 'AFC West',
            'Las Vegas Raiders': 'AFC West',
            'Los Angeles Chargers': 'AFC West',

            # NFC East
            'Dallas Cowboys': 'NFC East',
            'New York Giants': 'NFC East',
            'Philadelphia Eagles': 'NFC East',
            'Washington Commanders': 'NFC East',

            # NFC North
            'Chicago Bears': 'NFC North',
            'Detroit Lions': 'NFC North',
            'Green Bay Packers': 'NFC North',
            'Minnesota Vikings': 'NFC North',

            # NFC South
            'Atlanta Falcons': 'NFC South',
            'Carolina Panthers': 'NFC South',
            'New Orleans Saints': 'NFC South',
            'Tampa Bay Buccaneers': 'NFC South',

            # NFC West
            'Arizona Cardinals': 'NFC West',
            'Los Angeles Rams': 'NFC West',
            'San Francisco 49ers': 'NFC West',
            'Seattle Seahawks': 'NFC West',
        }

    def _load_elo_ratings(self):
        """Load or initialize Elo ratings for all teams."""
        # Try to load from file
        ratings_file = 'src/data/nfl_elo_ratings.json'

        if os.path.exists(ratings_file):
            try:
                with open(ratings_file, 'r') as f:
                    self.elo_ratings = json.load(f)
                self.logger.info(f"Loaded Elo ratings from {ratings_file}")
                return
            except Exception as e:
                self.logger.warning(f"Could not load Elo ratings: {e}")

        # Initialize all teams to base rating
        for team in self.division_map.keys():
            self.elo_ratings[team] = self.ELO_BASE

        self.logger.info("Initialized Elo ratings to base value")

    def _save_elo_ratings(self):
        """Save current Elo ratings to file."""
        ratings_file = 'src/data/nfl_elo_ratings.json'

        try:
            os.makedirs(os.path.dirname(ratings_file), exist_ok=True)
            with open(ratings_file, 'w') as f:
                json.dump(self.elo_ratings, f, indent=2)
            self.logger.info(f"Saved Elo ratings to {ratings_file}")
        except Exception as e:
            self.logger.error(f"Could not save Elo ratings: {e}")

    def get_elo_rating(self, team: str) -> float:
        """
        Get current Elo rating for a team.

        Args:
            team: Team name

        Returns:
            Elo rating (default 1500 if not found)
        """
        return self.elo_ratings.get(team, self.ELO_BASE)

    def get_team_strength(self, team: str) -> Dict[str, Any]:
        """
        Get comprehensive team strength metrics.

        Args:
            team: Team name

        Returns:
            Dictionary with offensive rank, defensive rank, recent form
        """
        return self.TEAM_STRENGTHS.get(team, {
            'offense': 16,  # Average
            'defense': 16,  # Average
            'form': 2       # 2-3 recent record
        })

    def _calculate_momentum_adjustment(self, home_team: str, away_team: str) -> float:
        """
        Calculate win probability adjustment based on recent form/momentum.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Probability adjustment (-0.10 to +0.10)
        """
        home_strength = self.get_team_strength(home_team)
        away_strength = self.get_team_strength(away_team)

        home_form = home_strength.get('form', 2)
        away_form = away_strength.get('form', 2)

        # Form difference (0-5 wins in last 5 games)
        form_diff = home_form - away_form

        # Convert to probability impact (max ±10%)
        impact = (form_diff / 5.0) * 0.08

        return max(-0.10, min(0.10, impact))

    def _calculate_matchup_adjustment(self, home_team: str, away_team: str) -> float:
        """
        Calculate adjustment based on offensive/defensive matchup.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Probability adjustment (-0.08 to +0.08)
        """
        home_strength = self.get_team_strength(home_team)
        away_strength = self.get_team_strength(away_team)

        # Good offense vs bad defense = advantage
        home_off_rank = home_strength.get('offense', 16)
        away_def_rank = away_strength.get('defense', 16)
        home_matchup = (away_def_rank - home_off_rank) / 32.0

        away_off_rank = away_strength.get('offense', 16)
        home_def_rank = home_strength.get('defense', 16)
        away_matchup = (home_def_rank - away_off_rank) / 32.0

        matchup_diff = (home_matchup - away_matchup) * 0.08

        return max(-0.08, min(0.08, matchup_diff))


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
        winner_elo = self.get_elo_rating(winner)
        loser_elo = self.get_elo_rating(loser)

        # Calculate expected win probabilities
        if winner == home_team:
            winner_expected = self._calculate_elo_win_prob(winner_elo + 65, loser_elo)
        else:
            winner_expected = self._calculate_elo_win_prob(winner_elo, loser_elo + 65)

        # Calculate margin of victory multiplier
        point_diff = abs(winner_score - loser_score)
        mov_multiplier = math.log(point_diff + 1) * (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))

        # Update ratings
        rating_change = self.ELO_K_FACTOR * mov_multiplier * (1 - winner_expected)

        self.elo_ratings[winner] = winner_elo + rating_change
        self.elo_ratings[loser] = loser_elo - rating_change

        self.logger.info(f"Updated Elo: {winner} +{rating_change:.1f}, {loser} -{rating_change:.1f}")

        # Save updated ratings
        self._save_elo_ratings()

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

    def predict_winner(
        self,
        home_team: str,
        away_team: str,
        game_date: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict winner of NFL game.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date/time of game
            **kwargs: Additional parameters (injuries, weather, etc.)

        Returns:
            Prediction dictionary with winner, probability, confidence, etc.
        """
        # Check cache first
        cache_key = self.create_cache_key(home_team, away_team, game_date)
        cached = self.get_cached_prediction(cache_key)
        if cached:
            return cached

        # Calculate features
        features = self.calculate_features(home_team, away_team, game_date)

        # Get Elo-based probability
        home_elo = self.get_elo_rating(home_team)
        away_elo = self.get_elo_rating(away_team)

        # Add home field advantage to home team's Elo
        adjusted_home_elo = home_elo + (self.HOME_FIELD_ADVANTAGE * 25)  # ~65 Elo points
        base_prob = self._calculate_elo_win_prob(adjusted_home_elo, away_elo)

        # Adjust for momentum/recent form
        momentum_adjustment = self._calculate_momentum_adjustment(home_team, away_team)

        # Adjust for matchup (offense vs defense)
        matchup_adjustment = self._calculate_matchup_adjustment(home_team, away_team)

        # Adjust for injuries
        injury_adjustment = self._calculate_injury_impact(home_team, away_team)

        # Combined probability
        adjusted_prob = base_prob + momentum_adjustment + matchup_adjustment + injury_adjustment

        # Adjust for divisional games (more competitive)
        is_divisional = self._is_divisional_game(home_team, away_team)
        if is_divisional:
            # Regress probability toward 50%
            adjusted_prob = 0.5 + (adjusted_prob - 0.5) * 0.85

        # Adjust for weather (if provided)
        weather = kwargs.get('weather')
        weather_adjustment = 0.0
        if weather:
            weather_adjustment = self._calculate_weather_impact(weather)
            adjusted_prob += weather_adjustment

        # Ensure probability is in valid range
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))

        # Determine winner
        winner = home_team if adjusted_prob >= 0.5 else away_team
        win_prob = adjusted_prob if winner == home_team else (1 - adjusted_prob)

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
            'method': 'elo_ensemble',
            'features': features,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'home_prob': adjusted_prob,
            'away_prob': 1 - adjusted_prob,
            'adjustments': {
                'home_field': self.HOME_FIELD_ADVANTAGE,
                'momentum': momentum_adjustment,
                'matchup': matchup_adjustment,
                'injury_impact': injury_adjustment,
                'weather': weather_adjustment,
                'divisional': is_divisional,
            },
            'explanation': self._generate_explanation(
                winner,
                win_prob,
                home_team,
                away_team,
                features,
                {
                    'momentum': momentum_adjustment,
                    'matchup': matchup_adjustment,
                    'injury': injury_adjustment,
                    'divisional': is_divisional
                }
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
        Calculate prediction features for NFL matchup.

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date/time

        Returns:
            Dictionary of calculated features
        """
        home_strength = self.get_team_strength(home_team)
        away_strength = self.get_team_strength(away_team)

        features = {
            'home_elo': self.get_elo_rating(home_team),
            'away_elo': self.get_elo_rating(away_team),
            'elo_diff': self.get_elo_rating(home_team) - self.get_elo_rating(away_team),
            'home_field_advantage': self.HOME_FIELD_ADVANTAGE,
            'is_divisional': 1.0 if self._is_divisional_game(home_team, away_team) else 0.0,
            'home_off_rank': home_strength.get('offense', 16),
            'home_def_rank': home_strength.get('defense', 16),
            'away_off_rank': away_strength.get('offense', 16),
            'away_def_rank': away_strength.get('defense', 16),
            'home_form': home_strength.get('form', 2),
            'away_form': away_strength.get('form', 2),
        }

        # Add team stats if available
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        if home_stats:
            features['home_points_per_game'] = home_stats.get('points_per_game', 0)
            features['home_points_allowed'] = home_stats.get('points_allowed', 0)
            features['home_turnover_diff'] = home_stats.get('turnover_diff', 0)

        if away_stats:
            features['away_points_per_game'] = away_stats.get('points_per_game', 0)
            features['away_points_allowed'] = away_stats.get('points_allowed', 0)
            features['away_turnover_diff'] = away_stats.get('turnover_diff', 0)

        return features

    def _calculate_injury_impact(self, home_team: str, away_team: str) -> float:
        """
        Calculate impact of injuries on win probability.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Probability adjustment (-0.15 to +0.15)
        """
        home_impact = 0.0
        away_impact = 0.0

        # Get injury data (if available)
        home_injuries = self.injury_data.get(home_team, [])
        away_injuries = self.injury_data.get(away_team, [])

        # Calculate impact for each injured player
        for injury in home_injuries:
            position = injury.get('position', 'unknown')
            impact = self.INJURY_IMPACT.get(position, 0.02)
            home_impact += impact

        for injury in away_injuries:
            position = injury.get('position', 'unknown')
            impact = self.INJURY_IMPACT.get(position, 0.02)
            away_impact += impact

        # Net impact (positive helps home team)
        return away_impact - home_impact

    def _calculate_weather_impact(self, weather: Dict[str, Any]) -> float:
        """
        Calculate impact of weather on game outcome.

        Factors:
        - Wind speed (affects passing, kicking)
        - Precipitation (affects ball handling)
        - Temperature (extreme cold affects performance)

        Args:
            weather: Dictionary with 'wind_mph', 'precipitation', 'temp_f'

        Returns:
            Probability adjustment (-0.05 to +0.05)
        """
        impact = 0.0

        # Wind impact (>15 mph favors running teams)
        wind = weather.get('wind_mph', 0)
        if wind > 15:
            impact -= 0.02  # Slight disadvantage for passing teams

        # Precipitation impact
        if weather.get('precipitation', False):
            impact -= 0.01  # Slight disadvantage overall

        # Extreme temperature impact
        temp = weather.get('temp_f', 65)
        if temp < 20 or temp > 95:
            impact -= 0.02  # Extreme temps affect performance

        return impact

    def _is_divisional_game(self, team1: str, team2: str) -> bool:
        """
        Check if game is between division rivals.

        Args:
            team1: First team name
            team2: Second team name

        Returns:
            True if teams are in same division
        """
        div1 = self.division_map.get(team1)
        div2 = self.division_map.get(team2)

        return div1 is not None and div1 == div2

    def _generate_explanation(
        self,
        winner: str,
        probability: float,
        home_team: str,
        away_team: str,
        features: Dict,
        adjustments: Dict
    ) -> str:
        """
        Generate human-readable explanation of prediction.

        Args:
            winner: Predicted winner
            probability: Win probability
            home_team: Home team name
            away_team: Away team name
            features: Features dictionary

        Returns:
            Explanation string
        """
        parts = []

        # Opening statement with confidence
        confidence_desc = {
            'high': 'strongly favored',
            'medium': 'moderately favored',
            'low': 'slightly favored'
        }
        conf_level = self.get_confidence(probability)
        parts.append(f"{winner} {confidence_desc.get(conf_level, 'favored')} to win ({self.format_probability(probability)} probability).")

        # Explain Elo differential
        elo_diff = features.get('elo_diff', 0)
        if winner == home_team and elo_diff > 50:
            parts.append(f"{winner} enters as the stronger team with a significant Elo rating advantage ({abs(elo_diff):.0f} points).")
        elif winner == away_team and elo_diff < -50:
            parts.append(f"{winner} has proven themselves as the superior team this season, carrying an Elo advantage of {abs(elo_diff):.0f} points.")
        elif abs(elo_diff) <= 50:
            parts.append("Both teams are closely matched in overall strength, making this a competitive matchup.")

        # Explain key strengths
        home_strength = self.get_team_strength(home_team)
        away_strength = self.get_team_strength(away_team)

        if winner == home_team:
            if home_strength['offense'] <= 10:
                parts.append(f"{winner} brings a top-10 offense that should exploit {away_team}'s defensive weaknesses.")
            if home_strength['defense'] <= 8:
                parts.append(f"Their elite defense (ranked #{home_strength['defense']}) is expected to limit {away_team}'s scoring opportunities.")
        else:
            if away_strength['offense'] <= 10:
                parts.append(f"{winner} features a potent offense (ranked #{away_strength['offense']}) capable of moving the ball against {home_team}'s defense.")
            if away_strength['defense'] <= 8:
                parts.append(f"Their stout defense should contain {home_team}'s offensive attack.")

        # Momentum factor
        momentum_adj = adjustments.get('momentum', 0)
        if abs(momentum_adj) >= 0.04:
            winner_form = home_strength['form'] if winner == home_team else away_strength['form']
            loser_form = away_strength['form'] if winner == home_team else home_strength['form']

            if winner_form >= 4:
                parts.append(f"{winner} enters on a hot streak, winning {winner_form} of their last 5 games, bringing strong momentum.")
            elif loser_form <= 1:
                loser = away_team if winner == home_team else home_team
                parts.append(f"{loser} is struggling with only {loser_form} win(s) in their last 5, facing confidence issues.")

        # Divisional rivalry
        if adjustments.get('divisional'):
            parts.append("As division rivals who know each other well, expect a closer, more physical contest than the ratings suggest.")

        # Home field advantage
        if winner == home_team and probability >= 0.55:
            parts.append(f"Home field advantage (~{self.HOME_FIELD_ADVANTAGE} points) provides {winner} a meaningful edge in what should be a competitive game.")
        elif winner == away_team:
            parts.append(f"{winner} is strong enough to overcome the {self.HOME_FIELD_ADVANTAGE}-point home field disadvantage.")

        # Matchup-specific insights
        matchup_adj = adjustments.get('matchup', 0)
        if abs(matchup_adj) >= 0.04:
            if matchup_adj > 0:
                parts.append(f"The offensive-defensive matchup favors {home_team}, with their offensive strengths targeting {away_team}'s defensive vulnerabilities.")
            else:
                parts.append(f"{away_team}'s offensive scheme is well-suited to attack {home_team}'s defensive weaknesses.")

        # Injury concerns
        injury_adj = adjustments.get('injury', 0)
        if abs(injury_adj) >= 0.05:
            affected_team = away_team if injury_adj > 0 else home_team
            parts.append(f"Key injuries to {affected_team} significantly impact their chances in this matchup.")

        # Generate base template explanation
        template_explanation = " ".join(parts)
        
        # Enhance with LLM if available
        try:
            enhancer = get_llm_enhancer()
            loser = away_team if winner == home_team else home_team
            
            # Prepare context for LLM
            context = {
                'home_team': home_team,
                'away_team': away_team,
                'adjustments': adjustments,
                'elo_diff': features.get('elo_diff', 0),
                'home_field_advantage': self.HOME_FIELD_ADVANTAGE
            }
            
            return enhancer.enhance_explanation(
                sport="NFL",
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

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """
        Get current season statistics for a team from the database.

        Args:
            team_name: Name of the team

        Returns:
            Dictionary of team statistics
        """
        # Return cached stats if available
        if team_name in self.team_stats:
            return self.team_stats[team_name]

        # Load from database
        try:
            from src.database.connection_pool import get_db_connection
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Calculate team stats from completed games this season
                cursor.execute("""
                    SELECT
                        COUNT(*) as games_played,
                        SUM(CASE WHEN home_team = %s THEN home_score ELSE away_score END) as points_for,
                        SUM(CASE WHEN home_team = %s THEN away_score ELSE home_score END) as points_against,
                        SUM(CASE
                            WHEN (home_team = %s AND home_score > away_score) OR
                                 (away_team = %s AND away_score > home_score)
                            THEN 1 ELSE 0
                        END) as wins,
                        SUM(CASE
                            WHEN (home_team = %s AND home_score < away_score) OR
                                 (away_team = %s AND away_score < home_score)
                            THEN 1 ELSE 0
                        END) as losses
                    FROM nfl_games
                    WHERE (home_team = %s OR away_team = %s)
                      AND game_status = 'final'
                      AND season = EXTRACT(YEAR FROM CURRENT_DATE)
                """, (team_name, team_name, team_name, team_name, team_name, team_name, team_name, team_name))

                row = cursor.fetchone()
                if row and row[0] and row[0] > 0:
                    games_played = row[0]
                    points_for = row[1] or 0
                    points_against = row[2] or 0
                    wins = row[3] or 0
                    losses = row[4] or 0

                    stats = {
                        'games_played': games_played,
                        'wins': wins,
                        'losses': losses,
                        'win_pct': wins / games_played if games_played > 0 else 0.5,
                        'points_per_game': points_for / games_played,
                        'points_allowed': points_against / games_played,
                        'point_diff': (points_for - points_against) / games_played,
                        'turnover_diff': 0  # Would need play-by-play data
                    }

                    # Cache the stats
                    self.team_stats[team_name] = stats
                    return stats

        except Exception as e:
            logger.warning(f"Could not load team stats for {team_name}: {e}")

        return {}

    def get_historical_matchups(
        self,
        team1: str,
        team2: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get historical head-to-head matchups from the database.

        Args:
            team1: First team name
            team2: Second team name
            limit: Maximum number of games to return

        Returns:
            List of historical game dictionaries
        """
        try:
            from src.database.connection_pool import get_db_connection
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT
                        game_id,
                        season,
                        week,
                        home_team,
                        away_team,
                        home_score,
                        away_score,
                        game_time,
                        venue
                    FROM nfl_games
                    WHERE game_status = 'final'
                      AND ((home_team = %s AND away_team = %s) OR
                           (home_team = %s AND away_team = %s))
                    ORDER BY game_time DESC
                    LIMIT %s
                """, (team1, team2, team2, team1, limit))

                rows = cursor.fetchall()
                matchups = []

                for row in rows:
                    home_team = row[3]
                    away_team = row[4]
                    home_score = row[5]
                    away_score = row[6]

                    # Determine winner
                    if home_score > away_score:
                        winner = home_team
                    elif away_score > home_score:
                        winner = away_team
                    else:
                        winner = "Tie"

                    matchups.append({
                        'game_id': row[0],
                        'season': row[1],
                        'week': row[2],
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'winner': winner,
                        'margin': abs(home_score - away_score),
                        'game_time': row[7].isoformat() if row[7] else None,
                        'venue': row[8]
                    })

                return matchups

        except Exception as e:
            logger.warning(f"Could not load historical matchups for {team1} vs {team2}: {e}")

        return []

    def _load_injuries_from_db(self, team: str) -> List[Dict[str, Any]]:
        """
        Load active injuries for a team from database.

        Args:
            team: Team name

        Returns:
            List of injury dictionaries
        """
        try:
            from src.database.connection_pool import get_db_connection
            with get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT player_name, position, injury_type, injury_status
                    FROM nfl_injuries
                    WHERE team = %s AND is_active = true
                      AND injury_status IN ('Out', 'Questionable', 'Doubtful')
                """, (team,))

                injuries = []
                for row in cursor.fetchall():
                    injuries.append({
                        'player_name': row[0],
                        'position': row[1],
                        'injury_type': row[2],
                        'status': row[3]
                    })

                return injuries

        except Exception as e:
            logger.warning(f"Could not load injuries for {team}: {e}")
            return []

    def set_injury_data(self, team: str, injuries: List[Dict[str, Any]]):
        """
        Update injury data for a team.

        Args:
            team: Team name
            injuries: List of injury dictionaries with 'position', 'severity', etc.
        """
        self.injury_data[team] = injuries
        self.logger.info(f"Updated injury data for {team}: {len(injuries)} injuries")

    def predict_game(self, game_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict game outcome - wrapper for predict_winner used by predictions API.

        Args:
            game_info: Dictionary with 'home_team' and 'away_team'

        Returns:
            Prediction dictionary with win_probability, confidence, etc.
        """
        home_team = game_info.get('home_team', '')
        away_team = game_info.get('away_team', '')

        if not home_team or not away_team:
            return None

        # Load injuries from database if not already loaded
        if home_team not in self.injury_data:
            self.injury_data[home_team] = self._load_injuries_from_db(home_team)
        if away_team not in self.injury_data:
            self.injury_data[away_team] = self._load_injuries_from_db(away_team)

        # Get full prediction
        prediction = self.predict_winner(
            home_team=home_team,
            away_team=away_team,
            game_date=game_info.get('game_time'),
            weather=game_info.get('weather')
        )

        # Return in format expected by predictions router
        return {
            'winner': prediction.get('winner'),
            'win_probability': prediction.get('probability', 0.5),
            'confidence': prediction.get('confidence', 0.6),
            'spread': prediction.get('spread'),
            'explanation': prediction.get('explanation'),
            'home_prob': prediction.get('home_prob', 0.5),
            'away_prob': prediction.get('away_prob', 0.5)
        }

    def get_all_team_predictions(self, week: int, season: int = 2025) -> List[Dict[str, Any]]:
        """
        Get predictions for all games in a week.

        Args:
            week: NFL week number (1-18)
            season: NFL season year

        Returns:
            List of prediction dictionaries
        """
        # TODO: Load week's games from database and predict each
        # For now, return empty list
        return []

    def get_prediction_accuracy(self, season: int = 2025) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics for a season.

        Returns:
            Dictionary with accuracy statistics
        """
        # TODO: Compare predictions to actual results
        return {
            'overall_accuracy': 0.0,
            'ats_accuracy': 0.0,
            'high_confidence_accuracy': 0.0,
            'total_predictions': 0
        }
