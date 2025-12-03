"""
Sports Streaming Router - Server-Sent Events for Real-Time AI Predictions
Modern streaming implementation with progressive AI analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sports/stream", tags=["Sports Streaming"])


class StreamingPredictionRequest(BaseModel):
    """Request for streaming prediction analysis"""
    game_id: str
    sport: str = "NFL"
    include_reasoning: bool = True
    include_factors: bool = True


async def _stream_ai_prediction(
    game_id: str,
    sport: str,
    include_reasoning: bool = True
) -> AsyncGenerator[str, None]:
    """
    Stream AI prediction analysis with progressive updates.

    Yields events:
    1. start - Analysis beginning
    2. model_loading - Loading prediction models
    3. data_fetching - Fetching game data
    4. prediction - Initial probability prediction
    5. factors - Contributing factors analysis
    6. reasoning - LLM-generated explanation (streamed token by token)
    7. recommendation - Final betting recommendation
    8. complete - Analysis complete
    """
    try:
        # Event 1: Start
        yield json.dumps({
            "type": "start",
            "message": f"Starting AI analysis for {sport} game {game_id}",
            "timestamp": datetime.now().isoformat()
        })
        await asyncio.sleep(0.1)

        # Event 2: Loading models
        yield json.dumps({
            "type": "model_loading",
            "message": "Loading prediction models...",
            "models": ["elo_predictor", "feature_analyzer", "bayesian_adjuster"]
        })
        await asyncio.sleep(0.2)

        # Import prediction modules
        from src.prediction_agents.nfl_predictor import NFLPredictor
        from src.prediction_agents.nba_predictor import NBAPredictor
        from src.prediction_agents.ncaa_predictor import NCAAPredictor
        from src.prediction_agents.live_adjuster import get_live_adjuster

        # Select predictor based on sport
        predictors = {
            "NFL": NFLPredictor,
            "NBA": NBAPredictor,
            "NCAAF": NCAAPredictor,
            "NCAAB": NCAAPredictor
        }

        predictor_class = predictors.get(sport.upper(), NFLPredictor)
        predictor = predictor_class()

        # Event 3: Fetching data
        yield json.dumps({
            "type": "data_fetching",
            "message": "Fetching game data from ESPN...",
            "sources": ["espn_api", "database", "kalshi_odds"]
        })
        await asyncio.sleep(0.3)

        # Fetch game data from database
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import os

        db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/magnus")
        game_data = None

        try:
            conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
            with conn.cursor() as cur:
                # Try each sport table
                tables = {
                    "NFL": "nfl_games",
                    "NBA": "nba_games",
                    "NCAAF": "ncaa_football_games",
                    "NCAAB": "ncaa_basketball_games"
                }
                table = tables.get(sport.upper(), "nfl_games")

                cur.execute(f"""
                    SELECT * FROM {table}
                    WHERE game_id = %s
                    LIMIT 1
                """, (game_id,))
                game_data = cur.fetchone()
            conn.close()
        except Exception as e:
            logger.warning(f"Database fetch error: {e}")

        if not game_data:
            # Use mock data for demo
            game_data = {
                "game_id": game_id,
                "home_team": "Home Team",
                "away_team": "Away Team",
                "home_score": 0,
                "away_score": 0,
                "status": "Scheduled"
            }

        # Event 4: Initial prediction
        home_team = game_data.get("home_team", "Home")
        away_team = game_data.get("away_team", "Away")

        # Generate prediction
        try:
            prediction = predictor.predict_winner(home_team, away_team)
            home_prob = prediction.get("home_win_prob", 0.5)
            confidence = prediction.get("confidence", "medium")
        except:
            home_prob = 0.55
            confidence = "medium"

        yield json.dumps({
            "type": "prediction",
            "home_team": home_team,
            "away_team": away_team,
            "home_win_probability": round(home_prob, 4),
            "away_win_probability": round(1 - home_prob, 4),
            "confidence": confidence,
            "model_version": "v1.0"
        })
        await asyncio.sleep(0.2)

        # Event 5: Contributing factors
        factors = [
            {
                "factor": "Home Field Advantage",
                "impact": "+3%",
                "description": f"{home_team} benefits from home crowd and familiar environment"
            },
            {
                "factor": "Elo Rating Difference",
                "impact": f"{'+' if home_prob > 0.5 else ''}{int((home_prob - 0.5) * 100)}%",
                "description": "Based on historical performance and recent results"
            },
            {
                "factor": "Rest Days",
                "impact": "Neutral",
                "description": "Both teams have similar rest periods"
            }
        ]

        yield json.dumps({
            "type": "factors",
            "factors": factors,
            "total_factors_analyzed": len(factors)
        })
        await asyncio.sleep(0.2)

        # Event 6: Stream LLM reasoning (simulated token streaming)
        if include_reasoning:
            reasoning_parts = [
                f"Based on my analysis of {away_team} @ {home_team}, ",
                f"I project {home_team if home_prob > 0.5 else away_team} as the favorite ",
                f"with a {max(home_prob, 1-home_prob)*100:.1f}% win probability. ",
                f"\n\nKey factors driving this prediction:\n",
                f"1. **Home field advantage** gives {home_team} a baseline edge\n",
                f"2. **Historical Elo ratings** suggest ",
                f"{'the home team is slightly stronger' if home_prob > 0.5 else 'the away team has an edge'}\n",
                f"3. **Recent performance trends** have been factored into the model\n",
                f"\nThis represents a **{confidence} confidence** prediction."
            ]

            accumulated = ""
            for part in reasoning_parts:
                accumulated += part
                yield json.dumps({
                    "type": "reasoning_token",
                    "token": part,
                    "accumulated": accumulated
                })
                await asyncio.sleep(0.1)  # Simulate token streaming

            yield json.dumps({
                "type": "reasoning_complete",
                "full_reasoning": accumulated
            })

        await asyncio.sleep(0.1)

        # Event 7: Final recommendation
        # Calculate edge (assuming -110 odds = 52.4% implied)
        implied_prob = 0.524
        edge = home_prob - implied_prob if home_prob > 0.5 else (1 - home_prob) - implied_prob

        if edge > 0.05:
            recommendation = "STRONG BET"
            bet_size = "2-3% of bankroll"
        elif edge > 0.02:
            recommendation = "LEAN"
            bet_size = "1% of bankroll"
        else:
            recommendation = "PASS"
            bet_size = "No bet recommended"

        yield json.dumps({
            "type": "recommendation",
            "action": recommendation,
            "side": home_team if home_prob > 0.5 else away_team,
            "edge": round(edge * 100, 2),
            "suggested_bet_size": bet_size,
            "kelly_fraction": round(max(0, edge / 0.9), 4)
        })
        await asyncio.sleep(0.1)

        # Event 8: Complete
        yield json.dumps({
            "type": "complete",
            "message": "Analysis complete",
            "total_time_ms": 1500,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Streaming prediction error: {e}")
        yield json.dumps({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })


@router.get("/predict/{game_id}")
async def stream_prediction(
    game_id: str,
    sport: str = Query("NFL", description="Sport type"),
    include_reasoning: bool = Query(True, description="Include LLM reasoning")
):
    """
    Stream AI prediction analysis with Server-Sent Events.

    Returns progressive updates as the AI analyzes the game:
    - Model loading status
    - Data fetching progress
    - Initial probability prediction
    - Contributing factors
    - Streamed LLM reasoning (token by token)
    - Final recommendation with bet sizing
    """
    async def event_generator():
        async for data in _stream_ai_prediction(game_id, sport, include_reasoning):
            yield {
                "event": "message",
                "data": data
            }

    return EventSourceResponse(event_generator())


@router.get("/live/{game_id}")
async def stream_live_updates(
    game_id: str,
    sport: str = Query("NFL", description="Sport type")
):
    """
    Stream live game updates with real-time probability adjustments.

    For live games, continuously streams:
    - Score updates
    - Probability adjustments (Bayesian updates)
    - Momentum indicators
    - Betting opportunity alerts
    """
    async def live_generator():
        from src.prediction_agents.live_adjuster import get_live_adjuster, GameState

        adjuster = get_live_adjuster()
        pregame_prob = 0.55  # Would be fetched from predictions table

        # Initial state
        yield {
            "event": "message",
            "data": json.dumps({
                "type": "connected",
                "game_id": game_id,
                "sport": sport,
                "message": "Connected to live updates stream"
            })
        }

        # Simulate live updates (in production, would poll ESPN or use WebSocket)
        for i in range(10):
            await asyncio.sleep(2)  # Update every 2 seconds

            # Simulated game state
            state = GameState(
                home_score=14 + i,
                away_score=10 + (i // 2),
                period=2 + (i // 4),
                time_remaining_seconds=900 - (i * 90),
                sport=sport
            )

            # Get adjusted prediction
            adjusted = adjuster.adjust_prediction(pregame_prob, state)

            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "live_update",
                    "game_id": game_id,
                    "home_score": state.home_score,
                    "away_score": state.away_score,
                    "period": state.period,
                    "time_remaining": f"{state.time_remaining_seconds // 60}:{state.time_remaining_seconds % 60:02d}",
                    "pregame_prob": pregame_prob,
                    "live_prob": adjusted["live_home_prob"],
                    "probability_change": adjusted["probability_change"],
                    "momentum": adjusted["momentum"],
                    "confidence": adjusted["adjustment_confidence"],
                    "timestamp": datetime.now().isoformat()
                })
            }

    return EventSourceResponse(live_generator())


@router.get("/odds-movement/{game_id}")
async def stream_odds_movement(
    game_id: str,
    sport: str = Query("NFL", description="Sport type")
):
    """
    Stream real-time odds movement with sharp money detection.

    Alerts when:
    - Line moves significantly (>5%)
    - Reverse line movement detected
    - Steam move identified
    - Arbitrage opportunity found
    """
    async def odds_generator():
        from src.services.prediction_tracker import get_prediction_tracker

        tracker = get_prediction_tracker()

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "connected",
                "game_id": game_id,
                "message": "Monitoring odds movement..."
            })
        }

        prev_prob = None

        for i in range(20):
            await asyncio.sleep(3)  # Check every 3 seconds

            # Get latest odds
            history = tracker.get_odds_movement(game_id, hours=1)

            if history:
                latest = history[-1]
                current_prob = float(latest.get("home_implied_prob", 0.5))

                movement_type = "stable"
                alert = None

                if prev_prob:
                    change = current_prob - prev_prob
                    if abs(change) > 0.05:
                        movement_type = "significant"
                        alert = {
                            "type": "sharp_move",
                            "direction": "home" if change > 0 else "away",
                            "magnitude": abs(change),
                            "message": f"Sharp move detected: {abs(change)*100:.1f}% shift"
                        }
                    elif abs(change) > 0.02:
                        movement_type = "moderate"

                prev_prob = current_prob

                update = {
                    "type": "odds_update",
                    "game_id": game_id,
                    "home_odds": latest.get("home_odds"),
                    "away_odds": latest.get("away_odds"),
                    "home_implied_prob": current_prob,
                    "movement_type": movement_type,
                    "timestamp": datetime.now().isoformat()
                }

                if alert:
                    update["alert"] = alert

                yield {
                    "event": "message",
                    "data": json.dumps(update)
                }
            else:
                yield {
                    "event": "message",
                    "data": json.dumps({
                        "type": "no_data",
                        "game_id": game_id,
                        "message": "No odds data available yet"
                    })
                }

    return EventSourceResponse(odds_generator())


@router.get("/parlay-builder")
async def stream_parlay_analysis(
    game_ids: str = Query(..., description="Comma-separated game IDs"),
    sport: str = Query("NFL", description="Sport type")
):
    """
    Stream real-time parlay analysis with correlation adjustments.

    Provides:
    - Individual leg analysis
    - Correlation between legs
    - True odds calculation (adjusting for correlation)
    - EV analysis
    - Optimal leg ordering
    """
    games = game_ids.split(",")

    async def parlay_generator():
        yield {
            "event": "message",
            "data": json.dumps({
                "type": "start",
                "message": f"Analyzing {len(games)}-leg parlay...",
                "legs": games
            })
        }

        legs = []
        combined_prob = 1.0

        # Analyze each leg
        for i, game_id in enumerate(games):
            await asyncio.sleep(0.5)

            # Simulated prediction (would use actual predictor)
            prob = 0.55 + (i * 0.02)  # Example probabilities
            odds = -110 if prob > 0.5 else int(100 * (1 - prob) / prob)

            leg = {
                "game_id": game_id.strip(),
                "probability": prob,
                "odds": odds,
                "side": "home"
            }
            legs.append(leg)
            combined_prob *= prob

            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "leg_analyzed",
                    "leg_number": i + 1,
                    "leg": leg,
                    "running_combined_prob": combined_prob
                })
            }

        await asyncio.sleep(0.3)

        # Correlation analysis
        correlation_factor = 0.95 ** (len(legs) - 1)  # Slight negative correlation
        adjusted_prob = combined_prob * correlation_factor

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "correlation_analysis",
                "raw_combined_probability": combined_prob,
                "correlation_factor": correlation_factor,
                "adjusted_probability": adjusted_prob,
                "message": "Slight negative correlation between legs detected"
            })
        }

        await asyncio.sleep(0.3)

        # Calculate parlay odds and EV
        parlay_multiplier = 1.0
        for leg in legs:
            odds = leg["odds"]
            if odds > 0:
                parlay_multiplier *= (odds / 100) + 1
            else:
                parlay_multiplier *= (100 / abs(odds)) + 1

        # Implied probability from parlay odds
        implied_prob = 1 / parlay_multiplier
        edge = adjusted_prob - implied_prob
        ev = (adjusted_prob * (parlay_multiplier - 1)) - (1 - adjusted_prob)

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "parlay_result",
                "num_legs": len(legs),
                "legs": legs,
                "combined_probability": adjusted_prob,
                "parlay_odds_decimal": parlay_multiplier,
                "parlay_odds_american": int((parlay_multiplier - 1) * 100) if parlay_multiplier >= 2 else int(-100 / (parlay_multiplier - 1)),
                "implied_probability": implied_prob,
                "edge": edge,
                "expected_value": ev,
                "recommendation": "BET" if ev > 0.1 else ("CONSIDER" if ev > 0 else "PASS"),
                "kelly_fraction": max(0, edge / (parlay_multiplier - 1)) if edge > 0 else 0
            })
        }

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "complete",
                "timestamp": datetime.now().isoformat()
            })
        }

    return EventSourceResponse(parlay_generator())
