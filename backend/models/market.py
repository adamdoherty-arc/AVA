from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class MarketPrediction(BaseModel):
    predicted_outcome: Optional[str] = None
    confidence_score: Optional[float] = None
    edge_percentage: Optional[float] = None
    overall_rank: Optional[int] = None
    recommended_action: Optional[str] = None
    recommended_stake_pct: Optional[float] = None
    reasoning: Optional[str] = None

class Market(BaseModel):
    id: int
    ticker: str
    title: str
    market_type: str
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_date: Optional[datetime] = None
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
    volume: Optional[float] = None
    close_time: Optional[datetime] = None
    
    # Prediction fields flattened for easier consumption
    predicted_outcome: Optional[str] = None
    confidence_score: Optional[float] = None
    edge_percentage: Optional[float] = None
    overall_rank: Optional[int] = None
    recommended_action: Optional[str] = None
    recommended_stake_pct: Optional[float] = None
    reasoning: Optional[str] = None

class MarketResponse(BaseModel):
    markets: List[Market]
    count: int
