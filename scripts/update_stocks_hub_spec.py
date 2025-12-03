#!/usr/bin/env python3
"""
Update Stocks Hub Feature Spec in Database
This script updates the ava_feature_specs table with the enhanced Stocks Hub documentation.
"""

import asyncio
import asyncpg
import json
from datetime import datetime


async def update_stocks_hub_spec():
    """Update the stocks-hub feature specification in the database."""

    # Database connection
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password='postgres',
        database='magnus'
    )

    try:
        # Stocks Hub Feature Spec
        spec_data = {
            "feature_id": "stocks_tile_hub",
            "feature_name": "Stocks Tile Hub",
            "category": "frontend_pages",
            "subcategory": "trading",
            "purpose": "AI-powered stock analysis dashboard with modern infrastructure, circuit breakers, and streaming analysis",
            "description": """The Stocks Tile Hub is a comprehensive stock analysis page that provides:

1. **AI Scoring System (v2.2)**: Multi-factor scoring (0-100) with diverse outputs per stock
   - Prediction Score: Uses 1d/5d predictions with symbol-specific adjustments
   - Technical Score: RSI, MACD, SMA, momentum with weighted averaging
   - Smart Money Score: Volume analysis and order block detection
   - Volatility Score: GARCH-based IV/RV analysis
   - Sentiment Score: Momentum and win-rate analysis

2. **Modern Infrastructure (v2.2)**:
   - Shared AsyncDatabaseManager with connection pooling, health checks, retry logic
   - Circuit breaker protection for yfinance and LLM API calls
   - OpenTelemetry tracing integration
   - Health endpoint with infrastructure status

3. **Watchlists**: 10 pre-configured watchlists covering sectors, strategies, and ETFs
   - Tech Leaders, AI & Semiconductor, Financials, Healthcare
   - Options Favorites, High IV Plays, ETF Universe
   - Dividend Aristocrats, Consumer, Energy

4. **Real-time Features**:
   - SSE streaming analysis with progressive updates (circuit breaker protected)
   - Price animations with color flash on change
   - Live technical signals (RSI, trend, vol regime)

5. **Modern UI Components**:
   - Animated stock cards with hover effects and skeleton loading
   - AI score meter with gradient visualization
   - Filter by: bullish, bearish, high IV, breakout, favorites
   - Sort by: AI score, change, IV, name
   - Grid layout: 2/3/4 columns responsive

6. **AI Reasoning**: Dynamic reasoning generation based on score components,
   technical context, and price predictions with optional LLM deep analysis.""",
            "key_responsibilities": [
                "Calculate diverse AI scores using 5-factor ensemble",
                "Provide circuit-breaker protected streaming analysis",
                "Manage watchlists (system + custom)",
                "Display interactive tiles with skeleton loading",
                "Generate AI-powered reasoning with LLM deep analysis",
                "Expose infrastructure health and circuit breaker status"
            ],
            "version": "2.2.0",
            "status": "active",
            "maturity_level": "stable",
            "technical_details": json.dumps({
                "frontend": {
                    "page": "frontend/src/pages/StocksTileHub.tsx",
                    "components": [
                        "frontend/src/components/stocks/AnimatedStockCard.tsx"
                    ],
                    "hooks": [
                        "frontend/src/hooks/useStockStreamingAnalysis.ts"
                    ],
                    "store": "frontend/src/store/stockWatchlistStore.ts",
                    "framework": "React + TypeScript",
                    "state_management": "Zustand",
                    "animations": "Framer Motion",
                    "ui": "shadcn/ui"
                },
                "backend": {
                    "router": "backend/routers/stocks_tiles.py",
                    "service": "backend/services/stock_score_service.py",
                    "models": "backend/services/advanced_risk_models.py",
                    "data": "backend/infrastructure/async_yfinance.py",
                    "endpoints": [
                        "GET /api/stocks/tiles/all-data",
                        "GET /api/stocks/tiles/score/{symbol}",
                        "POST /api/stocks/tiles/batch",
                        "GET /api/stocks/tiles/stream/analyze/{symbol}",
                        "GET /api/stocks/tiles/watchlists",
                        "POST /api/stocks/tiles/watchlists",
                        "DELETE /api/stocks/tiles/watchlists/{name}",
                        "GET /api/stocks/tiles/prices"
                    ]
                },
                "scoring": {
                    "method": "ensemble_v2",
                    "components": {
                        "prediction": {"weight": "0.15-0.30", "model": "mean_reversion + momentum + RSI"},
                        "technical": {"weight": "0.15-0.30", "indicators": ["RSI", "MACD", "SMA", "momentum"]},
                        "smart_money": {"weight": "0.15-0.25", "signals": ["volume_ratio", "order_blocks"]},
                        "volatility": {"weight": "0.10-0.30", "model": "GARCH(1,1)"},
                        "sentiment": {"weight": "0.15", "factors": ["momentum", "win_rate"]}
                    },
                    "regimes": ["NORMAL", "LOW_VOL", "ELEVATED", "EXTREME"],
                    "recommendations": ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
                },
                "caching": {
                    "all_data": "120s TTL",
                    "individual_scores": "300s TTL",
                    "custom_watchlists": "30 days TTL"
                },
                "watchlists": [
                    "Tech Leaders", "Options Favorites", "AI & Semiconductor",
                    "High IV Plays", "Dividend Aristocrats", "Financials",
                    "Healthcare", "ETF Universe", "Consumer", "Energy"
                ]
            })
        }

        # Check if feature exists
        existing = await conn.fetchrow(
            "SELECT id FROM ava_feature_specs WHERE feature_id = $1",
            spec_data["feature_id"]
        )

        if existing:
            # Update existing spec
            await conn.execute("""
                UPDATE ava_feature_specs SET
                    feature_name = $2,
                    category = $3::spec_category,
                    subcategory = $4,
                    purpose = $5,
                    description = $6,
                    key_responsibilities = $7,
                    version = $8,
                    status = $9,
                    maturity_level = $10,
                    technical_details = $11::jsonb,
                    updated_at = NOW(),
                    analyzed_at = NOW()
                WHERE feature_id = $1
            """,
                spec_data["feature_id"],
                spec_data["feature_name"],
                spec_data["category"],
                spec_data["subcategory"],
                spec_data["purpose"],
                spec_data["description"],
                spec_data["key_responsibilities"],
                spec_data["version"],
                spec_data["status"],
                spec_data["maturity_level"],
                spec_data["technical_details"]
            )
            print(f"Updated existing spec for {spec_data['feature_id']}")
        else:
            # Insert new spec
            await conn.execute("""
                INSERT INTO ava_feature_specs (
                    feature_id, feature_name, category, subcategory, purpose,
                    description, key_responsibilities, version, status,
                    maturity_level, technical_details, analyzed_at
                ) VALUES ($1, $2, $3::spec_category, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, NOW())
            """,
                spec_data["feature_id"],
                spec_data["feature_name"],
                spec_data["category"],
                spec_data["subcategory"],
                spec_data["purpose"],
                spec_data["description"],
                spec_data["key_responsibilities"],
                spec_data["version"],
                spec_data["status"],
                spec_data["maturity_level"],
                spec_data["technical_details"]
            )
            print(f"Inserted new spec for {spec_data['feature_id']}")

        # Also update the stock_score_service spec
        service_spec = {
            "feature_id": "stock_score_service",
            "feature_name": "Stock Score Service",
            "category": "backend_services",
            "subcategory": "trading",
            "purpose": "AI-powered scoring with shared database and circuit breakers",
            "description": """Enhanced stock scoring service (v2.2) with modern infrastructure.

Infrastructure (v2.2):
1. **Shared AsyncDatabaseManager**: Centralized connection pooling
2. **Circuit breaker protection**: Automatic failure detection
3. **OpenTelemetry tracing**: Query observability
4. **Health checks**: Automatic retry with exponential backoff

Scoring Features:
1. **Symbol-specific adjustments**: Hash-based variance for diversity
2. **Continuous scoring curves**: Linear mappings
3. **Proper EMA calculation**: Real exponential moving averages
4. **Score history persistence**: Tracking for backtesting

Score Components:
- Prediction Score: 1d/5d predictions with volatility weighting
- Technical Score: RSI, MACD, SMA with momentum overlay
- Smart Money Score: Volume analysis and price action
- Volatility Score: GARCH-based forecasting
- Sentiment Score: Momentum and win-rate analysis""",
            "key_responsibilities": [
                "Calculate diverse AI scores (0-100) per stock",
                "Use shared AsyncDatabaseManager for all DB operations",
                "Determine market regime for adaptive weighting",
                "Generate score components with raw values and weights",
                "Support batch scoring with metadata pre-fetching",
                "Persist scores to history for backtesting"
            ],
            "version": "2.2.0",
            "status": "active",
            "maturity_level": "stable"
        }

        existing_service = await conn.fetchrow(
            "SELECT id FROM ava_feature_specs WHERE feature_id = $1",
            service_spec["feature_id"]
        )

        if existing_service:
            await conn.execute("""
                UPDATE ava_feature_specs SET
                    feature_name = $2,
                    category = $3::spec_category,
                    subcategory = $4,
                    purpose = $5,
                    description = $6,
                    key_responsibilities = $7,
                    version = $8,
                    status = $9,
                    maturity_level = $10,
                    updated_at = NOW(),
                    analyzed_at = NOW()
                WHERE feature_id = $1
            """,
                service_spec["feature_id"],
                service_spec["feature_name"],
                service_spec["category"],
                service_spec["subcategory"],
                service_spec["purpose"],
                service_spec["description"],
                service_spec["key_responsibilities"],
                service_spec["version"],
                service_spec["status"],
                service_spec["maturity_level"]
            )
            print(f"Updated existing spec for {service_spec['feature_id']}")
        else:
            await conn.execute("""
                INSERT INTO ava_feature_specs (
                    feature_id, feature_name, category, subcategory, purpose,
                    description, key_responsibilities, version, status,
                    maturity_level, analyzed_at
                ) VALUES ($1, $2, $3::spec_category, $4, $5, $6, $7, $8, $9, $10, NOW())
            """,
                service_spec["feature_id"],
                service_spec["feature_name"],
                service_spec["category"],
                service_spec["subcategory"],
                service_spec["purpose"],
                service_spec["description"],
                service_spec["key_responsibilities"],
                service_spec["version"],
                service_spec["status"],
                service_spec["maturity_level"]
            )
            print(f"Inserted new spec for {service_spec['feature_id']}")

        print("\nSpec agents updated successfully!")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(update_stocks_hub_spec())
