# AVA Comprehensive Options Trading Platform - Implementation Plan

**Date:** 2025-11-28
**Scope:** Full options trading platform with AI, backtesting, and all major strategies
**Broker:** Robinhood (robin_stocks)

---

## Executive Summary

AVA already has a **sophisticated foundation** with:
- Greeks calculator (Black-Scholes)
- Robinhood integration (login, positions, rate limiting)
- LLM integration (Qwen, Llama via Ollama)
- Premium scanner
- Database schema (options_chains, positions, wheel_cycles)
- Frontend pages (OptionsGreeks, PremiumScanner, Positions)

**What needs to be built:**
1. Full backtesting engine with vectorized performance
2. All 15+ options strategies with automation
3. Multi-agent AI trading system (TradingAgents-style)
4. Real-time streaming Greeks
5. Order execution via Robinhood
6. Advanced risk management

---

## Phase 1: Core Infrastructure Upgrades (Week 1-2)

### 1.1 Install Modern Libraries

```bash
# Backtesting & Performance
pip install vectorbt optopsy py_vollib py_vollib_vectorized polars

# AI/ML
pip install langchain-anthropic langgraph fingpt transformers

# Real-time Data
pip install websocket-client aiohttp

# Already have: robin_stocks, yfinance, numpy, scipy, pandas
```

### 1.2 Upgrade Greeks Calculator

**File:** `src/ava/systems/greeks_calculator.py`

Add:
- Vectorized calculations using `py_vollib_vectorized`
- Higher-order Greeks (Vomma, Vanna, Charm, Speed)
- IV Surface calculation
- Greeks for multi-leg positions
- Theta decay curve projection

### 1.3 Create Robinhood Order Execution Service

**New File:** `src/services/robinhood_order_service.py`

```python
class RobinhoodOrderService:
    """Execute options orders via Robinhood"""

    def buy_option(symbol, strike, expiration, option_type, quantity)
    def sell_option(symbol, strike, expiration, option_type, quantity)
    def close_position(position_id)
    def roll_option(old_position, new_strike, new_expiration)

    # Multi-leg orders
    def execute_spread(legs: List[OptionLeg])
    def execute_iron_condor(symbol, call_spread, put_spread)
    def execute_straddle(symbol, strike, expiration)
```

---

## Phase 2: Vectorized Backtesting Engine (Week 2-3)

### 2.1 Core Backtesting Framework

**New Directory:** `src/ava/backtesting/`

```
src/ava/backtesting/
├── __init__.py
├── engine.py              # Main backtesting engine
├── data_loader.py         # Historical options data loading
├── strategy_base.py       # Base class for strategies
├── metrics.py             # Performance metrics (Sharpe, Sortino, etc.)
├── visualizations.py      # P&L charts, Greeks over time
└── reports.py             # PDF/HTML report generation
```

### 2.2 Backtesting Engine Architecture

**File:** `src/ava/backtesting/engine.py`

```python
class OptionsBacktestEngine:
    """
    Vectorized options backtesting engine
    Uses optopsy + vectorbt for maximum performance
    """

    def __init__(self):
        self.data = None
        self.strategy = None
        self.results = None

    def load_data(self, symbols: List[str], start: date, end: date):
        """Load historical options chain data"""
        # Support: yfinance, CBOE, local CSV

    def set_strategy(self, strategy: OptionsStrategy):
        """Set the strategy to backtest"""

    def run(self,
            initial_capital: float = 100000,
            commission: float = 0.65,
            slippage: float = 0.01) -> BacktestResults:
        """Execute backtest with realistic costs"""

    def optimize(self, param_grid: Dict) -> OptimizationResults:
        """Optimize strategy parameters"""
        # Grid search, random search, or Bayesian optimization

    def monte_carlo(self, n_simulations: int = 1000) -> MonteCarloResults:
        """Run Monte Carlo simulations"""
```

### 2.3 Historical Data Pipeline

**File:** `src/ava/backtesting/data_loader.py`

```python
class OptionsDataLoader:
    """Load and cache historical options data"""

    def load_from_yfinance(symbol, start, end)
    def load_from_cboe(symbol, start, end)  # If purchased
    def load_from_csv(filepath)
    def load_from_database()

    def calculate_historical_greeks(chain_data)
    def interpolate_missing_strikes()
    def handle_splits_and_adjustments()
```

---

## Phase 3: Options Strategies Library (Week 3-5)

### 3.1 Strategy Framework

**New Directory:** `src/ava/strategies/`

```
src/ava/strategies/
├── __init__.py
├── base.py                    # Abstract base strategy
├── entry_rules.py             # Entry condition definitions
├── exit_rules.py              # Exit condition definitions
├── position_sizing.py         # Kelly criterion, fixed fractional
│
├── income/                    # Income-generating strategies
│   ├── covered_call.py
│   ├── cash_secured_put.py
│   ├── wheel.py
│   ├── poor_mans_covered_call.py
│   └── jade_lizard.py
│
├── spreads/                   # Spread strategies
│   ├── vertical_spread.py     # Bull/Bear Call/Put spreads
│   ├── calendar_spread.py
│   ├── diagonal_spread.py
│   ├── ratio_spread.py
│   └── backspread.py
│
├── neutral/                   # Neutral/Range strategies
│   ├── iron_condor.py
│   ├── iron_butterfly.py
│   ├── straddle.py
│   ├── strangle.py
│   └── butterfly.py
│
├── directional/               # Directional strategies
│   ├── long_call.py
│   ├── long_put.py
│   ├── synthetic_long.py
│   └── collar.py
│
├── volatility/                # Volatility strategies
│   ├── long_straddle.py
│   ├── long_strangle.py
│   ├── vix_strategies.py
│   └── volatility_skew.py
│
└── advanced/                  # Advanced strategies
    ├── zero_dte.py            # 0DTE specific logic
    ├── earnings_plays.py
    ├── dividend_capture.py
    └── gamma_scalping.py
```

### 3.2 Base Strategy Class

**File:** `src/ava/strategies/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class StrategyType(Enum):
    INCOME = "income"
    SPREAD = "spread"
    NEUTRAL = "neutral"
    DIRECTIONAL = "directional"
    VOLATILITY = "volatility"

@dataclass
class OptionLeg:
    option_type: str  # 'call' or 'put'
    action: str       # 'buy' or 'sell'
    strike: float
    expiration: date
    quantity: int

@dataclass
class StrategySetup:
    legs: List[OptionLeg]
    max_profit: float
    max_loss: float
    breakeven: List[float]
    probability_of_profit: float

class OptionsStrategy(ABC):
    """Base class for all options strategies"""

    name: str
    strategy_type: StrategyType
    min_iv_rank: float = 0
    max_iv_rank: float = 100
    min_dte: int = 7
    max_dte: int = 45

    @abstractmethod
    def find_opportunities(self, chain: OptionsChain) -> List[StrategySetup]:
        """Find valid setups in the options chain"""
        pass

    @abstractmethod
    def entry_conditions(self, setup: StrategySetup, market_data: Dict) -> bool:
        """Check if entry conditions are met"""
        pass

    @abstractmethod
    def exit_conditions(self, position: Position) -> Tuple[bool, str]:
        """Check exit conditions, return (should_exit, reason)"""
        pass

    @abstractmethod
    def calculate_position_size(self, setup: StrategySetup, account: Account) -> int:
        """Calculate optimal position size"""
        pass

    def calculate_greeks(self, setup: StrategySetup) -> Dict:
        """Calculate aggregate Greeks for the strategy"""
        pass

    def visualize_payoff(self, setup: StrategySetup) -> Figure:
        """Generate payoff diagram"""
        pass
```

### 3.3 Wheel Strategy Implementation

**File:** `src/ava/strategies/income/wheel.py`

```python
class WheelStrategy(OptionsStrategy):
    """
    The Wheel Strategy - Automated premium harvesting

    Cycle:
    1. Sell cash-secured puts on stocks you want to own
    2. If assigned, own the stock at a discount
    3. Sell covered calls on the stock
    4. If called away, restart with puts
    5. Repeat, collecting premium throughout

    AI-Enhanced Features:
    - Dynamic strike selection based on IV rank
    - Optimal DTE selection based on theta decay curve
    - Smart roll decisions using LLM analysis
    - Earnings avoidance
    """

    name = "Wheel Strategy"
    strategy_type = StrategyType.INCOME

    def __init__(self,
                 target_delta: float = 0.30,
                 min_premium_yield: float = 1.0,  # 1% per trade
                 target_dte_range: Tuple[int, int] = (30, 45),
                 roll_when_itm: bool = True,
                 avoid_earnings: bool = True):
        self.target_delta = target_delta
        self.min_premium_yield = min_premium_yield
        self.target_dte_range = target_dte_range
        self.roll_when_itm = roll_when_itm
        self.avoid_earnings = avoid_earnings

    def find_put_opportunities(self, symbol: str, chain: OptionsChain) -> List[WheelSetup]:
        """Find CSP opportunities for the wheel"""

    def find_call_opportunities(self, symbol: str, shares: int, cost_basis: float,
                                chain: OptionsChain) -> List[WheelSetup]:
        """Find covered call opportunities"""

    def should_roll(self, position: Position, current_price: float) -> RollDecision:
        """Determine if position should be rolled"""
        # Uses AI to analyze market conditions

    def get_roll_target(self, position: Position, chain: OptionsChain) -> OptionLeg:
        """Find optimal roll target"""
```

### 3.4 Iron Condor Implementation

**File:** `src/ava/strategies/neutral/iron_condor.py`

```python
class IronCondorStrategy(OptionsStrategy):
    """
    Iron Condor - Profit from low volatility / range-bound markets

    Structure:
    - Sell OTM put (short put)
    - Buy further OTM put (long put) - protection
    - Sell OTM call (short call)
    - Buy further OTM call (long call) - protection

    AI-Enhanced Features:
    - Dynamic wing width based on volatility
    - Optimal short strike selection (delta-based)
    - Smart adjustment recommendations
    - Early exit optimization
    """

    def __init__(self,
                 short_delta: float = 0.16,  # ~1 std dev
                 wing_width: int = 5,        # $5 wide wings
                 min_credit: float = 0.30,   # Min $0.30 credit
                 profit_target: float = 0.50, # Exit at 50% profit
                 max_loss_percent: float = 2.0):  # 2x credit max loss
        pass

    def find_opportunities(self, symbol: str, chain: OptionsChain) -> List[IronCondorSetup]:
        """Find IC opportunities"""

    def adjust_position(self, position: IronCondorPosition,
                       market_data: Dict) -> AdjustmentRecommendation:
        """Recommend adjustments when tested"""
        # Roll untested side, close tested side, etc.
```

### 3.5 Zero DTE Strategy

**File:** `src/ava/strategies/advanced/zero_dte.py`

```python
class ZeroDTEStrategy(OptionsStrategy):
    """
    0DTE Options Trading

    High-risk, high-reward intraday options trading

    Sub-strategies:
    - Scalping (quick in/out on momentum)
    - Iron Condor (collect theta on expiration day)
    - Butterfly (low cost, high reward)
    - Directional (leverage gamma for big moves)

    AI-Enhanced Features:
    - Real-time GEX (Gamma Exposure) analysis
    - Intraday momentum detection
    - Optimal entry/exit timing
    - Dynamic position sizing based on volatility
    """

    def __init__(self,
                 strategy_type: str = "iron_condor",  # or "scalp", "butterfly"
                 max_position_size: float = 0.02,    # 2% of portfolio per trade
                 profit_target: float = 0.30,        # 30% profit target
                 stop_loss: float = 0.50,            # 50% stop loss
                 trading_hours: Tuple[time, time] = (time(9, 30), time(15, 30))):
        pass

    def analyze_gex(self, options_data: Dict) -> GEXAnalysis:
        """Analyze Gamma Exposure for key levels"""

    def get_intraday_signal(self, market_data: Dict) -> TradingSignal:
        """Generate intraday trading signal"""
```

---

## Phase 4: Multi-Agent AI Trading System (Week 5-7)

### 4.1 TradingAgents-Style Architecture

**New Directory:** `src/ava/agents/trading/`

```
src/ava/agents/trading/
├── __init__.py
├── orchestrator.py            # Main coordinator
├── agents/
│   ├── fundamental_analyst.py
│   ├── technical_analyst.py
│   ├── sentiment_analyst.py
│   ├── options_specialist.py
│   ├── risk_manager.py
│   ├── bull_researcher.py
│   ├── bear_researcher.py
│   └── trader.py
├── tools/
│   ├── market_data.py
│   ├── options_chain.py
│   ├── news_fetcher.py
│   ├── social_sentiment.py
│   └── portfolio_tools.py
└── debate/
    ├── debate_protocol.py
    └── consensus.py
```

### 4.2 Trading Orchestrator

**File:** `src/ava/agents/trading/orchestrator.py`

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

class TradingState(TypedDict):
    symbol: str
    fundamental_analysis: Dict
    technical_analysis: Dict
    sentiment_analysis: Dict
    options_analysis: Dict
    bull_case: str
    bear_case: str
    risk_assessment: Dict
    final_recommendation: Dict
    trade_execution: Dict

class TradingOrchestrator:
    """
    Multi-Agent Trading System

    Workflow:
    1. Parallel Analysis Phase
       - Fundamental Analyst examines financials, earnings, growth
       - Technical Analyst examines charts, indicators, patterns
       - Sentiment Analyst examines news, social media, options flow
       - Options Specialist examines IV, Greeks, unusual activity

    2. Debate Phase
       - Bull Researcher presents bullish case
       - Bear Researcher presents bearish case
       - Structured debate with rebuttals

    3. Risk Assessment
       - Risk Manager evaluates all positions
       - Portfolio-level Greeks analysis
       - Position sizing recommendations

    4. Trade Decision
       - Trader synthesizes all inputs
       - Generates specific trade recommendation
       - Includes entry, exit, position size

    5. Execution (if approved)
       - Execute via Robinhood
       - Log to database
       - Set up monitoring
    """

    def __init__(self, llm_provider: str = "local"):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

    async def analyze(self, symbol: str) -> TradingRecommendation:
        """Run full analysis pipeline"""

    async def analyze_batch(self, symbols: List[str]) -> List[TradingRecommendation]:
        """Analyze multiple symbols in parallel"""
```

### 4.3 Options Specialist Agent

**File:** `src/ava/agents/trading/agents/options_specialist.py`

```python
class OptionsSpecialistAgent:
    """
    AI Agent specialized in options analysis

    Capabilities:
    - Analyze options chains for opportunities
    - Calculate optimal strikes and expirations
    - Identify unusual options activity
    - Recommend specific strategies based on outlook
    - Manage existing positions
    """

    SYSTEM_PROMPT = """
    You are an expert options trader with deep knowledge of:
    - All options strategies (spreads, condors, butterflies, etc.)
    - Greeks and their implications
    - Volatility analysis (IV rank, IV percentile, skew)
    - Options flow analysis
    - Position management and adjustments

    Always consider:
    - Risk/reward ratio
    - Probability of profit
    - Maximum loss scenarios
    - Liquidity (bid-ask spread, open interest)
    - Earnings and dividend dates
    """

    def analyze_chain(self, symbol: str, outlook: str) -> OptionsAnalysis:
        """Analyze options chain and recommend strategy"""

    def find_unusual_activity(self, symbol: str) -> List[UnusualActivity]:
        """Detect unusual options activity"""

    def recommend_adjustment(self, position: Position) -> AdjustmentPlan:
        """Recommend position adjustments"""
```

### 4.4 Debate System

**File:** `src/ava/agents/trading/debate/debate_protocol.py`

```python
class BullBearDebate:
    """
    Structured debate between Bull and Bear researchers

    Protocol:
    1. Bull presents initial thesis (3 key points)
    2. Bear presents initial thesis (3 key points)
    3. Bull rebuts bear's points
    4. Bear rebuts bull's points
    5. Final summary from each side
    6. Moderator synthesizes into probability-weighted outlook
    """

    async def run_debate(self,
                        symbol: str,
                        market_data: Dict,
                        time_horizon: str = "30d") -> DebateResult:
        """Run full debate and return consensus"""
```

---

## Phase 5: Real-Time Streaming & Monitoring (Week 7-8)

### 5.1 WebSocket Streaming Service

**New File:** `src/ava/streaming/websocket_service.py`

```python
class OptionsStreamingService:
    """
    Real-time options data streaming

    Data Sources:
    - Robinhood (via polling - no native websocket)
    - yfinance (delayed)
    - Polygon.io (if subscribed - real-time)
    """

    async def stream_positions(self, callback: Callable):
        """Stream position updates"""

    async def stream_greeks(self, positions: List[Position], callback: Callable):
        """Stream real-time Greeks for positions"""

    async def stream_chain(self, symbol: str, callback: Callable):
        """Stream options chain updates"""
```

### 5.2 Real-Time Dashboard Updates

**Update:** `frontend/src/pages/Positions.tsx`

Add:
- WebSocket connection for live Greeks
- Theta decay visualization (countdown timer)
- P&L sparklines
- Alert triggers (delta breach, profit target, stop loss)

### 5.3 Alert System

**New File:** `src/ava/alerts/options_alerts.py`

```python
class OptionsAlertSystem:
    """
    Multi-channel alert system for options positions

    Alert Types:
    - Price alerts (underlying moves X%)
    - Greeks alerts (delta > 0.5, gamma spike, etc.)
    - Profit target reached
    - Stop loss triggered
    - DTE warnings (7 days, 3 days, 1 day)
    - Earnings approaching
    - Assignment risk (ITM at expiration)

    Channels:
    - Discord (existing integration)
    - Telegram (existing integration)
    - Email (existing integration)
    - In-app notifications
    """
```

---

## Phase 6: Risk Management System (Week 8-9)

### 6.1 Portfolio Risk Engine

**New File:** `src/ava/risk/portfolio_risk.py`

```python
class PortfolioRiskEngine:
    """
    Comprehensive portfolio risk management

    Metrics:
    - Portfolio Greeks (net delta, gamma, theta, vega)
    - Value at Risk (VaR) - 95% and 99%
    - Expected Shortfall (CVaR)
    - Maximum Drawdown
    - Beta-weighted delta
    - Correlation analysis

    Limits:
    - Max portfolio delta
    - Max single position size
    - Max sector exposure
    - Max correlation concentration
    """

    def calculate_portfolio_greeks(self, positions: List[Position]) -> PortfolioGreeks:
        """Calculate aggregate portfolio Greeks"""

    def calculate_var(self,
                     positions: List[Position],
                     confidence: float = 0.95,
                     horizon: int = 1) -> float:
        """Calculate Value at Risk"""

    def stress_test(self,
                   positions: List[Position],
                   scenarios: List[Scenario]) -> StressTestResults:
        """Run stress test scenarios"""

    def check_limits(self,
                    positions: List[Position],
                    new_trade: Trade) -> LimitCheckResult:
        """Check if new trade violates risk limits"""
```

### 6.2 Position Sizing Calculator

**Update:** `src/ava/strategies/position_sizing.py`

```python
class AdvancedPositionSizer:
    """
    Advanced position sizing algorithms

    Methods:
    - Fixed fractional (X% of portfolio)
    - Kelly Criterion (optimal mathematical sizing)
    - Risk parity (equal risk contribution)
    - Maximum drawdown constrained
    - Vol-adjusted sizing
    """

    def kelly_criterion(self,
                       win_rate: float,
                       avg_win: float,
                       avg_loss: float) -> float:
        """Calculate Kelly optimal position size"""

    def max_contracts_by_risk(self,
                              account_value: float,
                              max_risk_percent: float,
                              max_loss_per_contract: float) -> int:
        """Calculate max contracts based on risk tolerance"""
```

---

## Phase 7: Enhanced Frontend (Week 9-11)

### 7.1 New Pages

| Page | Purpose |
|------|---------|
| `/strategies` | Strategy library browser with backtest results |
| `/backtesting` | Interactive backtesting interface |
| `/ai-trader` | Multi-agent trading dashboard |
| `/risk-dashboard` | Portfolio risk visualization |
| `/automation` | Automated trading rules setup |
| `/trade-journal` | Trade history with P&L analysis |

### 7.2 Strategy Builder UI

**New File:** `frontend/src/pages/StrategyBuilder.tsx`

Features:
- Visual strategy builder (drag & drop legs)
- Real-time payoff diagram
- Greeks display
- Probability of profit
- Risk/reward analysis
- One-click execution

### 7.3 Backtesting Dashboard

**New File:** `frontend/src/pages/BacktestDashboard.tsx`

Features:
- Strategy selector
- Date range picker
- Parameter optimization grid
- Results visualization:
  - Equity curve
  - Drawdown chart
  - Monthly returns heatmap
  - Trade distribution
  - Greeks over time
- Comparison mode (multiple strategies)

---

## Phase 8: Database Schema Updates (Throughout)

### 8.1 New Tables

```sql
-- Backtesting results
CREATE TABLE backtest_results (
    id UUID PRIMARY KEY,
    strategy_name VARCHAR(100),
    parameters JSONB,
    start_date DATE,
    end_date DATE,
    initial_capital DECIMAL(15,2),
    final_capital DECIMAL(15,2),
    total_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    sortino_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    profit_factor DECIMAL(10,4),
    total_trades INTEGER,
    trades JSONB,
    equity_curve JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Strategy configurations
CREATE TABLE strategy_configs (
    id UUID PRIMARY KEY,
    name VARCHAR(100),
    strategy_type VARCHAR(50),
    parameters JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Automated trading rules
CREATE TABLE trading_rules (
    id UUID PRIMARY KEY,
    name VARCHAR(100),
    strategy_id UUID REFERENCES strategy_configs(id),
    entry_conditions JSONB,
    exit_conditions JSONB,
    position_sizing JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    last_triggered TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- AI analysis cache
CREATE TABLE ai_analysis_cache (
    id UUID PRIMARY KEY,
    symbol VARCHAR(10),
    analysis_type VARCHAR(50),
    analysis JSONB,
    confidence DECIMAL(5,2),
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Trade journal
CREATE TABLE trade_journal (
    id UUID PRIMARY KEY,
    trade_id UUID,
    symbol VARCHAR(10),
    strategy_used VARCHAR(100),
    entry_reasoning TEXT,
    exit_reasoning TEXT,
    lessons_learned TEXT,
    screenshots JSONB,
    tags VARCHAR(50)[],
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Phase 9: Testing & Validation (Week 11-12)

### 9.1 Unit Tests

```
tests/
├── strategies/
│   ├── test_wheel.py
│   ├── test_iron_condor.py
│   ├── test_zero_dte.py
│   └── ...
├── backtesting/
│   ├── test_engine.py
│   ├── test_data_loader.py
│   └── test_metrics.py
├── agents/
│   ├── test_orchestrator.py
│   ├── test_debate.py
│   └── test_agents.py
└── integration/
    ├── test_robinhood_orders.py
    ├── test_end_to_end.py
    └── test_streaming.py
```

### 9.2 Paper Trading Validation

1. Run all strategies in paper trading mode for 2 weeks
2. Compare backtest results to paper trading results
3. Validate order execution logic
4. Test alert system
5. Stress test the system with high volume

---

## Implementation Order (Recommended)

### Sprint 1 (Week 1-2): Foundation
1. ✅ Install new libraries
2. ✅ Upgrade Greeks calculator
3. ✅ Create order execution service
4. ✅ Set up backtesting directory structure

### Sprint 2 (Week 3-4): Strategies Part 1
1. ✅ Base strategy class
2. ✅ Wheel strategy
3. ✅ Covered call / CSP
4. ✅ Vertical spreads
5. ✅ Basic backtesting engine

### Sprint 3 (Week 5-6): Strategies Part 2
1. ✅ Iron condor
2. ✅ Iron butterfly
3. ✅ Straddle/strangle
4. ✅ Calendar spreads
5. ✅ 0DTE strategies

### Sprint 4 (Week 7-8): AI System
1. ✅ Agent architecture
2. ✅ Options specialist agent
3. ✅ Bull/bear debate system
4. ✅ Trading orchestrator
5. ✅ Integration with LLM

### Sprint 5 (Week 9-10): Real-Time & Risk
1. ✅ Streaming service
2. ✅ Alert system
3. ✅ Portfolio risk engine
4. ✅ Position sizing

### Sprint 6 (Week 11-12): Frontend & Polish
1. ✅ New frontend pages
2. ✅ Strategy builder UI
3. ✅ Backtest dashboard
4. ✅ Testing & validation
5. ✅ Documentation

---

## Key Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Backtesting | vectorbt | 0.26+ | Vectorized performance |
| Options Backtest | optopsy | 2.0+ | Options-specific |
| Greeks | py_vollib_vectorized | 1.0+ | Fast Greeks |
| AI Orchestration | langgraph | 0.1+ | Multi-agent workflow |
| LLM | langchain-anthropic | 0.1+ | Claude integration |
| Broker | robin_stocks | 3.4+ | Robinhood API |
| Data | yfinance | 0.2+ | Market data |
| Database | sqlalchemy | 2.0+ | ORM |
| Frontend | react | 18+ | UI |
| Charts | recharts | 2.0+ | Visualizations |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Backtest speed | 1000 trades/second |
| Strategy coverage | 15+ strategies |
| AI analysis time | <30 seconds |
| Order execution | <2 seconds |
| Greeks calculation | Real-time (<100ms) |
| Uptime | 99.9% |
| Paper trading validation | 2 weeks before live |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Robinhood API rate limits | Aggressive caching, request batching |
| Historical data gaps | Multiple data sources, interpolation |
| LLM hallucinations | Structured outputs, validation, human approval |
| Order execution errors | Paper trading first, position limits |
| Market volatility | Risk limits, position sizing, alerts |

---

## Questions for User

1. **Paper trading first?** Recommend 2 weeks minimum before live
2. **Position size limits?** Suggest max 5% per trade, 20% per strategy
3. **Which strategies to prioritize?** Recommend: Wheel → Iron Condor → 0DTE
4. **AI automation level?** Suggest: AI recommends, human approves (initially)
5. **Alert preferences?** Discord, Telegram, or both?

---

*This plan integrates all 27+ GitHub repositories' concepts into a unified, modern, AI-powered options trading platform.*
