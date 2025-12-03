# GitHub Options Trading Strategies - Comprehensive Repository Guide

**Date:** 2025-11-28
**Purpose:** Curated list of GitHub repositories for options trading, backtesting, pricing, and strategy analysis
**Research Method:** GitHub API topics, star rankings, and community recommendations

---

## Table of Contents

1. [Tier 1: High-Star General Trading Platforms (10k+ stars)](#tier-1-high-star-general-trading-platforms)
2. [Tier 2: Options-Focused Libraries (1k-10k stars)](#tier-2-options-focused-libraries)
3. [Tier 3: Specialized Options Tools (100-1k stars)](#tier-3-specialized-options-tools)
4. [Tier 4: Niche & Emerging Projects (<100 stars)](#tier-4-niche--emerging-projects)
5. [Curated Awesome Lists](#curated-awesome-lists)
6. [Broker API Libraries](#broker-api-libraries)
7. [Options Pricing & Greeks Libraries](#options-pricing--greeks-libraries)
8. [Backtesting Frameworks](#backtesting-frameworks)
9. [Options Screeners & Scanners](#options-screeners--scanners)
10. [Strategy-Specific Tools](#strategy-specific-tools)
11. [Data Sources & APIs](#data-sources--apis)

---

## Tier 1: High-Star General Trading Platforms

### 1. QuantConnect/Lean - ⭐ 10k+
**URL:** https://github.com/QuantConnect/Lean
**Language:** C#, Python
**Description:** Open-source algorithmic trading engine supporting Equities, Forex, Options, Futures, Crypto, and CFDs.
**Options Support:** Full options chain analysis, American/European styles, Greeks calculation
**Key Features:**
- Multi-asset portfolio management
- Cloud deployment via QuantConnect
- Extensive documentation and community
- Historical options data integration

### 2. nautechsystems/nautilus_trader - ⭐ 16.5k
**URL:** https://github.com/nautechsystems/nautilus_trader
**Language:** Rust, Python
**Description:** High-performance algorithmic trading platform and event-driven backtester
**Options Support:** Derivatives trading support
**Key Features:**
- Ultra-low latency execution
- Real-time and historical backtesting
- Multiple broker integrations

### 3. vnpy/vnpy - ⭐ 34.1k
**URL:** https://github.com/vnpy/vnpy
**Language:** Python
**Description:** Python-based open-source quantitative trading platform
**Options Support:** Options trading module included
**Key Features:**
- Chinese market focus but extensible
- Comprehensive trading gateway support
- Active community

---

## Tier 2: Options-Focused Libraries

### 4. je-suis-tm/quant-trading - ⭐ 8.7k
**URL:** https://github.com/je-suis-tm/quant-trading
**Language:** Python
**Description:** Python quantitative trading strategies including VIX Calculator, Options Straddle, Monte Carlo, Pattern Recognition
**Options Strategies:**
- VIX Calculator implementation
- Options Straddle backtest
- Volatility trading strategies
**Key Features:**
- Well-documented Jupyter notebooks
- Educational focus with explanations
- Multiple strategy implementations

### 5. kernc/backtesting.py - ⭐ 5.6k
**URL:** https://github.com/kernc/backtesting.py
**Language:** Python
**Description:** Lightweight, fast backtesting framework for trading strategies
**Options Support:** Can be extended for options via custom logic
**Key Features:**
- Built on Pandas, NumPy, Bokeh
- Interactive visualizations
- Simple API
- AGPL-3.0 license

### 6. jmfernandes/robin_stocks - ⭐ 2k
**URL:** https://github.com/jmfernandes/robin_stocks
**Language:** Python
**Description:** Library for Robinhood, Gemini, and TD Ameritrade APIs
**Options Support:** Full options trading, chains, Greeks
**Key Features:**
- Options order placement
- Options chain retrieval
- Portfolio analysis
- Real-time quotes

### 7. jasonstrimpel/volatility-trading - ⭐ 1.8k
**URL:** https://github.com/jasonstrimpel/volatility-trading
**Language:** Python
**Description:** Complete set of volatility estimators based on Euan Sinclair's "Volatility Trading"
**Options Relevance:** Essential for options pricing and IV analysis
**Estimators Included:**
- Parkinson
- Garman-Klass
- Rogers-Satchell
- Yang-Zhang
- Historical volatility

### 8. erdewit/ib_insync → ib-api-reloaded/ib_async - ⭐ 1.2k+
**URL:** https://github.com/ib-api-reloaded/ib_async
**Language:** Python
**Description:** Sync/async framework for Interactive Brokers API
**Options Support:** Full options trading, chains, real-time data
**Key Features:**
- Jupyter notebook friendly
- Clean async/await interface
- Options chain scanning
- Greeks retrieval

### 9. michaelchu/optopsy - ⭐ 1.2k
**URL:** https://github.com/michaelchu/optopsy
**Language:** Python
**Description:** Nimble options backtesting library specifically for options strategies
**Strategies Supported:**
- Single-leg options
- Multi-leg spreads
- Straddles/Strangles
- Vertical spreads
**Key Features:**
- Pandas DataFrame input
- Customizable filters
- Performance statistics
- Works with any data source

### 10. PyPatel/Options-Trading-Strategies-in-Python - ⭐ 943
**URL:** https://github.com/PyPatel/Options-Trading-Strategies-in-Python
**Language:** Python
**Description:** Options trading strategies using technical indicators and quantitative methods
**Strategies:**
- Covered calls
- Protective puts
- Spreads
- Straddles

---

## Tier 3: Specialized Options Tools

### 11. brndnmtthws/thetagang - ⭐ 800+
**URL:** https://github.com/brndnmtthws/thetagang
**Language:** Python
**Description:** IBKR bot for "The Wheel" strategy - automated theta harvesting
**Strategy:** Cash-secured puts → Covered calls cycle
**Key Features:**
- Fully automated trading
- Configurable portfolio allocation
- Position management
- Roll management for ITM options
**Requirements:** IBKR account, IBC installation

### 12. rburkholder/trade-frame - ⭐ 614
**URL:** https://github.com/rburkholder/trade-frame
**Language:** C++17
**Description:** Library for testing equities, futures, ETFs & options with real-time data
**Options Support:** Built-in Greeks/IV calculation library
**Key Features:**
- DTN IQFeed integration
- Interactive Brokers execution
- High-performance C++ core

### 13. rgaveiga/optionlab - ⭐ 445
**URL:** https://github.com/rgaveiga/optionlab
**Language:** Python
**Description:** Python library for evaluating option trading strategies
**Key Features:**
- Strategy evaluation
- P&L visualization
- Risk metrics

### 14. mcf-long-short/ibkr-options-volatility-trading - ⭐ 322
**URL:** https://github.com/mcf-long-short/ibkr-options-volatility-trading
**Language:** Python
**Description:** Volatility trading using Long/Short Straddle strategies via IBKR
**Strategies:**
- Long straddle (high volatility play)
- Short straddle (low volatility play)
- Momentum-based entry signals

### 15. rahuljoshi44/GraphVega - ⭐ 298
**URL:** https://github.com/rahuljoshi44/GraphVega
**Language:** JavaScript
**Description:** Open Source Options Analytics Platform
**Key Features:**
- Visual options analysis
- Greeks visualization
- Strategy builder

### 16. tyrneh/options-implied-probability - ⭐ 289
**URL:** https://github.com/tyrneh/options-implied-probability
**Language:** Python
**Description:** Compute market expectations about asset future prices using options data
**Use Case:** Extract implied probability distributions from options chains

### 17. sirnfs/OptionSuite - ⭐ 275
**URL:** https://github.com/sirnfs/OptionSuite
**Language:** Python
**Description:** Option/stock strategy backtester and live trader framework
**Strategies:**
- Full strangle strategy
- Put vertical strategy
- Extensible framework

### 18. philipodonnell/paperbroker - ⭐ 275
**URL:** https://github.com/philipodonnell/paperbroker
**Language:** Python
**Description:** Open source simulated options brokerage for paper trading
**Key Features:**
- Paper trading simulation
- UI included
- Algorithmic interface
- Backtesting support

### 19. AnthonyBradford/optionmatrix - ⭐ 233
**URL:** https://github.com/AnthonyBradford/optionmatrix
**Language:** C++
**Description:** Financial Derivatives Calculator with 171+ pricing models
**Models Include:**
- Black-Scholes
- Binomial trees
- Monte Carlo
- Exotic options

### 20. srikar-kodakandla/fully-automated-nifty-options-trading - ⭐ 200
**URL:** https://github.com/srikar-kodakandla/fully-automated-nifty-options-trading
**Language:** Python
**Description:** Automated algo trading for Nifty options via Zerodha Kite
**Market:** Indian markets (NSE)

### 21. tastyware/tastytrade - ⭐ 196
**URL:** https://github.com/tastyware/tastytrade
**Language:** Python
**Description:** Unofficial sync/async Python SDK for Tastytrade
**Key Features:**
- Full API coverage
- Websocket streaming
- 100% typed with Pydantic
- MIT License

### 22. lambdaclass/options_backtester - ⭐ ~400
**URL:** https://github.com/lambdaclass/options_backtester
**Language:** Python
**Description:** Simple backtester for evaluating options strategies
**Key Features:**
- Multi-leg strategy creation
- Entry/exit filters
- Greeks integration
- Jupyter notebook examples

---

## Tier 4: Niche & Emerging Projects

### 23. romanrdgz/smartcondor - ⭐ <100
**URL:** https://github.com/romanrdgz/smartcondor
**Language:** Python
**Description:** Options strategy analysis tool
**Strategies:**
- Calendar spreads
- Diagonal spreads
- Iron condors
- Strangles
- Butterfly spreads

### 24. aicheung/0dte-trader - ⭐ <100
**URL:** https://github.com/aicheung/0dte-trader
**Language:** Python
**Description:** Trade 0DTE options algorithmically via IBKR API
**Strategies:**
- Bull/Bear Put spreads
- Iron Condor
- Iron Butterfly
- Butterfly
- Calendar/Diagonal spreads
**Key Feature:** Configurable DTE for far-dated legs

### 25. atkrish0/OpStrat - ⭐ <100
**URL:** https://github.com/atkrish0/OpStrat
**Language:** Python
**Description:** Implementations of widely used options trading strategies
**Strategies:**
- Iron condor spread
- Broken wing butterfly
- Bear calendar spread
- Diagonal bear spread

### 26. mattou78400/Option-Pricing-and-Strategies - ⭐ <100
**URL:** https://github.com/mattou78400/Option-Pricing-and-Strategies
**Language:** Python
**Description:** Option pricing with multiple models + strategy implementations
**Pricing Models:**
- Monte Carlo Simulation
- Black-Scholes
- Cox-Ross-Rubinstein
- Jarrow-Rudd
**Strategies:** Butterfly spread, Iron condor

### 27. ldt9/PyOptionTrader - ⭐ <100
**URL:** https://github.com/ldt9/PyOptionTrader
**Language:** Python
**Description:** Options trader using ib_insync
**Strategy:** Delta-neutral short strangle with IV arbitrage

---

## Curated Awesome Lists

### wilsonfreitas/awesome-quant - ⭐ 23.1k
**URL:** https://github.com/wilsonfreitas/awesome-quant
**Description:** Curated list of libraries, packages, and resources for Quants
**Options-Related Sections:**
- vollib - Options pricing and Greeks
- QuantLib - Comprehensive derivatives framework
- FinancePy - Financial derivatives pricing
- pysabr - SABR volatility model
- optlib - Options pricing library
- finoptions - R fOptions port to Python
- tf-quant-finance - Google's TensorFlow for finance

### wangzhe3224/awesome-systematic-trading - ⭐ 3k+
**URL:** https://github.com/wangzhe3224/awesome-systematic-trading
**Description:** Curated list for systematic trading including options

### WenchenLi/awesome-option-trading - ⭐ <100
**URL:** https://github.com/WenchenLi/awesome-option-trading
**Description:** Curated list specifically for option trading resources

### leoncuhk/awesome-quant-ai - ⭐ <100
**URL:** https://github.com/leoncuhk/awesome-quant-ai
**Description:** AI/ML applications in quantitative finance

---

## Broker API Libraries

| Library | Stars | Broker | Language | Options Support |
|---------|-------|--------|----------|-----------------|
| robin_stocks | 2k | Robinhood, TDA, Gemini | Python | Full |
| ib_async | 1.2k | Interactive Brokers | Python | Full |
| tastytrade | 196 | Tastytrade | Python | Full |
| pyrh | 500+ | Robinhood | Python | Limited |
| tda-api | 1k+ | TD Ameritrade | Python | Full |
| alpaca-trade-api | 1.5k+ | Alpaca | Python | Full |

---

## Options Pricing & Greeks Libraries

| Library | Stars | Language | Key Features |
|---------|-------|----------|--------------|
| **vollib/py_vollib** | 500+ | Python | Black-Scholes, Greeks, IV (Peter Jäckel's algorithm) |
| **py_vollib_vectorized** | 200+ | Python | Vectorized version, fastest IV calculation |
| **QuantLib** | 5k+ | C++/Python | Comprehensive derivatives framework, 171+ models |
| **dbrojas/optlib** | 100+ | Python | Generalized Black-Scholes, American options |
| **mcdallas/wallstreet** | 500+ | Python | Real-time stock and option data with Greeks |
| **bwrob/options-dataframes** | <100 | Python | Polars-based high-performance Greeks calculation |
| **ThePredictiveDev/Option-Pricing-Models** | <100 | Python | 7 pricing models including Heston, Merton Jump |

### Greeks Calculator Repositories

| Repository | Description |
|------------|-------------|
| AmirDehkordi/OptionGreeks | Educational - Delta, Gamma, Theta, Vega, Rho, Speed, Vomma, Charm, Vanna with 2D/3D visualizations |
| guiregueira/Greeks-Calculator | All Greeks including Epsilon, Veta, Zomma, Color, Ultima, Dual Delta/Gamma |
| gnagel/greeks | Ruby implementation |
| MattL922/greeks | JavaScript/Node.js implementation |

---

## Backtesting Frameworks

| Framework | Stars | Options-Specific | Language | Notes |
|-----------|-------|------------------|----------|-------|
| **optopsy** | 1.2k | Yes | Python | Purpose-built for options |
| **OptionSuite** | 275 | Yes | Python | Strangle, vertical strategies |
| **options_backtester** | 400+ | Yes | Python | Lambda Class, multi-leg |
| **backtesting.py** | 5.6k | Extend | Python | General, can add options |
| **Lean (QuantConnect)** | 10k+ | Yes | C#/Python | Full options support |
| **nautilus_trader** | 16.5k | Yes | Rust/Python | High-performance |
| **trade-frame** | 614 | Yes | C++ | Real-time with Greeks |

---

## Options Screeners & Scanners

| Tool | URL | Description |
|------|-----|-------------|
| **PutPremiumProcessor** | GitHub Topics | Custom scoring for cash-secured puts |
| **Robinhood-options-screener** | GitOffMyPorch | Percent win calculation, delta-based probability |
| **RyanElliott10/options_screener** | GitHub | C/Python hybrid, Yahoo Finance scraping |
| **Stocksera** | 746 stars | 60+ alternative data sources including options flow |

---

## Strategy-Specific Tools

### The Wheel Strategy
- **thetagang** (800+ stars) - Fully automated wheel bot for IBKR

### Volatility Trading
- **volatility-trading** (1.8k stars) - Sinclair's volatility estimators
- **ibkr-options-volatility-trading** (322 stars) - Straddle strategies

### Iron Condors / Butterflies
- **smartcondor** - Iron condors, butterflies, calendar spreads
- **OpStrat** - Iron condor, broken wing butterfly
- **Option-Pricing-and-Strategies** - Butterfly, iron condor implementations

### Calendar & Diagonal Spreads
- **smartcondor** - Calendar and diagonal spreads
- **0dte-trader** - Calendar/diagonal spread modes
- **calspread** (QuantRocket) - Futures calendar spreads

### 0DTE Trading
- **0dte-trader** - Multiple 0DTE strategies via IBKR

---

## Data Sources & APIs

### Free/Open Source
| Source | Type | Notes |
|--------|------|-------|
| Yahoo Finance | Delayed quotes | Via yfinance library |
| CBOE | Historical VIX | Direct download |
| Alpha Vantage | Limited free tier | Options chains available |

### Commercial (Recommended)
| Provider | Specialization | Cost Range |
|----------|---------------|------------|
| **Polygon.io** | Real-time options | $29-299/mo |
| **Tradier** | Options trading API | $0 (trading) |
| **CBOE DataShop** | Historical options | Custom pricing |
| **DTN IQFeed** | Real-time data | ~$100/mo |

### Data Pipeline Projects
- **chrischow/open-options-chains** - TD Ameritrade → PostgreSQL pipeline
- **trading-with-python/cboe.py** - CBOE data fetching utilities
- **finance-vix** (datasets) - VIX time series dataset

---

## Quick Reference: Top Picks by Use Case

### For Beginners
1. **optopsy** - Simple API, good documentation
2. **robin_stocks** - Easy Robinhood integration
3. **quant-trading** - Educational notebooks

### For Production Trading
1. **thetagang** - Automated wheel strategy
2. **ib_async** - Professional IBKR integration
3. **Lean/QuantConnect** - Enterprise-grade platform

### For Research & Analysis
1. **py_vollib_vectorized** - Fast Greeks calculation
2. **volatility-trading** - Volatility estimators
3. **awesome-quant** - Comprehensive resource list

### For Specific Strategies
| Strategy | Best Tool |
|----------|-----------|
| Wheel | thetagang |
| Straddles/Strangles | ibkr-options-volatility-trading |
| Iron Condors | smartcondor, OpStrat |
| Calendar Spreads | 0dte-trader, smartcondor |
| 0DTE | 0dte-trader |
| Volatility | volatility-trading |

---

## Contributing

This document should be updated periodically as new repositories emerge and star counts change. Key GitHub topics to monitor:
- `options-trading`
- `options-strategies`
- `options-pricing`
- `backtesting`
- `quantitative-finance`

---

## Sources

- [GitHub Topics: options-trading](https://github.com/topics/options-trading)
- [GitHub Topics: options-strategies](https://github.com/topics/options-strategies)
- [awesome-quant](https://github.com/wilsonfreitas/awesome-quant)
- [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading)

---

*Last Updated: 2025-11-28*
