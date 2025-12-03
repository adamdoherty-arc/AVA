# Positions Feature Requirements

## Overview
The Positions page displays the user's portfolio holdings from Robinhood, including stocks and options with real-time pricing, P&L calculations, and Greeks for options.

## User Stories

### US-POS-001: View All Positions
**As a** trader
**I want to** see all my open positions in one place
**So that** I can monitor my portfolio at a glance

**Acceptance Criteria:**
- Display all stock positions with symbol, quantity, avg price, current price, P&L
- Display all option positions with symbol, type, strike, expiration, DTE, P&L
- Show summary card with total equity
- Data refreshes automatically every 30 seconds during market hours

### US-POS-002: View Position Details
**As a** trader
**I want to** see detailed information about each position
**So that** I can make informed trading decisions

**Acceptance Criteria:**
- Stocks show: cost basis, current value, day change
- Options show: Greeks (delta, gamma, theta, vega), IV, volume
- P&L is color-coded (green for profit, red for loss)

### US-POS-003: Sync Portfolio
**As a** trader
**I want to** manually sync my portfolio from Robinhood
**So that** I can ensure data is up-to-date

**Acceptance Criteria:**
- Sync button triggers portfolio refresh
- Loading state shown during sync
- Success/error feedback displayed
- Data updates immediately after sync

### US-POS-004: P&L Calculations
**As a** trader
**I want** accurate P&L calculations
**So that** I know my true profit/loss

**Acceptance Criteria:**
- Stock P&L: (current_price - avg_price) * quantity
- Option P&L: (current_price - avg_price) * quantity * 100
- Percentage P&L calculated correctly
- Total portfolio P&L matches sum of individual positions

### US-POS-005: Options Greeks Display
**As an** options trader
**I want to** see Greeks for my option positions
**So that** I can assess risk exposure

**Acceptance Criteria:**
- Delta displayed for all options
- Theta (time decay) shown and highlighted when high
- All Greeks use Black-Scholes calculation
- Greeks update with price changes

## Functional Requirements

### FR-POS-001: Data Source
- Primary source: Robinhood API
- Fallback: Cached data from database
- Max staleness: 5 minutes during market hours

### FR-POS-002: P&L Calculation
- Use FIFO cost basis method
- Include dividends in stock P&L (if applicable)
- Options use 100x multiplier

### FR-POS-003: DTE Calculation
- Days To Expiration = expiration_date - today
- 0 DTE on expiration day
- Highlight options with DTE <= 7

### FR-POS-004: Greeks Calculation
- Use Black-Scholes model
- Risk-free rate: Current Fed Funds Rate
- IV: From market data or calculated

## Non-Functional Requirements

### NFR-POS-001: Performance
- Page load time < 2 seconds
- Data refresh < 500ms
- No blocking during sync

### NFR-POS-002: Reliability
- Graceful degradation if Robinhood unavailable
- Show cached data with "stale" indicator
- Retry logic for failed API calls

### NFR-POS-003: Accuracy
- P&L calculations accurate to $0.01
- Greeks accurate to 0.01
- DTE must be exact

## Edge Cases

1. **Zero positions**: Show empty state with message
2. **Expired options**: Filter out or mark as expired
3. **Missing Greeks**: Show "-" with tooltip explaining
4. **API timeout**: Show cached data with warning
5. **Market closed**: Show last known prices

## Dependencies
- Robinhood API authentication
- Database for caching
- Dashboard feature (portfolio totals must match)
