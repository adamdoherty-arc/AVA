-- Options Wheel Strategy Trading System Database Schema
-- PostgreSQL with TimescaleDB extension for time-series data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User management
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    risk_tolerance VARCHAR(20) DEFAULT 'moderate', -- conservative, moderate, aggressive
    max_portfolio_risk DECIMAL(5,4) DEFAULT 0.02, -- 2% max risk per trade
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Stock symbols and metadata
CREATE TABLE stocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT true,
    is_optionable BOOLEAN DEFAULT false,
    average_volume BIGINT,
    beta DECIMAL(6,3),
    dividend_yield DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User watchlists
CREATE TABLE watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, name)
);

-- Watchlist items
CREATE TABLE watchlist_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    watchlist_id UUID NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    target_entry_price DECIMAL(10,2),
    target_premium_yield DECIMAL(5,4),
    max_position_size INTEGER,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(watchlist_id, stock_id)
);

-- Stock price data (time-series)
CREATE TABLE stock_prices (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    open_price DECIMAL(10,2) NOT NULL,
    high_price DECIMAL(10,2) NOT NULL,
    low_price DECIMAL(10,2) NOT NULL,
    close_price DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close DECIMAL(10,2),
    PRIMARY KEY (stock_id, time)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('stock_prices', 'time', 'stock_id', number_partitions => 4);

-- Options chains data
CREATE TABLE options_chains (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    expiration_date DATE NOT NULL,
    strike_price DECIMAL(8,2) NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('CALL', 'PUT')),
    bid_price DECIMAL(6,2),
    ask_price DECIMAL(6,2),
    last_price DECIMAL(6,2),
    volume INTEGER DEFAULT 0,
    open_interest INTEGER DEFAULT 0,
    implied_volatility DECIMAL(6,4),
    delta DECIMAL(6,4),
    gamma DECIMAL(6,4),
    theta DECIMAL(6,4),
    vega DECIMAL(6,4),
    rho DECIMAL(6,4),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(stock_id, expiration_date, strike_price, option_type)
);

-- Trading accounts
CREATE TABLE trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_name VARCHAR(100) NOT NULL,
    broker VARCHAR(50),
    account_number VARCHAR(50),
    account_type VARCHAR(20) DEFAULT 'margin', -- cash, margin, ira
    buying_power DECIMAL(12,2) DEFAULT 0,
    total_value DECIMAL(12,2) DEFAULT 0,
    cash_balance DECIMAL(12,2) DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Positions tracking
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES trading_accounts(id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    position_type VARCHAR(20) NOT NULL, -- stock, put, call
    strategy_type VARCHAR(20) NOT NULL, -- csp, cc, wheel
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(8,2) NOT NULL,
    current_price DECIMAL(8,2),
    strike_price DECIMAL(8,2), -- for options
    expiration_date DATE, -- for options
    opening_premium DECIMAL(8,2), -- premium collected/paid
    current_premium DECIMAL(8,2),
    unrealized_pnl DECIMAL(10,2) DEFAULT 0,
    realized_pnl DECIMAL(10,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open', -- open, closed, assigned, expired
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    notes TEXT
);

-- Wheel strategy cycles tracking
CREATE TABLE wheel_cycles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    cycle_number INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'active', -- active, completed, stopped
    total_premium_collected DECIMAL(10,2) DEFAULT 0,
    total_pnl DECIMAL(10,2) DEFAULT 0,
    stock_assignment_price DECIMAL(8,2),
    target_exit_price DECIMAL(8,2),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(user_id, stock_id, cycle_number)
);

-- Link positions to wheel cycles
ALTER TABLE positions ADD COLUMN wheel_cycle_id UUID REFERENCES wheel_cycles(id);

-- Trade executions log
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
    trade_type VARCHAR(20) NOT NULL, -- buy, sell, sell_to_open, buy_to_close
    quantity INTEGER NOT NULL,
    price DECIMAL(8,2) NOT NULL,
    commission DECIMAL(6,2) DEFAULT 0,
    fees DECIMAL(6,2) DEFAULT 0,
    total_amount DECIMAL(10,2) NOT NULL,
    execution_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    order_id VARCHAR(100), -- broker order ID
    notes TEXT
);

-- Strategy signals and recommendations
CREATE TABLE strategy_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    signal_type VARCHAR(20) NOT NULL, -- csp_opportunity, cc_opportunity, close_position
    strategy VARCHAR(20) NOT NULL, -- wheel, csp, cc
    strike_price DECIMAL(8,2),
    expiration_date DATE,
    premium_yield DECIMAL(5,4),
    probability_profit DECIMAL(5,4),
    risk_reward_ratio DECIMAL(6,3),
    confidence_score DECIMAL(3,2), -- 0.0 to 1.0
    reasoning TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Price alerts configuration
CREATE TABLE price_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    stock_id UUID NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    alert_type VARCHAR(20) NOT NULL, -- price_above, price_below, volume_spike, iv_spike
    threshold_value DECIMAL(10,4) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    notification_methods TEXT[], -- ['email', 'sms', 'webhook']
    last_triggered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alert events log
CREATE TABLE alert_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id UUID NOT NULL REFERENCES price_alerts(id) ON DELETE CASCADE,
    triggered_value DECIMAL(10,4) NOT NULL,
    message TEXT NOT NULL,
    notification_sent BOOLEAN DEFAULT false,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk metrics tracking
CREATE TABLE risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID REFERENCES trading_accounts(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    portfolio_value DECIMAL(12,2),
    total_delta DECIMAL(8,2),
    total_theta DECIMAL(8,2),
    total_vega DECIMAL(8,2),
    buying_power_used DECIMAL(12,2),
    margin_requirement DECIMAL(12,2),
    var_1_day DECIMAL(10,2), -- Value at Risk 1 day
    var_30_day DECIMAL(10,2), -- Value at Risk 30 days
    max_drawdown DECIMAL(5,4),
    sharpe_ratio DECIMAL(6,3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, account_id, metric_date)
);

-- System configuration
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance optimization

-- Stock price data indexes
CREATE INDEX IF NOT EXISTS idx_stock_prices_time ON stock_prices (time DESC);
CREATE INDEX IF NOT EXISTS idx_stock_prices_stock_id ON stock_prices (stock_id);

-- Stock metadata indexes - critical for filtering operations
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks (symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_active ON stocks (is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_stocks_sector ON stocks (sector) WHERE sector IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_stocks_industry ON stocks (industry) WHERE industry IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_stocks_optionable ON stocks (is_optionable) WHERE is_optionable = true;

-- Options chains indexes - optimized for common query patterns
CREATE INDEX IF NOT EXISTS idx_options_chains_stock_expiry ON options_chains (stock_id, expiration_date);
CREATE INDEX IF NOT EXISTS idx_options_chains_strike ON options_chains (strike_price);
CREATE INDEX IF NOT EXISTS idx_options_symbol_expiry ON options_chains (stock_id, expiration_date, option_type);
CREATE INDEX IF NOT EXISTS idx_options_strike_type ON options_chains (strike_price, option_type);

-- Positions indexes - optimized for user queries and status filtering
CREATE INDEX IF NOT EXISTS idx_positions_user_status ON positions (user_id, status);
CREATE INDEX IF NOT EXISTS idx_positions_stock_id ON positions (stock_id);
CREATE INDEX IF NOT EXISTS idx_positions_user_symbol ON positions (user_id, stock_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status) WHERE status IN ('open', 'assigned', 'expired');

-- Trade execution indexes
CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades (position_id);
CREATE INDEX IF NOT EXISTS idx_trades_execution_time ON trades (execution_time DESC);

-- Strategy and alerts indexes
CREATE INDEX IF NOT EXISTS idx_strategy_signals_active ON strategy_signals (is_active, created_at DESC) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_price_alerts_active ON price_alerts (is_active, stock_id) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_alert_events_triggered_at ON alert_events (triggered_at DESC);

-- Risk metrics indexes
CREATE INDEX IF NOT EXISTS idx_risk_metrics_user_date ON risk_metrics (user_id, metric_date DESC);

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('market_hours', '{"open": "09:30", "close": "16:00", "timezone": "America/New_York"}', 'Market trading hours'),
('risk_limits', '{"max_position_size_pct": 0.05, "max_sector_exposure": 0.25, "max_single_stock_exposure": 0.10}', 'Default risk limits'),
('wheel_strategy_params', '{"min_dte": 15, "max_dte": 45, "target_delta": 0.30, "min_premium_yield": 0.01}', 'Wheel strategy parameters'),
('data_retention', '{"price_data_days": 2555, "trade_data_years": 7, "alert_events_days": 90}', 'Data retention policies');

-- Views for common queries
CREATE VIEW v_current_positions AS
SELECT
    p.*,
    s.symbol,
    s.company_name,
    ta.account_name,
    wc.cycle_number,
    (p.quantity * p.current_price) as current_value,
    ((p.current_price - p.entry_price) * p.quantity) as unrealized_pnl_calc
FROM positions p
JOIN stocks s ON p.stock_id = s.id
JOIN trading_accounts ta ON p.account_id = ta.id
LEFT JOIN wheel_cycles wc ON p.wheel_cycle_id = wc.id
WHERE p.status = 'open';

CREATE VIEW v_wheel_performance AS
SELECT
    wc.*,
    s.symbol,
    s.company_name,
    COUNT(p.id) as total_positions,
    SUM(CASE WHEN p.position_type = 'put' THEN p.opening_premium ELSE 0 END) as csp_premium,
    SUM(CASE WHEN p.position_type = 'call' THEN p.opening_premium ELSE 0 END) as cc_premium,
    AVG(p.realized_pnl) as avg_trade_pnl
FROM wheel_cycles wc
JOIN stocks s ON wc.stock_id = s.id
LEFT JOIN positions p ON wc.id = p.wheel_cycle_id
GROUP BY wc.id, s.symbol, s.company_name;

-- Functions for common calculations
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    position_id UUID,
    current_price DECIMAL(8,2)
) RETURNS DECIMAL(10,2) AS $$
DECLARE
    pos positions%ROWTYPE;
    pnl DECIMAL(10,2);
BEGIN
    SELECT * INTO pos FROM positions WHERE id = position_id;

    CASE pos.position_type
        WHEN 'stock' THEN
            pnl := (current_price - pos.entry_price) * pos.quantity;
        WHEN 'put', 'call' THEN
            -- For options, calculate based on premium changes
            pnl := (pos.opening_premium - current_price) * pos.quantity * 100;
    END CASE;

    RETURN pnl;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update position P&L when prices change
CREATE OR REPLACE FUNCTION update_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE positions
    SET
        current_price = NEW.close_price,
        unrealized_pnl = calculate_position_pnl(id, NEW.close_price),
        updated_at = NOW()
    WHERE stock_id = NEW.stock_id AND status = 'open';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_position_pnl
    AFTER INSERT OR UPDATE ON stock_prices
    FOR EACH ROW
    EXECUTE FUNCTION update_position_pnl();

-- Data retention policy (TimescaleDB)
SELECT add_retention_policy('stock_prices', INTERVAL '7 years');