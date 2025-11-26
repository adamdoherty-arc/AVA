-- Trade History Table for tracking closed positions
-- This tracks cash-secured puts that are closed early or held to expiration

CREATE TABLE IF NOT EXISTS trade_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    strategy_type VARCHAR(20) DEFAULT 'cash_secured_put',

    -- Opening trade details
    open_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    strike_price DECIMAL(10, 2) NOT NULL,
    expiration_date DATE NOT NULL,
    premium_collected DECIMAL(10, 2) NOT NULL,
    contracts INTEGER DEFAULT 1,
    dte_at_open INTEGER,

    -- Closing trade details (NULL if still open)
    close_date TIMESTAMP WITH TIME ZONE,
    close_price DECIMAL(10, 2),
    close_reason VARCHAR(20), -- 'early_close', 'expiration', 'assignment'

    -- Calculated P&L
    days_held INTEGER,
    profit_loss DECIMAL(10, 2),
    profit_loss_percent DECIMAL(10, 4),
    annualized_return DECIMAL(10, 4),

    -- Status
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'assigned'

    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_status ON trade_history(status);
CREATE INDEX IF NOT EXISTS idx_trade_history_close_date ON trade_history(close_date DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_trade_history_open_date ON trade_history(open_date DESC);

-- Comments
COMMENT ON TABLE trade_history IS 'Tracks all option trades (open and closed positions)';
COMMENT ON COLUMN trade_history.strategy_type IS 'Type of strategy: cash_secured_put, covered_call, etc';
COMMENT ON COLUMN trade_history.close_reason IS 'Why position was closed: early_close, expiration, assignment';
COMMENT ON COLUMN trade_history.profit_loss IS 'Net profit/loss in dollars';
COMMENT ON COLUMN trade_history.annualized_return IS 'Annualized return percentage';
