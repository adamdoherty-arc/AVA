import logging
import robin_stocks.robinhood as rh
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from backend.config import get_settings

logger = logging.getLogger(__name__)

class PortfolioService:
    def __init__(self):
        self.settings = get_settings()
        self._logged_in = False

    def _ensure_login(self):
        """Ensure Robinhood is logged in."""
        if self._logged_in:
            return

        username = os.getenv('ROBINHOOD_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD')

        if not username or not password:
            logger.error("Robinhood credentials not found")
            raise ValueError("Robinhood credentials not configured")

        try:
            rh.login(username=username, password=password, store_session=True)
            self._logged_in = True
            logger.info("Logged in to Robinhood")
        except Exception as e:
            logger.error(f"Robinhood login failed: {e}")
            raise

    async def get_positions(self) -> Dict[str, Any]:
        """
        Get all active positions (stocks and options) with P/L and metrics.
        """
        self._ensure_login()
        
        try:
            # 1. Get Account Info
            portfolio = rh.profiles.load_portfolio_profile()
            account = rh.profiles.load_account_profile()
            
            total_equity = float(portfolio.get('equity', 0)) if portfolio else 0
            buying_power = float(account.get('buying_power', 0)) if account else 0

            # 2. Get Stock Positions
            stock_positions = self._get_stock_positions()
            
            # 3. Get Option Positions
            option_positions = self._get_option_positions()

            return {
                "summary": {
                    "total_equity": total_equity,
                    "buying_power": buying_power,
                    "total_positions": len(stock_positions) + len(option_positions)
                },
                "stocks": stock_positions,
                "options": option_positions
            }

        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    def _get_stock_positions(self) -> List[Dict[str, Any]]:
        """Fetch and process stock positions."""
        raw_positions = rh.get_open_stock_positions()
        processed = []

        for pos in raw_positions:
            quantity = float(pos.get('quantity', 0))
            if quantity == 0:
                continue

            # Get symbol
            instrument_url = pos.get('instrument')
            instrument_data = rh.get_instrument_by_url(instrument_url)
            symbol = instrument_data.get('symbol')

            # Get prices
            avg_buy_price = float(pos.get('average_buy_price', 0))
            current_price = float(rh.get_latest_price(symbol)[0])

            # Calculate metrics
            cost_basis = avg_buy_price * quantity
            current_value = current_price * quantity
            pl = current_value - cost_basis
            pl_pct = (pl / cost_basis * 100) if cost_basis > 0 else 0

            processed.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_buy_price": avg_buy_price,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "current_value": current_value,
                "pl": pl,
                "pl_pct": pl_pct,
                "type": "stock"
            })
        
        return processed

    def _get_option_positions(self) -> List[Dict[str, Any]]:
        """Fetch and process option positions with Greeks."""
        raw_positions = rh.get_open_option_positions()
        processed = []

        for pos in raw_positions:
            # Get option details
            opt_id = pos.get('option_id')
            opt_data = rh.get_option_instrument_data_by_id(opt_id)

            symbol = opt_data.get('chain_symbol')
            strike = float(opt_data.get('strike_price'))
            exp_date = opt_data.get('expiration_date')
            opt_type = opt_data.get('type')

            # Position details
            position_type = pos.get('type')  # 'long' or 'short'
            quantity = float(pos.get('quantity', 0))
            avg_price = float(pos.get('average_price', 0)) / 100 # Convert to per share

            # Market data with Greeks
            market_data = rh.get_option_market_data_by_id(opt_id)
            current_price = 0
            delta = theta = gamma = vega = iv = 0

            if market_data and len(market_data) > 0:
                md = market_data[0]
                current_price = float(md.get('adjusted_mark_price', 0))
                delta = float(md.get('delta', 0) or 0)
                theta = float(md.get('theta', 0) or 0)
                gamma = float(md.get('gamma', 0) or 0)
                vega = float(md.get('vega', 0) or 0)
                iv = float(md.get('implied_volatility', 0) or 0)

            # Calculate days to expiration
            dte = 0
            if exp_date:
                try:
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days
                except:
                    pass

            # Calculations
            total_premium = avg_price * 100 * quantity
            current_value = current_price * 100 * quantity

            if position_type == 'short':
                pl = total_premium - current_value
                # For short positions, flip delta sign
                delta = -delta
            else:
                pl = current_value - total_premium

            # Determine Strategy
            strategy = "Other"
            if position_type == 'short':
                strategy = "CSP" if opt_type == 'put' else "CC"
            elif position_type == 'long':
                strategy = f"Long {opt_type.title()}"

            # Calculate break-even
            if opt_type == 'put':
                breakeven = strike - avg_price if position_type == 'short' else strike - avg_price
            else:
                breakeven = strike + avg_price if position_type == 'short' else strike + avg_price

            processed.append({
                "symbol": symbol,
                "strategy": strategy,
                "type": position_type,
                "option_type": opt_type,
                "strike": strike,
                "expiration": exp_date,
                "dte": dte,
                "quantity": quantity,
                "avg_price": avg_price * 100, # Per contract
                "current_price": current_price * 100, # Per contract
                "total_premium": total_premium,
                "current_value": current_value,
                "pl": pl,
                "pl_pct": (pl / total_premium * 100) if total_premium > 0 else 0,
                "breakeven": breakeven,
                "greeks": {
                    "delta": round(delta * 100, 2),  # As percentage
                    "theta": round(theta * 100, 2),  # Per day per contract
                    "gamma": round(gamma * 100, 4),
                    "vega": round(vega * 100, 2),
                    "iv": round(iv * 100, 1)  # As percentage
                }
            })

        return processed

# Singleton
_portfolio_service = None

def get_portfolio_service() -> PortfolioService:
    global _portfolio_service
    if _portfolio_service is None:
        _portfolio_service = PortfolioService()
    return _portfolio_service
