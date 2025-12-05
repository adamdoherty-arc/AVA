"""
Robinhood Order Execution Service
=================================

Execute options orders via Robinhood API.

Features:
- Single-leg options orders (buy/sell calls/puts)
- Multi-leg spread orders (verticals, iron condors, straddles)
- Order management (cancel, modify, roll)
- Paper trading mode for testing
- Risk checks before execution
- Order confirmation and logging

Author: AVA Trading Platform
Created: 2025-11-28
"""

import robin_stocks.robinhood as rh
import os
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from src.services.robinhood_client import get_robinhood_client, RobinhoodClient
from src.services.rate_limiter import rate_limit


class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'


class OrderSide(Enum):
    BUY_TO_OPEN = 'buy_to_open'
    BUY_TO_CLOSE = 'buy_to_close'
    SELL_TO_OPEN = 'sell_to_open'
    SELL_TO_CLOSE = 'sell_to_close'


class OrderStatus(Enum):
    PENDING = 'pending'
    QUEUED = 'queued'
    CONFIRMED = 'confirmed'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    FAILED = 'failed'
    REJECTED = 'rejected'


class SpreadType(Enum):
    SINGLE = 'single'
    VERTICAL = 'vertical'
    IRON_CONDOR = 'iron_condor'
    IRON_BUTTERFLY = 'iron_butterfly'
    STRADDLE = 'straddle'
    STRANGLE = 'strangle'
    CALENDAR = 'calendar'
    DIAGONAL = 'diagonal'
    BUTTERFLY = 'butterfly'
    CUSTOM = 'custom'


@dataclass
class OptionLeg:
    """Represents a single leg of an options order"""
    symbol: str
    strike: float
    expiration: str  # YYYY-MM-DD
    option_type: str  # 'call' or 'put'
    side: OrderSide
    quantity: int
    price: Optional[float] = None  # Limit price per contract

    @property
    def is_buy(self) -> bool:
        return self.side in [OrderSide.BUY_TO_OPEN, OrderSide.BUY_TO_CLOSE]

    @property
    def is_opening(self) -> bool:
        return self.side in [OrderSide.BUY_TO_OPEN, OrderSide.SELL_TO_OPEN]


@dataclass
class OrderResult:
    """Result of an order execution"""
    success: bool
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    message: str = ''
    filled_quantity: int = 0
    average_price: float = 0.0
    legs: List[Dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Dict = field(default_factory=dict)


class RobinhoodOrderService:
    """
    Execute options orders via Robinhood

    Features:
    - Single and multi-leg orders
    - Paper trading mode
    - Risk validation
    - Order tracking
    """

    def __init__(self, paper_trading: bool = True):
        """
        Initialize order service.

        Args:
            paper_trading: If True, simulates orders without execution
        """
        self.client = get_robinhood_client()
        self.paper_trading = paper_trading
        self._order_history: List[OrderResult] = []
        self._lock = threading.Lock()

        # Risk limits
        self.max_single_order_value = float(os.getenv('MAX_SINGLE_ORDER_VALUE', '10000'))
        self.max_contracts_per_order = int(os.getenv('MAX_CONTRACTS_PER_ORDER', '10'))
        self.require_confirmation = os.getenv('REQUIRE_ORDER_CONFIRMATION', 'true').lower() == 'true'

        logger.info(f"Order service initialized (paper_trading={paper_trading})")

    def _ensure_logged_in(self) -> None:
        """Ensure Robinhood client is logged in"""
        if not self.client.logged_in:
            if not self.client.login():
                raise RuntimeError("Failed to login to Robinhood")

    def _validate_risk_limits(self, legs: List[OptionLeg], estimated_value: float) -> Tuple[bool, str]:
        """
        Validate order against risk limits.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check max order value
        if estimated_value > self.max_single_order_value:
            return False, f"Order value ${estimated_value:.2f} exceeds limit ${self.max_single_order_value:.2f}"

        # Check max contracts
        total_contracts = sum(leg.quantity for leg in legs)
        if total_contracts > self.max_contracts_per_order:
            return False, f"Total contracts {total_contracts} exceeds limit {self.max_contracts_per_order}"

        # Check for expiring options (warn if < 1 day)
        today = date.today()
        for leg in legs:
            exp_date = datetime.strptime(leg.expiration, '%Y-%m-%d').date()
            if exp_date <= today:
                return False, f"Cannot trade expired options ({leg.expiration})"
            if (exp_date - today).days == 0:
                logger.warning(f"Trading 0DTE options - high risk!")

        return True, ""

    def _get_option_instrument(self, symbol: str, strike: float, expiration: str, option_type: str) -> Optional[str]:
        """Get the Robinhood option instrument URL"""
        try:
            options = rh.options.find_options_by_expiration_and_strike(
                symbol,
                expirationDate=expiration,
                strikePrice=str(strike),
                optionType=option_type
            )

            if options and len(options) > 0:
                return options[0].get('url')

            return None
        except Exception as e:
            logger.error(f"Error finding option instrument: {e}")
            return None

    def _simulate_order(self, legs: List[OptionLeg], spread_type: SpreadType) -> OrderResult:
        """Simulate order execution for paper trading"""
        logger.info(f"[PAPER TRADE] Simulating {spread_type.value} order with {len(legs)} leg(s)")

        # Generate fake order ID
        order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self._order_history)}"

        # Calculate simulated fill
        total_value = 0
        leg_details = []

        for leg in legs:
            # Estimate price if not provided
            price = leg.price or 1.00  # Default $1 for simulation

            leg_value = price * leg.quantity * 100
            if leg.is_buy:
                total_value -= leg_value
            else:
                total_value += leg_value

            leg_details.append({
                'symbol': leg.symbol,
                'strike': leg.strike,
                'expiration': leg.expiration,
                'option_type': leg.option_type,
                'side': leg.side.value,
                'quantity': leg.quantity,
                'filled_price': price
            })

            logger.info(f"  [PAPER] {leg.side.value}: {leg.symbol} {leg.strike} {leg.option_type} x{leg.quantity} @ ${price:.2f}")

        result = OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.FILLED,
            message=f"Paper trade executed: {spread_type.value}",
            filled_quantity=legs[0].quantity,
            average_price=abs(total_value) / (legs[0].quantity * 100) if legs else 0,
            legs=leg_details
        )

        with self._lock:
            self._order_history.append(result)

        logger.info(f"[PAPER TRADE] Order {order_id} filled. Net value: ${total_value:.2f}")

        return result

    # =========================================================================
    # SINGLE LEG ORDERS
    # =========================================================================

    @rate_limit("robinhood", tokens=3)
    def buy_option(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str,  # 'call' or 'put'
        quantity: int,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: str = 'gfd'
    ) -> OrderResult:
        """
        Buy to open an option contract.

        Args:
            symbol: Underlying symbol (e.g., 'AAPL')
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'
            quantity: Number of contracts
            price: Limit price per contract (required for limit orders)
            order_type: Order type (market, limit)
            time_in_force: 'gfd' (good for day) or 'gtc' (good til cancelled)

        Returns:
            OrderResult with execution details
        """
        leg = OptionLeg(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            side=OrderSide.BUY_TO_OPEN,
            quantity=quantity,
            price=price
        )

        return self._execute_single_leg_order(leg, order_type, time_in_force)

    @rate_limit("robinhood", tokens=3)
    def sell_option(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: str = 'gfd',
        is_closing: bool = False
    ) -> OrderResult:
        """
        Sell an option contract (to open or to close).

        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiration: Expiration date (YYYY-MM-DD)
            option_type: 'call' or 'put'
            quantity: Number of contracts
            price: Limit price per contract
            order_type: Order type
            time_in_force: Duration
            is_closing: True if selling to close an existing position

        Returns:
            OrderResult with execution details
        """
        side = OrderSide.SELL_TO_CLOSE if is_closing else OrderSide.SELL_TO_OPEN

        leg = OptionLeg(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            side=side,
            quantity=quantity,
            price=price
        )

        return self._execute_single_leg_order(leg, order_type, time_in_force)

    def _execute_single_leg_order(
        self,
        leg: OptionLeg,
        order_type: OrderType,
        time_in_force: str
    ) -> OrderResult:
        """Execute a single leg option order"""

        # Validate risk limits
        estimated_value = (leg.price or 1.0) * leg.quantity * 100
        is_valid, error_msg = self._validate_risk_limits([leg], estimated_value)

        if not is_valid:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=f"Risk validation failed: {error_msg}"
            )

        # Paper trading mode
        if self.paper_trading:
            return self._simulate_order([leg], SpreadType.SINGLE)

        # Live execution
        self._ensure_logged_in()

        try:
            # Map OrderSide to Robinhood's expected values
            position_effect = 'open' if leg.is_opening else 'close'
            side = 'buy' if leg.is_buy else 'sell'

            # Execute order
            if order_type == OrderType.LIMIT and leg.price:
                order = rh.orders.order_buy_option_limit(
                    positionEffect=position_effect,
                    creditOrDebit='debit' if leg.is_buy else 'credit',
                    price=leg.price,
                    symbol=leg.symbol,
                    quantity=leg.quantity,
                    expirationDate=leg.expiration,
                    strike=leg.strike,
                    optionType=leg.option_type,
                    timeInForce=time_in_force
                ) if leg.is_buy else rh.orders.order_sell_option_limit(
                    positionEffect=position_effect,
                    creditOrDebit='credit',
                    price=leg.price,
                    symbol=leg.symbol,
                    quantity=leg.quantity,
                    expirationDate=leg.expiration,
                    strike=leg.strike,
                    optionType=leg.option_type,
                    timeInForce=time_in_force
                )
            else:
                # Market order
                order = rh.orders.order_buy_option_stop_limit(
                    symbol=leg.symbol,
                    quantity=leg.quantity,
                    expirationDate=leg.expiration,
                    strike=leg.strike,
                    optionType=leg.option_type,
                    timeInForce=time_in_force
                ) if leg.is_buy else rh.orders.order_sell_option_stop_limit(
                    symbol=leg.symbol,
                    quantity=leg.quantity,
                    expirationDate=leg.expiration,
                    strike=leg.strike,
                    optionType=leg.option_type,
                    timeInForce=time_in_force
                )

            if order and order.get('id'):
                result = OrderResult(
                    success=True,
                    order_id=order.get('id'),
                    status=OrderStatus.QUEUED,
                    message=f"Order placed successfully",
                    legs=[{
                        'symbol': leg.symbol,
                        'strike': leg.strike,
                        'expiration': leg.expiration,
                        'option_type': leg.option_type,
                        'side': leg.side.value,
                        'quantity': leg.quantity
                    }],
                    raw_response=order
                )

                with self._lock:
                    self._order_history.append(result)

                logger.info(f"Order placed: {result.order_id}")
                return result
            else:
                error_msg = order.get('detail', 'Unknown error') if order else 'No response'
                return OrderResult(
                    success=False,
                    status=OrderStatus.FAILED,
                    message=f"Order failed: {error_msg}",
                    raw_response=order or {}
                )

        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=f"Execution error: {str(e)}"
            )

    # =========================================================================
    # MULTI-LEG SPREAD ORDERS
    # =========================================================================

    @rate_limit("robinhood", tokens=5)
    def execute_spread(
        self,
        legs: List[OptionLeg],
        spread_type: SpreadType = SpreadType.CUSTOM,
        net_price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        time_in_force: str = 'gfd'
    ) -> OrderResult:
        """
        Execute a multi-leg spread order.

        Args:
            legs: List of OptionLeg objects defining the spread
            spread_type: Type of spread for logging
            net_price: Net credit (positive) or debit (negative) for the spread
            order_type: Order type
            time_in_force: Duration

        Returns:
            OrderResult with execution details
        """
        if not legs:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message="No legs provided"
            )

        # Validate risk limits
        estimated_value = abs(net_price or 1.0) * legs[0].quantity * 100
        is_valid, error_msg = self._validate_risk_limits(legs, estimated_value)

        if not is_valid:
            return OrderResult(
                success=False,
                status=OrderStatus.REJECTED,
                message=f"Risk validation failed: {error_msg}"
            )

        # Paper trading mode
        if self.paper_trading:
            return self._simulate_order(legs, spread_type)

        # Live execution
        self._ensure_logged_in()

        try:
            # Build spread legs for Robinhood
            spread_legs = []

            for leg in legs:
                instrument = self._get_option_instrument(
                    leg.symbol, leg.strike, leg.expiration, leg.option_type
                )

                if not instrument:
                    return OrderResult(
                        success=False,
                        status=OrderStatus.FAILED,
                        message=f"Could not find option: {leg.symbol} {leg.strike} {leg.option_type} {leg.expiration}"
                    )

                position_effect = 'open' if leg.is_opening else 'close'
                side = 'buy' if leg.is_buy else 'sell'

                spread_legs.append({
                    'option': instrument,
                    'side': side,
                    'position_effect': position_effect,
                    'ratio_quantity': leg.quantity
                })

            # Determine credit or debit
            if net_price is None:
                credit_or_debit = 'credit'  # Default for selling spreads
            elif net_price >= 0:
                credit_or_debit = 'credit'
            else:
                credit_or_debit = 'debit'

            # Execute spread order
            order = rh.orders.order_option_spread(
                direction=credit_or_debit,
                price=abs(net_price) if net_price else None,
                symbol=legs[0].symbol,
                quantity=legs[0].quantity,
                spread=spread_legs,
                timeInForce=time_in_force
            )

            if order and order.get('id'):
                result = OrderResult(
                    success=True,
                    order_id=order.get('id'),
                    status=OrderStatus.QUEUED,
                    message=f"{spread_type.value} order placed successfully",
                    legs=[{
                        'symbol': l.symbol,
                        'strike': l.strike,
                        'expiration': l.expiration,
                        'option_type': l.option_type,
                        'side': l.side.value,
                        'quantity': l.quantity
                    } for l in legs],
                    raw_response=order
                )

                with self._lock:
                    self._order_history.append(result)

                logger.info(f"Spread order placed: {result.order_id}")
                return result
            else:
                error_msg = order.get('detail', 'Unknown error') if order else 'No response'
                return OrderResult(
                    success=False,
                    status=OrderStatus.FAILED,
                    message=f"Spread order failed: {error_msg}",
                    raw_response=order or {}
                )

        except Exception as e:
            logger.error(f"Error executing spread: {e}")
            return OrderResult(
                success=False,
                status=OrderStatus.FAILED,
                message=f"Spread execution error: {str(e)}"
            )

    # =========================================================================
    # STRATEGY-SPECIFIC ORDER METHODS
    # =========================================================================

    def sell_covered_call(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        quantity: int,
        price: Optional[float] = None
    ) -> OrderResult:
        """Sell covered calls against existing stock position"""
        return self.sell_option(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type='call',
            quantity=quantity,
            price=price,
            is_closing=False
        )

    def sell_cash_secured_put(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        quantity: int,
        price: Optional[float] = None
    ) -> OrderResult:
        """Sell cash-secured puts"""
        return self.sell_option(
            symbol=symbol,
            strike=strike,
            expiration=expiration,
            option_type='put',
            quantity=quantity,
            price=price,
            is_closing=False
        )

    def open_iron_condor(
        self,
        symbol: str,
        expiration: str,
        put_long_strike: float,
        put_short_strike: float,
        call_short_strike: float,
        call_long_strike: float,
        quantity: int,
        net_credit: Optional[float] = None
    ) -> OrderResult:
        """
        Open an iron condor position.

        Args:
            symbol: Underlying symbol
            expiration: Expiration date
            put_long_strike: Long put strike (lowest)
            put_short_strike: Short put strike
            call_short_strike: Short call strike
            call_long_strike: Long call strike (highest)
            quantity: Number of contracts
            net_credit: Net credit to receive

        Returns:
            OrderResult
        """
        legs = [
            OptionLeg(symbol, put_long_strike, expiration, 'put', OrderSide.BUY_TO_OPEN, quantity),
            OptionLeg(symbol, put_short_strike, expiration, 'put', OrderSide.SELL_TO_OPEN, quantity),
            OptionLeg(symbol, call_short_strike, expiration, 'call', OrderSide.SELL_TO_OPEN, quantity),
            OptionLeg(symbol, call_long_strike, expiration, 'call', OrderSide.BUY_TO_OPEN, quantity),
        ]

        return self.execute_spread(legs, SpreadType.IRON_CONDOR, net_credit)

    def open_straddle(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        quantity: int,
        is_long: bool = True,
        net_price: Optional[float] = None
    ) -> OrderResult:
        """
        Open a straddle position.

        Args:
            symbol: Underlying symbol
            strike: Strike price (ATM typically)
            expiration: Expiration date
            quantity: Number of contracts
            is_long: True for long straddle, False for short
            net_price: Net debit (long) or credit (short)

        Returns:
            OrderResult
        """
        side = OrderSide.BUY_TO_OPEN if is_long else OrderSide.SELL_TO_OPEN

        legs = [
            OptionLeg(symbol, strike, expiration, 'call', side, quantity),
            OptionLeg(symbol, strike, expiration, 'put', side, quantity),
        ]

        return self.execute_spread(legs, SpreadType.STRADDLE, net_price)

    def open_strangle(
        self,
        symbol: str,
        put_strike: float,
        call_strike: float,
        expiration: str,
        quantity: int,
        is_long: bool = True,
        net_price: Optional[float] = None
    ) -> OrderResult:
        """
        Open a strangle position.

        Args:
            symbol: Underlying symbol
            put_strike: Put strike price (below current price)
            call_strike: Call strike price (above current price)
            expiration: Expiration date
            quantity: Number of contracts
            is_long: True for long strangle, False for short
            net_price: Net debit (long) or credit (short)

        Returns:
            OrderResult
        """
        side = OrderSide.BUY_TO_OPEN if is_long else OrderSide.SELL_TO_OPEN

        legs = [
            OptionLeg(symbol, call_strike, expiration, 'call', side, quantity),
            OptionLeg(symbol, put_strike, expiration, 'put', side, quantity),
        ]

        return self.execute_spread(legs, SpreadType.STRANGLE, net_price)

    def open_vertical_spread(
        self,
        symbol: str,
        long_strike: float,
        short_strike: float,
        expiration: str,
        option_type: str,  # 'call' or 'put'
        quantity: int,
        net_price: Optional[float] = None
    ) -> OrderResult:
        """
        Open a vertical spread (bull/bear call/put spread).

        Args:
            symbol: Underlying symbol
            long_strike: Strike to buy
            short_strike: Strike to sell
            expiration: Expiration date
            option_type: 'call' or 'put'
            quantity: Number of contracts
            net_price: Net debit or credit

        Returns:
            OrderResult
        """
        legs = [
            OptionLeg(symbol, long_strike, expiration, option_type, OrderSide.BUY_TO_OPEN, quantity),
            OptionLeg(symbol, short_strike, expiration, option_type, OrderSide.SELL_TO_OPEN, quantity),
        ]

        return self.execute_spread(legs, SpreadType.VERTICAL, net_price)

    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================

    @rate_limit("robinhood", tokens=2)
    def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an open order"""
        if self.paper_trading:
            logger.info(f"[PAPER TRADE] Cancelling order {order_id}")
            return OrderResult(
                success=True,
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                message="Paper trade order cancelled"
            )

        self._ensure_logged_in()

        try:
            result = rh.orders.cancel_option_order(order_id)

            if result:
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.CANCELLED,
                    message="Order cancelled successfully"
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    status=OrderStatus.FAILED,
                    message="Failed to cancel order"
                )

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                status=OrderStatus.FAILED,
                message=f"Cancel error: {str(e)}"
            )

    @rate_limit("robinhood", tokens=1)
    def get_order_status(self, order_id: str) -> OrderResult:
        """Get the current status of an order"""
        if self.paper_trading:
            # Find in history
            for order in self._order_history:
                if order.order_id == order_id:
                    return order

            return OrderResult(
                success=False,
                order_id=order_id,
                message="Order not found"
            )

        self._ensure_logged_in()

        try:
            order = rh.orders.get_option_order_info(order_id)

            if order:
                status_map = {
                    'queued': OrderStatus.QUEUED,
                    'confirmed': OrderStatus.CONFIRMED,
                    'partially_filled': OrderStatus.PARTIALLY_FILLED,
                    'filled': OrderStatus.FILLED,
                    'cancelled': OrderStatus.CANCELLED,
                    'rejected': OrderStatus.REJECTED,
                    'failed': OrderStatus.FAILED,
                }

                rh_status = order.get('state', 'unknown')

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=status_map.get(rh_status, OrderStatus.PENDING),
                    filled_quantity=int(float(order.get('processed_quantity', 0))),
                    average_price=float(order.get('processed_premium', 0)) / 100,
                    raw_response=order
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message="Order not found"
                )

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Status error: {str(e)}"
            )

    def roll_option(
        self,
        close_leg: OptionLeg,
        open_leg: OptionLeg,
        net_price: Optional[float] = None
    ) -> OrderResult:
        """
        Roll an option position (close existing, open new).

        Args:
            close_leg: Leg to close (existing position)
            open_leg: Leg to open (new position)
            net_price: Net credit/debit for the roll

        Returns:
            OrderResult
        """
        # Adjust sides for closing/opening
        close_leg.side = OrderSide.BUY_TO_CLOSE if close_leg.side in [OrderSide.SELL_TO_OPEN, OrderSide.SELL_TO_CLOSE] else OrderSide.SELL_TO_CLOSE
        open_leg.side = OrderSide.SELL_TO_OPEN if open_leg.side in [OrderSide.SELL_TO_OPEN, OrderSide.SELL_TO_CLOSE] else OrderSide.BUY_TO_OPEN

        legs = [close_leg, open_leg]

        return self.execute_spread(legs, SpreadType.CUSTOM, net_price)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_order_history(self, limit: int = 50) -> List[OrderResult]:
        """Get recent order history"""
        with self._lock:
            return self._order_history[-limit:]

    def get_buying_power(self) -> float:
        """Get current buying power"""
        account = self.client.get_account_info()
        return account.get('buying_power', 0.0)

    def estimate_order_cost(self, legs: List[OptionLeg]) -> Dict[str, float]:
        """
        Estimate the cost/credit for an order.

        Returns:
            Dict with estimated values
        """
        total_debit = 0.0
        total_credit = 0.0

        for leg in legs:
            value = (leg.price or 0) * leg.quantity * 100

            if leg.is_buy:
                total_debit += value
            else:
                total_credit += value

        return {
            'total_debit': total_debit,
            'total_credit': total_credit,
            'net_cost': total_debit - total_credit,
            'max_risk': max(total_debit, 0),  # Simplified
            'buying_power_required': total_debit  # Simplified
        }


# =============================================================================
# Singleton Access
# =============================================================================

_order_service: Optional[RobinhoodOrderService] = None
_service_lock = threading.Lock()


def get_order_service(paper_trading: bool = True) -> RobinhoodOrderService:
    """Get singleton order service instance"""
    global _order_service

    if _order_service is None:
        with _service_lock:
            if _order_service is None:
                _order_service = RobinhoodOrderService(paper_trading=paper_trading)

    return _order_service


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "=" * 60)
    print("Testing Robinhood Order Service (Paper Trading)")
    print("=" * 60)

    service = get_order_service(paper_trading=True)

    # Test 1: Single leg order
    print("\nTest 1: Buy Call Option")
    result = service.buy_option(
        symbol='AAPL',
        strike=180.0,
        expiration='2025-01-17',
        option_type='call',
        quantity=1,
        price=5.50
    )
    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Status: {result.status.value}")

    # Test 2: Sell CSP
    print("\nTest 2: Sell Cash-Secured Put")
    result = service.sell_cash_secured_put(
        symbol='NVDA',
        strike=480.0,
        expiration='2025-01-17',
        quantity=1,
        price=8.00
    )
    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")

    # Test 3: Iron Condor
    print("\nTest 3: Open Iron Condor")
    result = service.open_iron_condor(
        symbol='SPY',
        expiration='2025-01-17',
        put_long_strike=450.0,
        put_short_strike=460.0,
        call_short_strike=490.0,
        call_long_strike=500.0,
        quantity=1,
        net_credit=2.50
    )
    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Legs: {len(result.legs)}")

    # Test 4: Straddle
    print("\nTest 4: Open Long Straddle")
    result = service.open_straddle(
        symbol='TSLA',
        strike=250.0,
        expiration='2025-01-17',
        quantity=1,
        is_long=True,
        net_price=-15.00
    )
    print(f"  Success: {result.success}")
    print(f"  Order ID: {result.order_id}")

    # Test 5: Order history
    print("\nTest 5: Order History")
    history = service.get_order_history()
    print(f"  Total orders: {len(history)}")

    print("\n" + "=" * 60)
    print("Order service tests complete!")
    print("=" * 60)
