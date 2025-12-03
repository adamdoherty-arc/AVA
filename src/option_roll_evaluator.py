"""
Option Roll Evaluator - Comprehensive analysis of roll strategies

This module evaluates all possible roll strategies for losing positions:
- Roll Down: Same expiration, lower strike
- Roll Out: Same strike, later expiration
- Roll Down & Out: Lower strike + later expiration
- Do Nothing: Let assignment happen for wheel strategy
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import norm
import logging
import os

# Try to import Robinhood - optional dependency
try:
    import robin_stocks.robinhood as rh
    ROBINHOOD_AVAILABLE = True
except ImportError:
    ROBINHOOD_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptionRollEvaluator:
    """Evaluates roll strategies for option positions"""

    def __init__(self) -> None:
        self.risk_free_rate = 0.045
        self.trading_days = 252
        self.commission_per_contract = 0.65  # Typical broker commission
        self.rh_logged_in = False

    def _login_robinhood(self) -> bool:
        """Login to Robinhood if not already logged in"""
        if not ROBINHOOD_AVAILABLE:
            return False

        if self.rh_logged_in:
            return True

        try:
            username = os.getenv('ROBINHOOD_USERNAME')
            password = os.getenv('ROBINHOOD_PASSWORD')

            if not username or not password:
                logger.debug("Robinhood credentials not found")
                return False

            rh.authentication.login(
                username=username,
                password=password,
                expiresIn=86400,
                store_session=True
            )
            self.rh_logged_in = True
            logger.info("Logged into Robinhood successfully")
            return True

        except Exception as e:
            logger.error(f"Robinhood login failed: {e}")
            return False

    def _get_robinhood_options_chain(self, symbol: str, expiration: str, option_type: str = 'put') -> pd.DataFrame:
        """
        Get options chain from Robinhood as fallback when Yahoo Finance fails

        Args:
            symbol: Stock ticker
            expiration: Expiration date in YYYY-MM-DD format
            option_type: 'put' or 'call'

        Returns:
            DataFrame with options chain data compatible with yfinance format
        """
        if not self._login_robinhood():
            return pd.DataFrame()

        try:
            # Get options for this expiration from Robinhood
            options = rh.options.find_options_by_expiration(
                symbol,
                expiration,
                optionType=option_type
            )

            if not options:
                return pd.DataFrame()

            # Convert to DataFrame with yfinance-compatible column names
            data = []
            for opt in options:
                try:
                    # Get market data for this option
                    instrument_url = opt.get('url', '')
                    market_data = rh.options.get_option_market_data_by_id(opt.get('id', ''))

                    if not market_data:
                        continue

                    data.append({
                        'strike': float(opt.get('strike_price', 0)),
                        'bid': float(market_data[0].get('bid_price', 0)) if market_data else 0,
                        'ask': float(market_data[0].get('ask_price', 0)) if market_data else 0,
                        'lastPrice': float(market_data[0].get('last_trade_price', 0)) if market_data else 0,
                        'volume': int(market_data[0].get('volume', 0)) if market_data else 0,
                        'openInterest': int(market_data[0].get('open_interest', 0)) if market_data else 0,
                        'impliedVolatility': float(market_data[0].get('implied_volatility', 0)) if market_data else 0
                    })
                except Exception as e:
                    logger.debug(f"Error processing option: {e}")
                    continue

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            # Sort by strike
            df = df.sort_values('strike')
            return df

        except Exception as e:
            logger.error(f"Error getting Robinhood options chain for {symbol}: {e}")
            return pd.DataFrame()

    def _get_robinhood_expiration_dates(self, symbol: str) -> List[str]:
        """Get available expiration dates from Robinhood"""
        if not self._login_robinhood():
            return []

        try:
            chains = rh.options.get_chains(symbol)
            if chains and 'expiration_dates' in chains:
                return chains['expiration_dates']
            return []
        except Exception as e:
            logger.error(f"Error getting Robinhood expirations for {symbol}: {e}")
            return []

    def evaluate_roll_down(self, position: Dict) -> Dict:
        """
        Evaluate rolling down to a lower strike at same expiration

        Args:
            position: Current option position details

        Returns:
            Dictionary with roll down analysis
        """
        symbol = position['symbol']
        current_strike = position['current_strike']
        current_price = position['current_price']
        expiration = position['expiration']

        evaluation = {
            'strategy': 'Roll Down',
            'description': 'Close current position and open new put at lower strike, same expiration',
            'current_strike': current_strike,
            'expiration': expiration
        }

        try:
            # Try Yahoo Finance first
            try:
                ticker = yf.Ticker(symbol)
                option_chain = ticker.option_chain(expiration)
                puts = option_chain.puts
            except Exception as yf_error:
                # Fallback to Robinhood if Yahoo fails
                logger.debug(f"Yahoo Finance failed for {symbol}, trying Robinhood: {yf_error}")
                puts = self._get_robinhood_options_chain(symbol, expiration, 'put')

                if puts.empty:
                    evaluation['feasible'] = False
                    evaluation['reason'] = f'No options data available from Yahoo or Robinhood'
                    return evaluation

            # Find strikes below current but above spot price
            target_strikes = puts[(puts['strike'] < current_strike) &
                                 (puts['strike'] > current_price * 0.95)]

            if target_strikes.empty:
                evaluation['feasible'] = False
                evaluation['reason'] = 'No suitable lower strikes available'
                return evaluation

            # Select best strike (highest premium while maintaining safety)
            target_strikes['safety_score'] = (current_price - target_strikes['strike']) / current_price
            target_strikes['premium_score'] = target_strikes['bid'] / target_strikes['strike']
            target_strikes['total_score'] = target_strikes['safety_score'] * 0.6 + target_strikes['premium_score'] * 0.4

            best_option = target_strikes.nlargest(1, 'total_score').iloc[0]

            # Calculate roll cost/credit
            close_cost = self._get_option_price(symbol, current_strike, expiration, 'ask', 'put')
            open_credit = best_option['bid']
            net_credit = open_credit - close_cost - (2 * self.commission_per_contract / 100)

            evaluation.update({
                'feasible': True,
                'new_strike': float(best_option['strike']),
                'close_cost': close_cost,
                'open_credit': open_credit,
                'net_credit': net_credit,
                'new_breakeven': float(best_option['strike']) - open_credit,
                'days_added': 0,  # Same expiration
                'capital_at_risk': float(best_option['strike']) * 100,
                'max_profit': open_credit * 100,
                'probability_profit': self._calculate_prob_otm(
                    current_price, float(best_option['strike']),
                    self._get_volatility(symbol),
                    self._days_to_expiry(expiration)
                )
            })

            # Generate pros and cons
            evaluation['pros'] = self._generate_roll_down_pros(evaluation)
            evaluation['cons'] = self._generate_roll_down_cons(evaluation)

        except Exception as e:
            logger.error(f"Error evaluating roll down for {symbol}: {e}")
            evaluation['feasible'] = False
            evaluation['reason'] = str(e)

        return evaluation

    def evaluate_roll_out(self, position: Dict) -> Dict:
        """
        Evaluate rolling out to a later expiration at same strike

        Args:
            position: Current option position details

        Returns:
            Dictionary with roll out analysis
        """
        symbol = position['symbol']
        current_strike = position['current_strike']
        current_price = position['current_price']
        current_expiration = position['expiration']

        evaluation = {
            'strategy': 'Roll Out',
            'description': 'Close current position and open new put at same strike, later expiration',
            'current_strike': current_strike,
            'current_expiration': current_expiration
        }

        try:
            # Try Yahoo Finance first
            try:
                ticker = yf.Ticker(symbol)
                exp_dates = ticker.options
            except Exception as yf_error:
                # Fallback to Robinhood if Yahoo fails
                logger.debug(f"Yahoo Finance failed for {symbol}, trying Robinhood: {yf_error}")
                exp_dates = self._get_robinhood_expiration_dates(symbol)

                if not exp_dates:
                    evaluation['feasible'] = False
                    evaluation['reason'] = 'No options data available from Yahoo or Robinhood'
                    return evaluation

            # Find next available expiration dates
            future_expirations = [exp for exp in exp_dates
                                 if datetime.strptime(exp, '%Y-%m-%d') >
                                 datetime.strptime(current_expiration, '%Y-%m-%d')]

            if not future_expirations:
                evaluation['feasible'] = False
                evaluation['reason'] = 'No future expiration dates available'
                return evaluation

            # Evaluate next 2-3 expirations
            best_roll = None
            best_score = -float('inf')

            for exp_date in future_expirations[:3]:
                # Try Yahoo Finance first, fallback to Robinhood
                try:
                    if 'ticker' in locals():
                        option_chain = ticker.option_chain(exp_date)
                        puts = option_chain.puts
                    else:
                        raise Exception("Using Robinhood")
                except:
                    puts = self._get_robinhood_options_chain(symbol, exp_date, 'put')

                if puts.empty:
                    continue

                # Find same strike in new expiration
                same_strike = puts[puts['strike'] == current_strike]

                if same_strike.empty:
                    continue

                target_option = same_strike.iloc[0]

                # Calculate roll cost/credit
                close_cost = self._get_option_price(symbol, current_strike, current_expiration, 'ask', 'put')
                open_credit = float(target_option['bid'])
                net_credit = open_credit - close_cost - (2 * self.commission_per_contract / 100)

                # Calculate time decay benefit
                days_added = self._days_to_expiry(exp_date) - self._days_to_expiry(current_expiration)
                daily_theta = net_credit / days_added if days_added > 0 else 0

                # Score this roll option
                prob_profit = self._calculate_prob_otm(
                    current_price, current_strike,
                    self._get_volatility(symbol),
                    self._days_to_expiry(exp_date)
                )

                score = (prob_profit * 0.5) + (daily_theta * 0.3) + (net_credit * 0.2)

                if score > best_score:
                    best_score = score
                    best_roll = {
                        'new_expiration': exp_date,
                        'close_cost': close_cost,
                        'open_credit': open_credit,
                        'net_credit': net_credit,
                        'days_added': days_added,
                        'daily_theta': daily_theta,
                        'probability_profit': prob_profit
                    }

            if best_roll:
                evaluation.update({
                    'feasible': True,
                    'new_strike': current_strike,
                    'new_expiration': best_roll['new_expiration'],
                    'close_cost': best_roll['close_cost'],
                    'open_credit': best_roll['open_credit'],
                    'net_credit': best_roll['net_credit'],
                    'new_breakeven': current_strike - best_roll['open_credit'],
                    'days_added': best_roll['days_added'],
                    'capital_at_risk': current_strike * 100,
                    'max_profit': best_roll['open_credit'] * 100,
                    'probability_profit': best_roll['probability_profit'],
                    'daily_theta': best_roll['daily_theta']
                })

                evaluation['pros'] = self._generate_roll_out_pros(evaluation)
                evaluation['cons'] = self._generate_roll_out_cons(evaluation)
            else:
                evaluation['feasible'] = False
                evaluation['reason'] = 'No profitable roll out opportunities found'

        except Exception as e:
            logger.error(f"Error evaluating roll out for {symbol}: {e}")
            evaluation['feasible'] = False
            evaluation['reason'] = str(e)

        return evaluation

    def evaluate_roll_down_and_out(self, position: Dict) -> Dict:
        """
        Evaluate rolling to lower strike and later expiration

        Args:
            position: Current option position details

        Returns:
            Dictionary with roll down and out analysis
        """
        symbol = position['symbol']
        current_strike = position['current_strike']
        current_price = position['current_price']
        current_expiration = position['expiration']

        evaluation = {
            'strategy': 'Roll Down & Out',
            'description': 'Close current position and open new put at lower strike, later expiration',
            'current_strike': current_strike,
            'current_expiration': current_expiration
        }

        try:
            # Try Yahoo Finance first
            try:
                ticker = yf.Ticker(symbol)
                exp_dates = ticker.options
            except Exception as yf_error:
                # Fallback to Robinhood if Yahoo fails
                logger.debug(f"Yahoo Finance failed for {symbol}, trying Robinhood: {yf_error}")
                exp_dates = self._get_robinhood_expiration_dates(symbol)

                if not exp_dates:
                    evaluation['feasible'] = False
                    evaluation['reason'] = 'No options data available from Yahoo or Robinhood'
                    return evaluation

            # Find future expiration dates
            future_expirations = [exp for exp in exp_dates
                                 if datetime.strptime(exp, '%Y-%m-%d') >
                                 datetime.strptime(current_expiration, '%Y-%m-%d')]

            if not future_expirations:
                evaluation['feasible'] = False
                evaluation['reason'] = 'No future expiration dates available'
                return evaluation

            # Evaluate combinations of strikes and expirations
            best_combo = None
            best_score = -float('inf')

            for exp_date in future_expirations[:3]:
                # Try Yahoo Finance first, fallback to Robinhood
                try:
                    if 'ticker' in locals():
                        option_chain = ticker.option_chain(exp_date)
                        puts = option_chain.puts
                    else:
                        raise Exception("Using Robinhood")
                except:
                    puts = self._get_robinhood_options_chain(symbol, exp_date, 'put')

                if puts.empty:
                    continue

                # Find strikes below current but reasonable
                target_strikes = puts[(puts['strike'] < current_strike) &
                                     (puts['strike'] > current_price * 0.90)]

                for _, target_option in target_strikes.iterrows():
                    # Calculate roll metrics
                    close_cost = self._get_option_price(symbol, current_strike, current_expiration, 'ask', 'put')
                    open_credit = float(target_option['bid'])
                    net_credit = open_credit - close_cost - (2 * self.commission_per_contract / 100)

                    if net_credit <= 0:
                        continue  # Skip if requires net debit

                    # Calculate value metrics
                    days_added = self._days_to_expiry(exp_date) - self._days_to_expiry(current_expiration)
                    new_strike = float(target_option['strike'])

                    prob_profit = self._calculate_prob_otm(
                        current_price, new_strike,
                        self._get_volatility(symbol),
                        self._days_to_expiry(exp_date)
                    )

                    # Composite score
                    strike_reduction = (current_strike - new_strike) / current_strike
                    time_value = days_added / 30  # Normalize to monthly
                    score = (net_credit * 0.4) + (prob_profit * 0.3) + \
                           (strike_reduction * 0.2) + (time_value * 0.1)

                    if score > best_score:
                        best_score = score
                        best_combo = {
                            'new_strike': new_strike,
                            'new_expiration': exp_date,
                            'close_cost': close_cost,
                            'open_credit': open_credit,
                            'net_credit': net_credit,
                            'days_added': days_added,
                            'probability_profit': prob_profit,
                            'strike_reduction_pct': strike_reduction * 100
                        }

            if best_combo:
                evaluation.update({
                    'feasible': True,
                    'new_strike': best_combo['new_strike'],
                    'new_expiration': best_combo['new_expiration'],
                    'close_cost': best_combo['close_cost'],
                    'open_credit': best_combo['open_credit'],
                    'net_credit': best_combo['net_credit'],
                    'new_breakeven': best_combo['new_strike'] - best_combo['open_credit'],
                    'days_added': best_combo['days_added'],
                    'capital_at_risk': best_combo['new_strike'] * 100,
                    'max_profit': best_combo['open_credit'] * 100,
                    'probability_profit': best_combo['probability_profit'],
                    'strike_reduction_pct': best_combo['strike_reduction_pct']
                })

                evaluation['pros'] = self._generate_roll_down_out_pros(evaluation)
                evaluation['cons'] = self._generate_roll_down_out_cons(evaluation)
            else:
                evaluation['feasible'] = False
                evaluation['reason'] = 'No profitable roll down and out opportunities found'

        except Exception as e:
            logger.error(f"Error evaluating roll down and out for {symbol}: {e}")
            evaluation['feasible'] = False
            evaluation['reason'] = str(e)

        return evaluation

    def evaluate_assignment(self, position: Dict) -> Dict:
        """
        Evaluate letting the option get assigned (do nothing strategy)

        Args:
            position: Current option position details

        Returns:
            Dictionary with assignment analysis
        """
        symbol = position['symbol']
        current_strike = position['current_strike']
        current_price = position['current_price']
        premium_collected = position.get('premium_collected', 0)

        evaluation = {
            'strategy': 'Accept Assignment',
            'description': 'Let put option get assigned and take delivery of shares',
            'current_strike': current_strike,
            'feasible': True
        }

        # Calculate assignment metrics
        shares_to_receive = 100  # Per contract per quantity
        quantity = abs(position.get('quantity', 1))

        # Premium collected is total for position, convert to per-share
        # If premium_collected is already in dollars (not cents), use it directly
        if premium_collected > 100:
            # Premium in dollars total - convert to per share
            premium_per_share = premium_collected / (100 * quantity)
        else:
            # Premium likely already per share or per contract
            premium_per_share = premium_collected / 100 if premium_collected > 1 else premium_collected

        # True cost basis = strike - premium received per share
        cost_basis = current_strike - premium_per_share

        # Calculate actual unrealized P&L (can be negative for profit, positive for loss)
        # True P&L = (cost_basis - current_price) per share
        unrealized_pnl_per_share = cost_basis - current_price
        total_unrealized_pnl = unrealized_pnl_per_share * shares_to_receive * quantity

        # For display purposes, separate loss and profit
        immediate_loss = max(0, total_unrealized_pnl)  # Only positive values (losses)
        immediate_profit = abs(min(0, total_unrealized_pnl))  # Only negative values (profits), converted to positive

        # Calculate percentage
        pnl_percentage = (unrealized_pnl_per_share / cost_basis * 100) if cost_basis > 0 else 0

        evaluation.update({
            'shares_to_receive': shares_to_receive * quantity,
            'cost_basis_per_share': cost_basis,
            'premium_per_share': premium_per_share,
            'total_cost': cost_basis * shares_to_receive * quantity,
            'current_market_value': current_price * shares_to_receive * quantity,
            'immediate_loss': immediate_loss,
            'immediate_profit': immediate_profit,
            'unrealized_pnl': total_unrealized_pnl,  # Full P&L (negative = profit)
            'loss_percentage': pnl_percentage,
            'capital_required': current_strike * shares_to_receive * quantity,
            'wheel_opportunity': True  # Can sell covered calls after assignment
        })

        # Calculate potential covered call income
        cc_analysis = self._analyze_covered_call_potential(symbol, cost_basis)
        evaluation['covered_call_potential'] = cc_analysis

        # Generate pros and cons
        evaluation['pros'] = self._generate_assignment_pros(evaluation)
        evaluation['cons'] = self._generate_assignment_cons(evaluation)

        return evaluation

    def compare_strategies(self, position: Dict) -> Dict:
        """
        Compare all four strategies and provide recommendation

        Args:
            position: Current option position details

        Returns:
            Dictionary with all strategies compared and AI recommendation
        """
        # Evaluate all strategies
        strategies = {
            'roll_down': self.evaluate_roll_down(position),
            'roll_out': self.evaluate_roll_out(position),
            'roll_down_out': self.evaluate_roll_down_and_out(position),
            'assignment': self.evaluate_assignment(position)
        }

        # Score each strategy
        scored_strategies = []
        for name, strategy in strategies.items():
            if strategy.get('feasible', False):
                score = self._calculate_strategy_score(strategy, position)
                strategy['score'] = score
                strategy['strategy_key'] = name
                scored_strategies.append(strategy)

        # Sort by score
        scored_strategies.sort(key=lambda x: x['score'], reverse=True)

        # Generate recommendation
        if scored_strategies:
            best_strategy = scored_strategies[0]
            recommendation = self._generate_recommendation(best_strategy, position)
        else:
            recommendation = {
                'recommended_strategy': 'Hold',
                'reasoning': 'No viable roll strategies available. Consider holding position or closing at loss.',
                'confidence': 'Low'
            }

        return {
            'position': position,
            'strategies': strategies,
            'ranked_strategies': scored_strategies,
            'recommendation': recommendation
        }

    def _get_option_price(self, symbol: str, strike: float, expiration: str,
                         price_type: str = 'mid', option_type: str = 'put') -> float:
        """Get option price from market data"""
        try:
            ticker = yf.Ticker(symbol)
            option_chain = ticker.option_chain(expiration)

            if option_type == 'put':
                options = option_chain.puts
            else:
                options = option_chain.calls

            target = options[options['strike'] == strike]

            if target.empty:
                return 0

            if price_type == 'bid':
                return float(target['bid'].iloc[0])
            elif price_type == 'ask':
                return float(target['ask'].iloc[0])
            else:  # mid
                bid = float(target['bid'].iloc[0])
                ask = float(target['ask'].iloc[0])
                return (bid + ask) / 2

        except Exception as e:
            logger.error(f"Error getting option price: {e}")
            return 0

    def _get_volatility(self, symbol: str) -> float:
        """Calculate implied volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            returns = hist['Close'].pct_change().dropna()
            return float(returns.std() * np.sqrt(252))
        except:
            return 0.3  # Default 30% volatility

    def _days_to_expiry(self, expiration: str) -> int:
        """Calculate days to expiration"""
        try:
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
            return max(0, (exp_date - datetime.now()).days)
        except:
            return 0

    def _calculate_prob_otm(self, spot: float, strike: float,
                           volatility: float, days: int) -> float:
        """Calculate probability option stays OTM"""
        # Guard against invalid inputs
        if days <= 0:
            return 0 if spot <= strike else 1

        if spot <= 0 or strike <= 0:
            return 0.5  # Neutral probability for invalid prices

        if volatility <= 0:
            # Deterministic case: no volatility
            return 0 if spot <= strike else 1

        try:
            time_to_exp = days / 365
            d2 = (np.log(spot / strike) +
                  (self.risk_free_rate - 0.5 * volatility**2) * time_to_exp) / \
                 (volatility * np.sqrt(time_to_exp))

            return float(norm.cdf(d2))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Return neutral probability if calculation fails
            return 0.5

    def _analyze_covered_call_potential(self, symbol: str, cost_basis: float) -> Dict:
        """Analyze potential covered call income after assignment"""
        try:
            ticker = yf.Ticker(symbol)
            next_expiry = ticker.options[0] if ticker.options else None

            if not next_expiry:
                return {'available': False}

            calls = ticker.option_chain(next_expiry).calls

            # Find OTM calls above cost basis
            otm_calls = calls[calls['strike'] >= cost_basis * 1.01]  # 1% OTM

            if otm_calls.empty:
                return {'available': False}

            best_call = otm_calls.iloc[0]

            monthly_income = float(best_call['bid']) * 100
            monthly_yield = (float(best_call['bid']) / cost_basis) * 100

            return {
                'available': True,
                'strike': float(best_call['strike']),
                'premium': float(best_call['bid']),
                'monthly_income': monthly_income,
                'monthly_yield': monthly_yield,
                'annualized_yield': monthly_yield * 12
            }
        except:
            return {'available': False}

    def _calculate_strategy_score(self, strategy: Dict, position: Dict) -> float:
        """Calculate composite score for a strategy"""
        score = 0

        # Probability of profit (30%)
        if 'probability_profit' in strategy:
            score += strategy['probability_profit'] * 0.3

        # Net credit received (25%)
        if 'net_credit' in strategy and strategy['net_credit'] > 0:
            normalized_credit = min(strategy['net_credit'] / 5, 1)  # Normalize to $5
            score += normalized_credit * 0.25

        # Capital efficiency (20%)
        if 'capital_at_risk' in strategy:
            capital_ratio = position['current_strike'] * 100 / strategy['capital_at_risk']
            score += min(capital_ratio, 1) * 0.2

        # Time value (15%)
        if 'days_added' in strategy:
            time_score = min(strategy['days_added'] / 30, 1)  # Normalize to 30 days
            score += time_score * 0.15

        # Special case for assignment
        if strategy.get('strategy') == 'Accept Assignment':
            if strategy.get('covered_call_potential', {}).get('available', False):
                score += 0.25  # Bonus for wheel opportunity

        return score

    def _generate_recommendation(self, strategy: Dict, position: Dict) -> Dict:
        """Generate AI recommendation for best strategy"""
        strategy_name = strategy.get('strategy', 'Unknown')
        score = strategy.get('score', 0)

        if score >= 0.7:
            confidence = 'High'
            action = 'RECOMMENDED'
        elif score >= 0.5:
            confidence = 'Medium'
            action = 'Consider'
        else:
            confidence = 'Low'
            action = 'Evaluate carefully'

        reasoning = f"{action}: {strategy_name} strategy "

        if strategy_name == 'Roll Down & Out':
            reasoning += f"offers best combination of premium collection (${strategy.get('net_credit', 0):.2f}) "
            reasoning += f"and risk reduction ({strategy.get('strike_reduction_pct', 0):.1f}% lower strike)"
        elif strategy_name == 'Roll Out':
            reasoning += f"provides {strategy.get('days_added', 0)} additional days "
            reasoning += f"for stock to recover above strike"
        elif strategy_name == 'Roll Down':
            reasoning += f"reduces risk by lowering strike to ${strategy.get('new_strike', 0):.2f} "
            reasoning += f"while collecting additional credit"
        elif strategy_name == 'Accept Assignment':
            reasoning += f"allows transition to wheel strategy with "
            reasoning += f"{strategy.get('covered_call_potential', {}).get('annualized_yield', 0):.1f}% "
            reasoning += f"potential annual yield from covered calls"

        return {
            'recommended_strategy': strategy_name,
            'reasoning': reasoning,
            'confidence': confidence,
            'expected_outcome': self._predict_outcome(strategy),
            'risk_assessment': self._assess_risk(strategy)
        }

    def _predict_outcome(self, strategy: Dict) -> str:
        """Predict likely outcome of strategy"""
        prob = strategy.get('probability_profit', 0)

        if prob >= 0.8:
            return "Very likely to expire worthless (keep full premium)"
        elif prob >= 0.6:
            return "Good chance of profitable outcome"
        elif prob >= 0.4:
            return "Moderate risk of assignment"
        else:
            return "High risk of assignment or loss"

    def _assess_risk(self, strategy: Dict) -> str:
        """Assess risk level of strategy"""
        if strategy.get('strategy') == 'Accept Assignment':
            return "Requires full capital for share ownership"

        net_credit = strategy.get('net_credit', 0)
        if net_credit < 0:
            return "Requires additional capital (net debit)"
        elif net_credit > 0:
            return "Reduces cost basis with net credit"
        else:
            return "Neutral cash flow impact"

    # Pros and cons generators
    def _generate_roll_down_pros(self, evaluation: Dict) -> List[str]:
        """Generate pros for roll down strategy"""
        pros = []
        if evaluation.get('net_credit', 0) > 0:
            pros.append(f"Collects additional ${evaluation['net_credit']:.2f} credit")
        pros.append(f"Reduces risk with lower strike of ${evaluation.get('new_strike', 0):.2f}")
        if evaluation.get('probability_profit', 0) > 0.6:
            pros.append(f"High probability of profit: {evaluation['probability_profit']*100:.1f}%")
        pros.append("Maintains same expiration timeline")
        return pros

    def _generate_roll_down_cons(self, evaluation: Dict) -> List[str]:
        """Generate cons for roll down strategy"""
        cons = []
        if evaluation.get('net_credit', 0) < 0:
            cons.append(f"Requires net debit of ${abs(evaluation['net_credit']):.2f}")
        cons.append("Locks in partial loss on original position")
        cons.append(f"Still at risk if stock drops below ${evaluation.get('new_strike', 0):.2f}")
        cons.append("May need multiple rolls if stock continues declining")
        return cons

    def _generate_roll_out_pros(self, evaluation: Dict) -> List[str]:
        """Generate pros for roll out strategy"""
        pros = []
        pros.append(f"Adds {evaluation.get('days_added', 0)} days for recovery")
        if evaluation.get('net_credit', 0) > 0:
            pros.append(f"Collects ${evaluation['net_credit']:.2f} additional premium")
        pros.append("Maintains original strike price target")
        pros.append(f"Daily theta decay of ${evaluation.get('daily_theta', 0):.2f}")
        return pros

    def _generate_roll_out_cons(self, evaluation: Dict) -> List[str]:
        """Generate cons for roll out strategy"""
        cons = []
        cons.append("Extends time capital is tied up")
        cons.append("No reduction in strike price risk")
        if evaluation.get('net_credit', 0) < 0:
            cons.append(f"Costs ${abs(evaluation['net_credit']):.2f} to roll")
        cons.append("Stock may continue declining during extended period")
        return cons

    def _generate_roll_down_out_pros(self, evaluation: Dict) -> List[str]:
        """Generate pros for roll down and out strategy"""
        pros = []
        pros.append(f"Best net credit: ${evaluation.get('net_credit', 0):.2f}")
        pros.append(f"Reduces strike by {evaluation.get('strike_reduction_pct', 0):.1f}%")
        pros.append(f"Adds {evaluation.get('days_added', 0)} days for recovery")
        pros.append("Maximum flexibility and premium collection")
        return pros

    def _generate_roll_down_out_cons(self, evaluation: Dict) -> List[str]:
        """Generate cons for roll down and out strategy"""
        cons = []
        cons.append("Most complex strategy to execute")
        cons.append("Extends time horizon significantly")
        cons.append("May still result in assignment at lower strike")
        cons.append("Locks in loss on original position")
        return cons

    def _generate_assignment_pros(self, evaluation: Dict) -> List[str]:
        """Generate pros for assignment strategy"""
        pros = []
        pros.append("Transitions to wheel strategy (sell covered calls)")
        if evaluation.get('covered_call_potential', {}).get('available'):
            pros.append(f"Potential {evaluation['covered_call_potential']['annualized_yield']:.1f}% annual yield from calls")
        pros.append("Owns underlying shares for potential appreciation")
        pros.append("No additional transaction costs for rolling")
        return pros

    def _generate_assignment_cons(self, evaluation: Dict) -> List[str]:
        """Generate cons for assignment strategy"""
        cons = []

        # Show profit or loss based on actual P&L
        immediate_loss = evaluation.get('immediate_loss', 0)
        immediate_profit = evaluation.get('immediate_profit', 0)

        if immediate_loss > 0:
            cons.append(f"Immediate paper loss of ${immediate_loss:.2f}")
        elif immediate_profit > 0:
            cons.append(f"Immediate paper profit of ${immediate_profit:.2f} (current price above cost basis)")
        else:
            cons.append("Position at breakeven (current price equals cost basis)")

        cons.append(f"Requires ${evaluation.get('capital_required', 0):.2f} capital")
        cons.append("Risk of further stock decline")
        cons.append("May take time to recover through covered calls")
        return cons


# Example usage
if __name__ == "__main__":
    evaluator = OptionRollEvaluator()

    # Example losing position
    position = {
        'symbol': 'AAPL',
        'current_strike': 180,
        'current_price': 175,
        'expiration': '2024-01-19',
        'premium_collected': 250,  # $2.50 per share
        'quantity': -1
    }

    # Compare all strategies
    comparison = evaluator.compare_strategies(position)

    print("\n=== ROLL STRATEGY EVALUATION ===")
    print(f"Symbol: {position['symbol']}")
    print(f"Current Strike: ${position['current_strike']}")
    print(f"Current Price: ${position['current_price']}\n")

    for strategy in comparison['ranked_strategies']:
        print(f"\n{strategy['strategy']} (Score: {strategy['score']:.2f})")
        if 'pros' in strategy:
            print("Pros:", ', '.join(strategy['pros']))
        if 'cons' in strategy:
            print("Cons:", ', '.join(strategy['cons']))

    print(f"\n\nRECOMMENDATION: {comparison['recommendation']['recommended_strategy']}")
    print(f"Reasoning: {comparison['recommendation']['reasoning']}")
    print(f"Confidence: {comparison['recommendation']['confidence']}")