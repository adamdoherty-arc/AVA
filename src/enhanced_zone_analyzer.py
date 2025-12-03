"""
Enhanced Zone Analyzer - Multi-Indicator Confirmation System
Integrates Supply/Demand Zones with Smart Money, Volume Profile, and Momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

from src.zone_analyzer import ZoneAnalyzer
from src.smart_money_indicators import SmartMoneyIndicators
from src.volume_profile_analyzer import VolumeProfileAnalyzer
from src.momentum_indicators import MomentumIndicators

logger = logging.getLogger(__name__)


class EnhancedZoneAnalyzer:
    """
    Enhanced zone analysis with multi-indicator confirmation

    Scoring factors:
    - Base zone strength (0-100)
    - Order block alignment (+15)
    - Fair value gap (+10)
    - POC alignment (+15)
    - High volume node (+10)
    - RSI confirmation (+10)
    - MACD alignment (+10)
    - EMA alignment (+10)
    - Liquidity pool (+10)
    - CVD confirmation (+10)

    Maximum: 100 (capped)
    """

    def __init__(self) -> None:
        """Initialize all analyzers"""
        self.base_analyzer = ZoneAnalyzer()
        self.smc = SmartMoneyIndicators()
        self.vpa = VolumeProfileAnalyzer()
        self.mi = MomentumIndicators()

    def analyze_zone_complete(
        self,
        zone: Dict,
        df: pd.DataFrame,
        current_price: float
    ) -> Dict:
        """
        Complete multi-indicator zone analysis

        Args:
            zone: Zone dictionary from ZoneDetector
            df: Price dataframe
            current_price: Current stock price

        Returns:
            Enhanced zone with all confirmations
        """

        # Start with base analysis
        zone = self.base_analyzer.analyze_zone(zone, current_price)

        # Get all indicator data
        smc_indicators = self.smc.get_all_smc_indicators(df)
        volume_analysis = self.vpa.get_complete_analysis(df)
        momentum = self.mi.get_all_momentum_indicators(df, current_price)

        # Calculate confirmations
        confirmations = self._calculate_confirmations(
            zone=zone,
            smc=smc_indicators,
            volume=volume_analysis,
            momentum=momentum,
            current_price=current_price
        )

        # Enhance zone score
        enhanced_score = self._calculate_enhanced_score(zone, confirmations)

        # Build enhanced zone
        zone['enhanced_score'] = enhanced_score['score']
        zone['confirmations'] = enhanced_score['confirmations']
        zone['confirmation_count'] = len(enhanced_score['confirmations'])
        zone['confirmation_details'] = confirmations
        zone['setup_quality'] = self._determine_setup_quality(enhanced_score['score'], len(enhanced_score['confirmations']))

        # Add trading recommendations
        zone['trading_plan'] = self._generate_trading_plan(zone, momentum, current_price)

        logger.info(f"Zone analyzed: {zone['symbol']} - Score: {enhanced_score['score']}/100 ({len(enhanced_score['confirmations'])} confirmations)")

        return zone

    def _calculate_confirmations(
        self,
        zone: Dict,
        smc: Dict,
        volume: Dict,
        momentum: Dict,
        current_price: float
    ) -> Dict:
        """Calculate all confirmation factors"""

        confirmations = {}

        # 1. Order Block Alignment
        confirmations['order_block'] = self._check_order_block_alignment(
            zone, smc['order_blocks'], current_price
        )

        # 2. Fair Value Gap
        confirmations['fvg'] = self._check_fvg_alignment(
            zone, smc['fair_value_gaps'], current_price
        )

        # 3. POC Alignment
        confirmations['poc'] = self._check_poc_alignment(
            zone, volume['volume_profile'], current_price
        )

        # 4. High Volume Node
        confirmations['hvn'] = self._check_hvn_alignment(
            zone, volume['volume_nodes'], current_price
        )

        # 5. RSI Confirmation
        confirmations['rsi'] = self._check_rsi_confirmation(
            zone, momentum['rsi']
        )

        # 6. MACD Alignment
        confirmations['macd'] = self._check_macd_alignment(
            zone, momentum['macd']
        )

        # 7. EMA Alignment
        confirmations['ema'] = self._check_ema_alignment(
            zone, momentum['emas'], current_price
        )

        # 8. Liquidity Pool
        confirmations['liquidity'] = self._check_liquidity_alignment(
            zone, smc['liquidity_pools'], current_price
        )

        # 9. CVD Confirmation
        confirmations['cvd'] = self._check_cvd_confirmation(
            zone, momentum['cvd']
        )

        # 10. Market Structure
        confirmations['structure'] = self._check_structure_alignment(
            zone, smc['market_structure']
        )

        return confirmations

    def _check_order_block_alignment(self, zone: Dict, order_blocks: List[Dict], current_price: float) -> Dict:
        """Check if zone aligns with order block"""
        for ob in order_blocks:
            # Check if order block overlaps zone
            if (ob['bottom'] <= zone['zone_top'] and ob['top'] >= zone['zone_bottom']):
                # Check if types match
                if (zone['zone_type'] == 'DEMAND' and ob['type'] == 'BULLISH_OB') or \
                   (zone['zone_type'] == 'SUPPLY' and ob['type'] == 'BEARISH_OB'):
                    return {
                        'confirmed': True,
                        'strength': ob['strength'],
                        'type': ob['type'],
                        'bonus_points': 15
                    }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_fvg_alignment(self, zone: Dict, fvgs: List[Dict], current_price: float) -> Dict:
        """Check if zone contains fair value gap"""
        for fvg in fvgs:
            if not fvg['filled']:  # Only unfilled FVGs
                # Check if FVG overlaps zone
                if (fvg['bottom'] <= zone['zone_top'] and fvg['top'] >= zone['zone_bottom']):
                    # Check if types match
                    if (zone['zone_type'] == 'DEMAND' and fvg['type'] == 'BULLISH_FVG') or \
                       (zone['zone_type'] == 'SUPPLY' and fvg['type'] == 'BEARISH_FVG'):
                        return {
                            'confirmed': True,
                            'gap_pct': fvg['gap_pct'],
                            'type': fvg['type'],
                            'bonus_points': 10
                        }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_poc_alignment(self, zone: Dict, volume_profile: Dict, current_price: float) -> Dict:
        """Check if zone near POC"""
        if not volume_profile or 'poc' not in volume_profile:
            return {'confirmed': False, 'bonus_points': 0}

        poc = volume_profile['poc']

        # Check if POC within zone
        if zone['zone_bottom'] <= poc <= zone['zone_top']:
            return {
                'confirmed': True,
                'poc_price': poc,
                'within_zone': True,
                'bonus_points': 15
            }

        # Check if zone near POC (within 5%)
        distance_to_bottom = abs(poc - zone['zone_bottom']) / zone['zone_bottom'] * 100
        distance_to_top = abs(poc - zone['zone_top']) / zone['zone_top'] * 100

        if min(distance_to_bottom, distance_to_top) <= 5.0:
            return {
                'confirmed': True,
                'poc_price': poc,
                'within_zone': False,
                'distance': min(distance_to_bottom, distance_to_top),
                'bonus_points': 10
            }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_hvn_alignment(self, zone: Dict, volume_nodes: Dict, current_price: float) -> Dict:
        """Check if zone at high volume node"""
        if not volume_nodes or 'hvn' not in volume_nodes:
            return {'confirmed': False, 'bonus_points': 0}

        for hvn in volume_nodes['hvn']:
            # Check if HVN within zone
            if zone['zone_bottom'] <= hvn['price'] <= zone['zone_top']:
                return {
                    'confirmed': True,
                    'hvn_price': hvn['price'],
                    'hvn_strength': hvn['strength'],
                    'bonus_points': 10
                }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_rsi_confirmation(self, zone: Dict, rsi: Dict) -> Dict:
        """Check RSI confirmation"""
        rsi_signal = rsi['signal']

        if zone['zone_type'] == 'DEMAND':
            # Want oversold RSI for demand zones
            if rsi_signal['signal'] == 'OVERSOLD':
                return {
                    'confirmed': True,
                    'rsi_value': rsi_signal['value'],
                    'signal': rsi_signal['strength'],
                    'bonus_points': 10 if rsi_signal['strength'] == 'STRONG_BUY' else 5
                }
        else:  # SUPPLY
            # Want overbought RSI for supply zones
            if rsi_signal['signal'] == 'OVERBOUGHT':
                return {
                    'confirmed': True,
                    'rsi_value': rsi_signal['value'],
                    'signal': rsi_signal['strength'],
                    'bonus_points': 10 if rsi_signal['strength'] == 'STRONG_SELL' else 5
                }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_macd_alignment(self, zone: Dict, macd: Dict) -> Dict:
        """Check MACD alignment"""
        macd_signal = macd['signal']

        if zone['zone_type'] == 'DEMAND':
            # Want bullish MACD for demand zones
            if 'BULLISH' in macd_signal['signal']:
                return {
                    'confirmed': True,
                    'signal': macd_signal['signal'],
                    'histogram': macd_signal['histogram'],
                    'bonus_points': 10 if 'CROSS' in macd_signal['signal'] else 5
                }
        else:  # SUPPLY
            # Want bearish MACD for supply zones
            if 'BEARISH' in macd_signal['signal']:
                return {
                    'confirmed': True,
                    'signal': macd_signal['signal'],
                    'histogram': macd_signal['histogram'],
                    'bonus_points': 10 if 'CROSS' in macd_signal['signal'] else 5
                }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_ema_alignment(self, zone: Dict, emas: Dict, current_price: float) -> Dict:
        """Check EMA alignment"""
        alignment = emas['alignment']

        if zone['zone_type'] == 'DEMAND':
            # Want bullish EMA alignment for demand
            if alignment['alignment'] == 'BULLISH':
                return {
                    'confirmed': True,
                    'alignment': alignment['alignment'],
                    'strength': alignment['strength'],
                    'above_ema_200': alignment['above_ema_200'],
                    'bonus_points': 10 if alignment['all_aligned'] else 5
                }
        else:  # SUPPLY
            # Want bearish EMA alignment for supply
            if alignment['alignment'] == 'BEARISH':
                return {
                    'confirmed': True,
                    'alignment': alignment['alignment'],
                    'strength': alignment['strength'],
                    'above_ema_200': alignment['above_ema_200'],
                    'bonus_points': 10 if alignment['all_aligned'] else 5
                }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_liquidity_alignment(self, zone: Dict, liquidity_pools: List[Dict], current_price: float) -> Dict:
        """Check liquidity pool alignment"""
        for pool in liquidity_pools:
            if not pool['swept']:  # Only unswept liquidity
                # Check if liquidity near zone
                distance_pct = abs(pool['price'] - zone['zone_midpoint']) / zone['zone_midpoint'] * 100

                if distance_pct <= 3.0:  # Within 3%
                    # Demand zone below sell-side liquidity = good
                    if zone['zone_type'] == 'DEMAND' and pool['type'] == 'SELL_SIDE_LIQUIDITY':
                        if pool['price'] > zone['zone_top']:
                            return {
                                'confirmed': True,
                                'pool_type': pool['type'],
                                'pool_strength': pool['strength'],
                                'bonus_points': 10
                            }
                    # Supply zone above buy-side liquidity = good
                    elif zone['zone_type'] == 'SUPPLY' and pool['type'] == 'BUY_SIDE_LIQUIDITY':
                        if pool['price'] < zone['zone_bottom']:
                            return {
                                'confirmed': True,
                                'pool_type': pool['type'],
                                'pool_strength': pool['strength'],
                                'bonus_points': 10
                            }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_cvd_confirmation(self, zone: Dict, cvd_data: pd.DataFrame) -> Dict:
        """Check CVD (Cumulative Volume Delta) confirmation"""
        cvd_signal = self.mi.get_cvd_signal(cvd_data, zone['zone_type'])

        if cvd_signal['strength'] == 'CONFIRMING':
            return {
                'confirmed': True,
                'signal': cvd_signal['signal'],
                'trend': cvd_signal['trend'],
                'cvd': cvd_signal['cvd'],
                'bonus_points': 10
            }

        return {'confirmed': False, 'bonus_points': 0}

    def _check_structure_alignment(self, zone: Dict, structure: Dict) -> Dict:
        """Check market structure alignment"""
        current_trend = structure['current_trend']

        # CHoCH (Change of Character) is strong reversal signal
        recent_choch = structure['choch'][-3:] if structure['choch'] else []

        for choch in recent_choch:
            if zone['zone_type'] == 'DEMAND' and choch['direction'] == 'BULLISH':
                return {
                    'confirmed': True,
                    'signal': 'CHOCH_BULLISH',
                    'bonus_points': 10
                }
            elif zone['zone_type'] == 'SUPPLY' and choch['direction'] == 'BEARISH':
                return {
                    'confirmed': True,
                    'signal': 'CHOCH_BEARISH',
                    'bonus_points': 10
                }

        # BOS (Break of Structure) confirms trend
        if zone['zone_type'] == 'DEMAND' and current_trend == 'BULLISH':
            return {
                'confirmed': True,
                'signal': 'TREND_ALIGNED',
                'bonus_points': 5
            }
        elif zone['zone_type'] == 'SUPPLY' and current_trend == 'BEARISH':
            return {
                'confirmed': True,
                'signal': 'TREND_ALIGNED',
                'bonus_points': 5
            }

        return {'confirmed': False, 'bonus_points': 0}

    def _calculate_enhanced_score(self, zone: Dict, confirmations: Dict) -> Dict:
        """Calculate final enhanced score"""
        base_score = zone.get('strength_score', 50)
        bonus_points = 0
        confirmed_list = []

        for indicator, conf in confirmations.items():
            if conf.get('confirmed', False):
                bonus_points += conf.get('bonus_points', 0)
                confirmed_list.append(indicator.upper().replace('_', ' '))

        # Final score (capped at 100)
        final_score = min(100, base_score + bonus_points)

        return {
            'score': final_score,
            'base_score': base_score,
            'bonus_points': bonus_points,
            'confirmations': confirmed_list
        }

    def _determine_setup_quality(self, score: int, confirmation_count: int) -> str:
        """Determine setup quality"""
        if score >= 91 and confirmation_count >= 7:
            return 'EXCELLENT'
        elif score >= 81 and confirmation_count >= 5:
            return 'VERY_GOOD'
        elif score >= 71 and confirmation_count >= 3:
            return 'GOOD'
        elif score >= 60:
            return 'FAIR'
        else:
            return 'POOR'

    def _generate_trading_plan(self, zone: Dict, momentum: Dict, current_price: float) -> Dict:
        """Generate complete trading plan with ATR-based stops"""
        atr_stops = momentum['atr']['stops']

        if zone['zone_type'] == 'DEMAND':
            return {
                'direction': 'LONG',
                'entry_zone': f"${zone['zone_bottom']:.2f} - ${zone['zone_top']:.2f}",
                'optimal_entry': f"${zone['zone_midpoint']:.2f}",
                'stop_loss': f"${atr_stops['long_stop']:.2f}",
                'target_1': f"${atr_stops['long_target_1']:.2f} (1:2 R/R)",
                'target_2': f"${atr_stops['long_target_2']:.2f} (1:3 R/R)",
                'risk_reward': '1:2 to 1:3',
                'position_size': 'Use 1-2% account risk'
            }
        else:
            return {
                'direction': 'SHORT',
                'entry_zone': f"${zone['zone_bottom']:.2f} - ${zone['zone_top']:.2f}",
                'optimal_entry': f"${zone['zone_midpoint']:.2f}",
                'stop_loss': f"${atr_stops['short_stop']:.2f}",
                'target_1': f"${atr_stops['short_target_1']:.2f} (1:2 R/R)",
                'target_2': f"${atr_stops['short_target_2']:.2f} (1:3 R/R)",
                'risk_reward': '1:2 to 1:3',
                'position_size': 'Use 1-2% account risk'
            }


if __name__ == "__main__":
    # Test
    import yfinance as yf
    from zone_detector import ZoneDetector

    logging.basicConfig(level=logging.INFO)

    print("Testing Enhanced Zone Analyzer with AAPL...")

    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="6mo", interval="1d")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={'date': 'timestamp'})

    current_price = float(df['close'].iloc[-1])

    # Detect zones
    detector = ZoneDetector()
    zones = detector.detect_zones(df, "AAPL")

    if zones:
        # Analyze first zone with all indicators
        enhanced_analyzer = EnhancedZoneAnalyzer()
        enhanced_zone = enhanced_analyzer.analyze_zone_complete(zones[0], df, current_price)

        print(f"\nZone Analysis:")
        print(f"Type: {enhanced_zone['zone_type']}")
        print(f"Range: ${enhanced_zone['zone_bottom']:.2f} - ${enhanced_zone['zone_top']:.2f}")
        print(f"Base Score: {enhanced_zone.get('strength_score', 0)}/100")
        print(f"Enhanced Score: {enhanced_zone['enhanced_score']}/100")
        print(f"Setup Quality: {enhanced_zone['setup_quality']}")
        print(f"\nConfirmations ({enhanced_zone['confirmation_count']}):")
        for conf in enhanced_zone['confirmations']:
            print(f"  ✅ {conf}")
        print(f"\nTrading Plan:")
        for key, value in enhanced_zone['trading_plan'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("No zones detected. Try a more volatile stock.")
