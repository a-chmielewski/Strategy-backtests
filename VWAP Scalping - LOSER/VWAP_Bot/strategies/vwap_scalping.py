import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .base_strategy import BaseStrategy
from logging_setup.logger import logger, refresh_logger
from indicators.technical_indicators import TechnicalIndicators

class VWAPScalpingStrategy(BaseStrategy):
    def __init__(self, config: Dict, indicators: TechnicalIndicators):
        """Initialize the VWAP Scalping Strategy"""
        self.config = config
        self.indicators = indicators
        
        # Initialize trading parameters
        trading_config = config.get('trading', {})
        self.stop_loss_pct = trading_config.get('stop_loss_pct', 0.01)
        self.take_profit_pct = trading_config.get('take_profit_pct', 0.02)
        self.trailing_stop_atr_multiplier = trading_config.get('trailing_stop_atr_multiplier', 2.0)
        self.risk_atr_multiplier = trading_config.get('risk_atr_multiplier', 1.5)
        
        # Initialize position tracking
        self.current_position = {'size': 0, 'side': None, 'entry_price': None}
        self.trailing_stop = None


    def update_trailing_stop(self, current_price: float, current_atr: float, 
                           dynamic_stop_loss: Optional[float] = None) -> None:
        """Update trailing stop based on current market conditions"""
        try:
            if self.trailing_stop is None:
                return

            # Calculate minimum stop distance based on ATR
            min_stop_distance = current_atr * self.risk_atr_multiplier

            if self.current_position['side'] == 'long':
                # For long positions, calculate new stop with dynamic parameters
                new_stop = current_price - max(
                    min_stop_distance,
                    current_atr * self.trailing_stop_atr_multiplier
                )
                # Use max to ensure stop only moves up
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    refresh_logger.info(
                        f"Updated long trailing stop up to: {self.trailing_stop:.2f} "
                        f"(Price: {current_price:.2f}, Dynamic SL: {dynamic_stop_loss:.4f})"
                    )
            else:  # short position
                # For short positions, calculate new stop with dynamic parameters
                new_stop = current_price + max(
                    min_stop_distance,
                    current_atr * self.trailing_stop_atr_multiplier
                )
                # Use min to ensure stop only moves down
                if new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                    refresh_logger.info(
                        f"Updated short trailing stop down to: {self.trailing_stop:.2f} "
                        f"(Price: {current_price:.2f}, Dynamic SL: {dynamic_stop_loss:.4f})"
                    )

            # Calculate and log distance to trailing stop with dynamic parameters
            stop_distance = abs(current_price - self.trailing_stop)
            stop_distance_pct = (stop_distance / current_price) * 100
            refresh_logger.info(
                f"Trailing stop distance: {stop_distance:.2f} ({stop_distance_pct:.2f}%) "
                f"Dynamic SL: {dynamic_stop_loss:.4f}"
            )

        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")

    def check_entry_conditions(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Check if entry conditions are met"""
        try:
            # Extract necessary data
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Long conditions
            long_condition = (
                previous['close'] < previous['vwap'] and
                current['close'] > current['vwap'] and
                current['rsi'] < self.indicators.get('rsi_dynamic_overbought', 70)
            )
            
            # Short conditions
            short_condition = (
                previous['close'] > previous['vwap'] and
                current['close'] < current['vwap'] and
                current['rsi'] > self.indicators.get('rsi_dynamic_oversold', 30)
            )
            
            if long_condition:
                logger.info("Long entry conditions met")
                return True, "long"
            elif short_condition:
                logger.info("Short entry conditions met")
                return True, "short"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {str(e)}")
            return False, ""

    def check_exit_conditions(self, data: pd.DataFrame, position_data: Dict) -> Tuple[bool, str]:
        """Check if exit conditions are met"""
        try:
            if not position_data or not position_data.get('side'):
                return False, ""

            current_price = data.iloc[-1]['close']
            entry_price = position_data['entry_price']
            
            # Calculate dynamic risk parameters
            current_atr = self.indicators.get('atr', [0.0])[-1]
            dynamic_sl, dynamic_tp = self.get_dynamic_risk_parameters(current_price, current_atr)
            
            # Update trailing stop
            self.update_trailing_stop(current_price, current_atr, dynamic_sl)
            
            # Check trailing stop
            if self.trailing_stop is not None:
                if (position_data['side'] == 'long' and current_price <= self.trailing_stop) or \
                   (position_data['side'] == 'short' and current_price >= self.trailing_stop):
                    logger.info("Trailing stop triggered")
                    return True, "trailing_stop"
            
            # Check take profit
            take_profit = entry_price * (1 + dynamic_tp) if position_data['side'] == 'long' else entry_price * (1 - dynamic_tp)
            if (position_data['side'] == 'long' and current_price >= take_profit) or \
               (position_data['side'] == 'short' and current_price <= take_profit):
                logger.info("Take profit reached")
                return True, "take_profit"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            return False, ""

    def get_dynamic_risk_parameters(self, current_price: float, current_atr: float) -> Tuple[float, float]:
        """Calculate dynamic risk parameters based on market conditions"""
        try:
            volatility_data = self.indicators.get('volatility', [])
            if len(volatility_data) < 20:
                return self.stop_loss_pct, self.take_profit_pct

            recent_volatility = volatility_data[-20:]
            mean_recent_volatility = np.nanmean(recent_volatility)
            
            if np.isnan(mean_recent_volatility) or mean_recent_volatility == 0:
                return self.stop_loss_pct, self.take_profit_pct
            
            volatility_factor = volatility_data[-1] / mean_recent_volatility
            
            # Calculate dynamic parameters
            dynamic_sl = max(
                self.stop_loss_pct,
                self.stop_loss_pct * volatility_factor,
                (current_atr * self.risk_atr_multiplier) / current_price
            )
            
            dynamic_tp = max(
                self.take_profit_pct,
                self.take_profit_pct * volatility_factor,
                dynamic_sl * 2
            )
            
            # Apply bounds
            dynamic_sl = min(dynamic_sl, 1.0)
            dynamic_tp = min(dynamic_tp, 3.0)
            
            return dynamic_sl, dynamic_tp
            
        except Exception as e:
            logger.error(f"Error calculating dynamic risk parameters: {str(e)}")
            return self.stop_loss_pct, self.take_profit_pct
