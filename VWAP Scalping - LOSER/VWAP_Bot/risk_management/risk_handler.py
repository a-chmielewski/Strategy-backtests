import numpy as np
from typing import Dict, Tuple, Optional
from logging_setup.logger import logger
import math

class RiskHandler:
    def __init__(self, config: Dict):
        """Initialize RiskHandler with configuration parameters"""
        self.config = config
        
        # Extract trading parameters
        trading_config = config.get('trading', {})
        self.risk_per_trade = trading_config.get('risk_per_trade', 0.01)  # 1% default
        self.leverage = trading_config.get('leverage', 50)
        self.max_position_size = trading_config.get('max_position_size', 5000)
        self.stop_loss_pct = trading_config.get('stop_loss_pct', 0.01)
        self.take_profit_pct = trading_config.get('take_profit_pct', 0.02)
        self.risk_atr_multiplier = trading_config.get('risk_atr_multiplier', 1.5)
        self.trailing_stop_atr_multiplier = trading_config.get('trailing_stop_atr_multiplier', 2.0)
        
        # Extract volatility parameters
        volatility_config = config.get('indicators', {}).get('volatility', {})
        self.volatility_lookback = volatility_config.get('lookback', 100)

    def get_position_size(self, current_price: float, equity: float) -> float:
        """Calculate position size based on risk management rules"""
        try:
            # Risk amount (configured percentage of equity)
            risk_amount = equity * self.risk_per_trade
            
            # Calculate position size in USDT with leverage
            position_size_usdt = risk_amount * self.leverage
            
            # Calculate maximum position size based on leverage and equity
            max_position_usdt = equity * self.leverage
            
            # Apply position limits
            position_size_usdt = min(position_size_usdt, max_position_usdt, self.max_position_size * current_price)
            
            # Ensure minimum position size (Bybit requirement)
            MIN_ORDER_SIZE_USDT = 1.0
            position_size_usdt = max(position_size_usdt, MIN_ORDER_SIZE_USDT)
            
            # Round down to 2 decimal places to ensure valid order size
            position_size_usdt = math.floor(position_size_usdt * 100) / 100
            
            logger.info(f"Calculated position size: {position_size_usdt} USDT (Equity: ${equity:.2f})")
            return position_size_usdt
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def get_dynamic_risk_parameters(self, current_price: float, current_atr: float, 
                                  volatility_data: np.ndarray) -> Tuple[float, float]:
        """Calculate dynamic risk parameters based on market volatility"""
        try:
            # Check if we have valid volatility data
            if len(volatility_data) < self.volatility_lookback:
                logger.warning("Insufficient volatility data for dynamic risk calculation, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            recent_volatility = volatility_data[-self.volatility_lookback:]
            
            # Check for valid data
            if np.all(np.isnan(recent_volatility)):
                logger.warning("All NaN values in volatility data, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            # Convert recent_volatility to a NumPy array if it's not already
            recent_volatility = np.array(recent_volatility)
            
            # Calculate volatility factor with additional checks
            mean_recent_volatility = np.nanmean(recent_volatility)
            if mean_recent_volatility == 0 or np.isnan(mean_recent_volatility):
                logger.warning("Invalid mean volatility, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            current_vol = volatility_data[-1]
            if np.isnan(current_vol):
                logger.warning("Current volatility is NaN, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            volatility_factor = (current_vol / mean_recent_volatility) if mean_recent_volatility != 0 else 1.0
            logger.debug(f"Mean Recent Volatility: {mean_recent_volatility}, Current Volatility: {current_vol}, Volatility Factor: {volatility_factor}")
            
            # Adjust stop loss and take profit based on volatility
            dynamic_stop_loss = max(
                self.stop_loss_pct,
                self.stop_loss_pct * volatility_factor
            )
            dynamic_take_profit = max(
                self.take_profit_pct,
                self.take_profit_pct * volatility_factor
            )
            
            # Use ATR for minimum stop distance
            min_stop_distance = current_atr * self.risk_atr_multiplier
            price_based_stop = min_stop_distance / current_price
            
            # Use the larger of the two stops
            final_stop_loss = max(dynamic_stop_loss, price_based_stop)
            final_take_profit = max(dynamic_take_profit, price_based_stop * 2)
            
            # Ensure that stop loss and take profit are within logical bounds
            final_stop_loss = min(final_stop_loss, 1.0)  # Stop loss cannot exceed 100%
            final_take_profit = min(final_take_profit, 3.0)  # Arbitrary upper limit for take profit
            
            return final_stop_loss, final_take_profit
            
        except Exception as e:
            logger.error(f"Error calculating dynamic risk parameters: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return self.stop_loss_pct, self.take_profit_pct

    def calculate_trailing_stop(self, current_price: float, current_atr: float, 
                              position_side: str) -> float:
        """Calculate trailing stop level based on ATR"""
        try:
            if position_side == 'long':
                return current_price - (current_atr * self.trailing_stop_atr_multiplier)
            else:
                return current_price + (current_atr * self.trailing_stop_atr_multiplier)
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {str(e)}")
            return None
