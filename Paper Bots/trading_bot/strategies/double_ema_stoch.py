import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import pandas_ta as ta

@dataclass
class StrategyState:
    ema_slow: List[float]
    ema_fast: List[float]
    stoch_k: List[float]
    stoch_d: List[float]
    high_prices: List[float]
    low_prices: List[float]
    close_prices: List[float]
    swing_high: float
    swing_low: float

class DoubleEMAStochStrategy:
    def __init__(self, 
                 ema_fast: int = 50,
                 ema_slow: int = 150,
                 stoch_k: int = 5,
                 stoch_d: int = 3,
                 slowing: int = 3,
                 stoch_overbought: int = 80,
                 stoch_oversold: int = 20,
                 stop_loss: float = 0.0025,
                 take_profit: float = 0.005):
        
        self.ema_slow = ema_slow
        self.ema_fast = ema_fast
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.slowing = slowing
        self.stoch_overbought = stoch_overbought
        self.stoch_oversold = stoch_oversold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Initialize state
        self.prev_stoch_k = 0
        self.prev_stoch_d = 0
        self.swing_high = 0.0
        self.swing_low = float('inf')
        
        # Initialize indicator values
        self.ema_slow_value = None
        self.ema_fast_value = None
        self.stoch_k_value = None
        self.stoch_d_value = None

    def calculate_indicators(self, high: float, low: float, close: float) -> None:
        # Update swing points
        self.swing_high = max(high, self.swing_high)
        self.swing_low = min(low, self.swing_low)
        
        # Store previous stoch values
        self.prev_stoch_k = self.stoch_k_value if self.stoch_k_value is not None else 0
        self.prev_stoch_d = self.stoch_d_value if self.stoch_d_value is not None else 0
        
        # Calculate new indicator values using pandas_ta
        df = pd.DataFrame({
            'high': [high], 
            'low': [low], 
            'close': [close]
        })
        
        try:
            # Calculate EMAs
            ema_slow_series = ta.ema(df['close'], length=self.ema_slow)
            ema_fast_series = ta.ema(df['close'], length=self.ema_fast)
            
            # Calculate Stochastic
            stoch = ta.stoch(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                k=self.stoch_k,
                d=self.stoch_d,
                smooth_k=self.slowing
            )
            
            # Update indicator values
            if ema_slow_series is not None and not ema_slow_series.empty:
                self.ema_slow_value = ema_slow_series.iloc[-1]
            
            if ema_fast_series is not None and not ema_fast_series.empty:
                self.ema_fast_value = ema_fast_series.iloc[-1]
            
            if stoch is not None and not stoch.empty:
                k_col = f'STOCHk_{self.stoch_k}_{self.stoch_d}_{self.slowing}'
                d_col = f'STOCHd_{self.stoch_k}_{self.stoch_d}_{self.slowing}'
                
                if k_col in stoch.columns and d_col in stoch.columns:
                    k_value = stoch[k_col].iloc[-1]
                    d_value = stoch[d_col].iloc[-1]
                    
                    if not pd.isna(k_value) and not pd.isna(d_value):
                        self.stoch_k_value = k_value
                        self.stoch_d_value = d_value
                    
        except Exception as e:
            # Log the error but don't raise it - let the strategy continue with previous values
            print(f"Error calculating indicators: {str(e)}")

    def generate_signal(self, current_price: float) -> Optional[Dict]:
        if None in (self.ema_slow_value, self.ema_fast_value, 
                   self.stoch_k_value, self.stoch_d_value):
            return None
            
        # Calculate if stochastic is crossing
        stoch_k_direction = self.stoch_k_value - self.prev_stoch_k
        stoch_d_direction = self.stoch_d_value - self.prev_stoch_d
        
        # Check if price is near EMA
        price_at_ema = (current_price <= self.ema_fast_value * 1.001 and 
                       current_price >= self.ema_fast_value * 0.999)
        
        # Long conditions
        if (self.ema_slow_value > self.ema_fast_value and  # Uptrend
            price_at_ema and
            self.stoch_k_value <= self.stoch_oversold and 
            self.stoch_d_value <= self.stoch_oversold and
            stoch_k_direction > 0 and 
            stoch_d_direction > 0):
            
            return {
                'action': 'BUY',
                'stop_loss': self.swing_low * (1 - self.stop_loss),
                'take_profit': current_price * (1 + self.take_profit)
            }
            
        # Short conditions
        elif (self.ema_slow_value < self.ema_fast_value and  # Downtrend
              price_at_ema and
              self.stoch_k_value >= self.stoch_overbought and 
              self.stoch_d_value >= self.stoch_overbought and
              stoch_k_direction < 0 and 
              stoch_d_direction < 0):
              
            return {
                'action': 'SELL',
                'stop_loss': self.swing_high * (1 + self.stop_loss),
                'take_profit': current_price * (1 - self.take_profit)
            }
            
        return None 