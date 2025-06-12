import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer
from results_logger import log_result

LEVERAGE = 50

class Volatility_Reversal_Scalp_Strategy(bt.Strategy):
    params = (
        ("bb_period", 20),
        ("bb_std", 2.0),
        ("bb_extreme_std", 2.5),  # Reduced from 3.0 to 2.5 for easier detection
        ("atr_period", 14),
        ("atr_extreme_multiplier", 2.5),  # Reduced from 4.0 to 2.5
        ("rsi_period", 7),  # Fast RSI for extremes
        ("rsi_overbought_extreme", 80),  # Reduced from 85 to 80
        ("rsi_oversold_extreme", 20),    # Increased from 15 to 20
        ("rsi_overbought_moderate", 75), # New moderate level
        ("rsi_oversold_moderate", 25),   # New moderate level
        ("volume_climax_multiplier", 2.0),  # Reduced from 3.0 to 2.0
        ("volume_avg_period", 20),
        ("ema_period", 20),  # For mean reversion target
        ("reversal_confirmation_bars", 3),  # Increased from 2 to allow more time
        ("fibonacci_retracement_1", 0.382),  # 38.2% retracement target
        ("fibonacci_retracement_2", 0.5),    # 50% retracement target
        ("max_hold_bars", 8),  # Increased from 5 to 8
        ("stop_buffer_pct", 0.002),  # Increased buffer for volatility
        ("min_spike_size_pct", 0.005),  # Reduced from 0.01 to 0.005
        ("position_size_reduction", 0.7),  # Increased from 0.5 to 0.7
        ("partial_exit_pct", 0.5),  # Take 50% profit at first target
        ("max_consecutive_losses", 4),  # Increased from 3 to 4
        ("min_score_threshold", 3),  # Minimum score to enter trade
        ("immediate_entry_score", 5),  # Score for immediate entry without reversal wait
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.p.bb_period, 
            devfactor=self.p.bb_std
        )
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        # Create a simple RSI-like indicator to avoid division by zero
        class SafeRSI(bt.Indicator):
            lines = ('rsi',)
            params = (('period', 14),)
            
            def __init__(self):
                # Calculate price changes
                price_change = self.data.close - self.data.close(-1)
                
                # Separate gains and losses with safety checks
                gain = bt.Max(price_change, 0.0)
                loss = bt.Max(-price_change, 0.0)
                
                # Use SMA for gains and losses (safer than EMA for division by zero)
                avg_gain = bt.indicators.SMA(gain, period=self.p.period)
                avg_loss = bt.indicators.SMA(loss, period=self.p.period)
                
                # Safe RSI calculation with division by zero protection
                self.lines.rsi = 100.0 - (100.0 / (1.0 + (avg_gain / bt.Max(avg_loss, 1e-10))))
        
        self.rsi = SafeRSI(period=self.p.rsi_period)
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        
        # Spike detection variables
        self.extreme_spike_detected = False
        self.spike_direction = None  # 'up' or 'down'
        self.spike_high = 0
        self.spike_low = 0
        self.spike_bar = None
        self.spike_volume = 0
        
        # Reversal pattern tracking
        self.reversal_pattern_detected = False
        self.pattern_type = None
        self.pattern_confirmation_bar = None
        
        # Risk management
        self.consecutive_losses = 0
        self.trading_enabled = True
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.spike_size = 0
        self.first_target_hit = False
        
        # Order and position tracking
        self.order = None
        self.order_parent_ref = None
        self.trade_dir = {}
        self.closing = False

    def calculate_position_size(self, current_price):
        try:
            if current_price is None or current_price == 0:
                return 0
            current_equity = self.broker.getvalue()

            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0

            leverage = LEVERAGE

            # Reduce position size for contrarian trades
            position_value *= self.p.position_size_reduction

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def evaluate_spike_conditions(self):
        """Evaluate spike conditions using a scoring system instead of requiring all conditions"""
        if (len(self.bb) == 0 or len(self.atr) == 0 or 
            len(self.rsi) == 0 or len(self.volume_sma) == 0):
            return 0, None, None
            
        try:
            current_price = self.data.close[0]
            current_high = self.data.high[0]
            current_low = self.data.low[0]
            current_volume = self.data.volume[0]
            
            bb_upper = self.bb.lines.top[0]
            bb_lower = self.bb.lines.bot[0]
            bb_middle = self.bb.lines.mid[0]
            
            # Safety checks for division by zero
            if (bb_middle == 0 or np.isnan(bb_middle) or np.isinf(bb_middle) or
                current_price == 0 or np.isnan(current_price) or np.isinf(current_price)):
                return 0, None, None
                
        except (IndexError, AttributeError, TypeError):
            return 0, None, None
        
        # Calculate extreme Bollinger Band levels
        bb_range = bb_upper - bb_lower
        bb_extreme_upper = bb_middle + (bb_range * self.p.bb_extreme_std / 2)
        bb_extreme_lower = bb_middle - (bb_range * self.p.bb_extreme_std / 2)
        
        atr_value = self.atr[0]
        rsi_value = self.rsi[0]
        avg_volume = self.volume_sma[0]
        
        # Additional safety checks for all indicators
        if (atr_value == 0 or np.isnan(atr_value) or np.isinf(atr_value) or
            np.isnan(rsi_value) or np.isinf(rsi_value) or
            avg_volume == 0 or np.isnan(avg_volume) or np.isinf(avg_volume)):
            return 0, None, None
        
        # Calculate candle metrics
        candle_range = current_high - current_low
        candle_body = abs(self.data.close[0] - self.data.open[0])
        candle_body_pct = candle_body / current_price if current_price > 0 else 0
        
        # Initialize scores for up and down spikes
        up_score = 0
        down_score = 0
        
        # UP-SPIKE SCORING
        # 1. Bollinger Band position (2 points for extreme, 1 for regular breach)
        if current_price > bb_extreme_upper:
            up_score += 2
        elif current_price > bb_upper:
            up_score += 1
            
        # 2. RSI levels (2 points for extreme, 1 for moderate)
        if rsi_value >= self.p.rsi_overbought_extreme:
            up_score += 2
        elif rsi_value >= self.p.rsi_overbought_moderate:
            up_score += 1
            
        # 3. Volume climax (2 points for high volume, 1 for elevated)
        if current_volume > avg_volume * self.p.volume_climax_multiplier:
            up_score += 2
        elif current_volume > avg_volume * (self.p.volume_climax_multiplier * 0.7):
            up_score += 1
            
        # 4. Candle size relative to ATR (2 points for extreme, 1 for large)
        if candle_range > atr_value * self.p.atr_extreme_multiplier:
            up_score += 2
        elif candle_range > atr_value * (self.p.atr_extreme_multiplier * 0.7):
            up_score += 1
            
        # 5. Minimum body size (1 point if met)
        if candle_body_pct > self.p.min_spike_size_pct:
            up_score += 1
            
        # 6. Bullish candle confirmation (1 point for green candle on up-spike)
        if self.data.close[0] > self.data.open[0]:
            up_score += 1
            
        # DOWN-SPIKE SCORING
        # 1. Bollinger Band position
        if current_price < bb_extreme_lower:
            down_score += 2
        elif current_price < bb_lower:
            down_score += 1
            
        # 2. RSI levels
        if rsi_value <= self.p.rsi_oversold_extreme:
            down_score += 2
        elif rsi_value <= self.p.rsi_oversold_moderate:
            down_score += 1
            
        # 3. Volume climax
        if current_volume > avg_volume * self.p.volume_climax_multiplier:
            down_score += 2
        elif current_volume > avg_volume * (self.p.volume_climax_multiplier * 0.7):
            down_score += 1
            
        # 4. Candle size relative to ATR
        if candle_range > atr_value * self.p.atr_extreme_multiplier:
            down_score += 2
        elif candle_range > atr_value * (self.p.atr_extreme_multiplier * 0.7):
            down_score += 1
            
        # 5. Minimum body size
        if candle_body_pct > self.p.min_spike_size_pct:
            down_score += 1
            
        # 6. Bearish candle confirmation (1 point for red candle on down-spike)
        if self.data.close[0] < self.data.open[0]:
            down_score += 1
            
        # Determine direction and return best score
        if up_score >= down_score and up_score >= self.p.min_score_threshold:
            spike_size = (current_high - bb_middle) / bb_middle if bb_middle != 0 and not np.isnan(bb_middle) and not np.isinf(bb_middle) else 0
            return up_score, 'up', {'direction': 'up', 'size': spike_size, 'extreme_price': current_high}
        elif down_score > up_score and down_score >= self.p.min_score_threshold:
            spike_size = (bb_middle - current_low) / bb_middle if bb_middle != 0 and not np.isnan(bb_middle) and not np.isinf(bb_middle) else 0
            return down_score, 'down', {'direction': 'down', 'size': spike_size, 'extreme_price': current_low}
        
        return 0, None, None

    def is_extreme_spike(self):
        """Legacy method - now uses scoring system"""
        score, direction, spike_data = self.evaluate_spike_conditions()
        return score >= self.p.min_score_threshold, spike_data

    def detect_reversal_pattern(self, spike_direction):
        """Detect reversal candlestick patterns after extreme spike"""
        if len(self) < 2:
            return False, None
            
        # Current and previous candle data
        curr_open = self.data.open[0]
        curr_high = self.data.high[0]
        curr_low = self.data.low[0]
        curr_close = self.data.close[0]
        
        prev_open = self.data.open[-1]
        prev_high = self.data.high[-1]
        prev_low = self.data.low[-1]
        prev_close = self.data.close[-1]
        
        # After up-spike, look for bearish reversal patterns
        if spike_direction == 'up':
            # Shooting star (long upper shadow, small body)
            body_size = abs(curr_close - curr_open)
            upper_shadow = curr_high - max(curr_open, curr_close)
            lower_shadow = min(curr_open, curr_close) - curr_low
            
            if (upper_shadow > 2 * body_size and  # Long upper shadow
                lower_shadow < body_size * 0.5 and  # Small lower shadow
                curr_high < prev_high):  # Failed to make new high
                return True, 'shooting_star'
            
            # Bearish engulfing
            if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Opens above prev close
                curr_close < prev_open and  # Closes below prev open
                curr_high <= prev_high):  # Failed to exceed previous high
                return True, 'bearish_engulfing'
            
            # Simple failure to make new high with red candle
            if (curr_close < curr_open and  # Red candle
                curr_high <= prev_high and  # No new high
                curr_close < prev_close):  # Closes below previous close
                return True, 'failed_breakout_bearish'
        
        # After down-spike, look for bullish reversal patterns
        elif spike_direction == 'down':
            # Hammer (long lower shadow, small body)
            body_size = abs(curr_close - curr_open)
            lower_shadow = min(curr_open, curr_close) - curr_low
            upper_shadow = curr_high - max(curr_open, curr_close)
            
            if (lower_shadow > 2 * body_size and  # Long lower shadow
                upper_shadow < body_size * 0.5 and  # Small upper shadow
                curr_low > prev_low):  # Failed to make new low
                return True, 'hammer'
            
            # Bullish engulfing
            if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Opens below prev close
                curr_close > prev_open and  # Closes above prev open
                curr_low >= prev_low):  # Failed to go below previous low
                return True, 'bullish_engulfing'
            
            # Simple failure to make new low with green candle
            if (curr_close > curr_open and  # Green candle
                curr_low >= prev_low and  # No new low
                curr_close > prev_close):  # Closes above previous close
                return True, 'failed_breakout_bullish'
        
        return False, None

    def check_rsi_divergence(self, spike_direction):
        """Check for RSI divergence at extremes"""
        if len(self.rsi) < 5:
            return False
            
        current_price = self.data.close[0]
        current_rsi = self.rsi[0]
        
        # Look back for previous extreme
        for i in range(2, min(10, len(self.rsi))):
            past_price = self.data.close[-i]
            past_rsi = self.rsi[-i]
            
            if spike_direction == 'up':
                # Bearish divergence: higher high in price, lower high in RSI
                if (current_price > past_price and 
                    current_rsi < past_rsi and 
                    past_rsi >= self.p.rsi_overbought_extreme):
                    return True
            elif spike_direction == 'down':
                # Bullish divergence: lower low in price, higher low in RSI
                if (current_price < past_price and 
                    current_rsi > past_rsi and 
                    past_rsi <= self.p.rsi_oversold_extreme):
                    return True
                    
        return False

    def calculate_retracement_targets(self, spike_data):
        """Calculate Fibonacci retracement targets"""
        if not spike_data:
            return None, None
            
        bb_middle = self.bb.lines.mid[0]
        spike_price = spike_data['extreme_price']
        
        if spike_data['direction'] == 'up':
            # For up-spike fade, calculate downward retracement
            spike_size = spike_price - bb_middle
            fib_382 = spike_price - (spike_size * self.p.fibonacci_retracement_1)
            fib_50 = spike_price - (spike_size * self.p.fibonacci_retracement_2)
            return fib_382, fib_50
        else:
            # For down-spike fade, calculate upward retracement
            spike_size = bb_middle - spike_price
            fib_382 = spike_price + (spike_size * self.p.fibonacci_retracement_1)
            fib_50 = spike_price + (spike_size * self.p.fibonacci_retracement_2)
            return fib_382, fib_50

    def check_exit_conditions(self):
        """Check for reversal scalp exit conditions"""
        if not self.position:
            return False, None
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Calculate key levels
        bb_middle = self.bb.lines.mid[0]
        bb_upper = self.bb.lines.top[0]
        bb_lower = self.bb.lines.bot[0]
        ema_level = self.ema[0]
        rsi_value = self.rsi[0]
        
        # Time-based exit (but more lenient)
        if bars_held >= self.p.max_hold_bars:
            return True, "time_exit"
        
        # Quick stop-loss check (moved prices significantly against us)
        if pos.size > 0:  # Long position
            # Emergency stop if we break below recent support
            if bars_held >= 2 and current_price < self.entry_price * 0.995:  # 0.5% stop
                return True, "emergency_stop_long"
                
        elif pos.size < 0:  # Short position  
            # Emergency stop if we break above recent resistance
            if bars_held >= 2 and current_price > self.entry_price * 1.005:  # 0.5% stop
                return True, "emergency_stop_short"
        
        if pos.size > 0:  # Long position (fading down-spike)
            # Target 1: Quick scalp target (even smaller move)
            if current_price >= self.entry_price * 1.002 and not self.first_target_hit:  # 0.2% profit
                self.first_target_hit = True
                return True, "quick_scalp_long"
            
            # Target 2: BB middle or EMA
            target_1 = max(ema_level, bb_middle)
            if current_price >= target_1 and bars_held >= 1:
                return True, "first_target_long"
                
            # Target 3: Further retracement
            if self.first_target_hit and current_price >= target_1 * 1.005:
                return True, "second_target_long"
                
            # RSI reversal signal
            if rsi_value >= 70 and bars_held >= 1:  # RSI getting overbought again
                return True, "rsi_reversal_long"
                
        elif pos.size < 0:  # Short position (fading up-spike)
            # Target 1: Quick scalp target
            if current_price <= self.entry_price * 0.998 and not self.first_target_hit:  # 0.2% profit
                self.first_target_hit = True
                return True, "quick_scalp_short"
            
            # Target 2: BB middle or EMA
            target_1 = min(ema_level, bb_middle)
            if current_price <= target_1 and bars_held >= 1:
                return True, "first_target_short"
                
            # Target 3: Further retracement
            if self.first_target_hit and current_price <= target_1 * 0.995:
                return True, "second_target_short"
                
            # RSI reversal signal
            if rsi_value <= 30 and bars_held >= 1:  # RSI getting oversold again
                return True, "rsi_reversal_short"
        
        # Momentum reversal against us (price continues in original spike direction)
        if bars_held >= 2:
            if pos.size > 0 and current_price < bb_lower:  # Long but price going lower
                return True, "momentum_reversal_long"
            elif pos.size < 0 and current_price > bb_upper:  # Short but price going higher
                return True, "momentum_reversal_short"
        
        return False, None

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.bb_period, self.p.atr_period, self.p.rsi_period, self.p.ema_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Additional safety checks for indicator values
        try:
            # Check if indicators have valid values
            if (len(self.bb) == 0 or len(self.atr) == 0 or 
                len(self.rsi) == 0 or len(self.ema) == 0 or len(self.volume_sma) == 0):
                return
                
            # Check for NaN or infinite values in indicators
            bb_top = self.bb.lines.top[0]
            bb_bottom = self.bb.lines.bot[0] 
            bb_mid = self.bb.lines.mid[0]
            atr_val = self.atr[0]
            rsi_val = self.rsi[0]
            ema_val = self.ema[0]
            volume_sma_val = self.volume_sma[0]
            
            if (np.isnan(bb_top) or np.isnan(bb_bottom) or np.isnan(bb_mid) or
                np.isnan(atr_val) or np.isnan(rsi_val) or np.isnan(ema_val) or
                np.isnan(volume_sma_val) or np.isinf(bb_top) or np.isinf(bb_bottom) or
                np.isinf(bb_mid) or np.isinf(atr_val) or np.isinf(rsi_val) or
                np.isinf(ema_val) or np.isinf(volume_sma_val)):
                return
                
            # Check for zero values that could cause division by zero
            if bb_mid == 0 or atr_val == 0 or volume_sma_val == 0:
                return
                
        except (IndexError, AttributeError, TypeError):
            # If we can't access indicator values safely, skip this bar
            return
        
        # Check if trading is disabled due to consecutive losses
        if not self.trading_enabled:
            return
        
        if not self.position:  # If no position is open
            # Evaluate current spike conditions
            score, direction, spike_data = self.evaluate_spike_conditions()
            
            # Method 1: Immediate entry on very high score (strong signals)
            if score >= self.p.immediate_entry_score and spike_data:
                position_size = self.calculate_position_size(current_price)
                
                if position_size > 0:
                    self._enter_contrarian_trade(spike_data, position_size, current_price, "immediate_high_score")
                    return
            
            # Method 2: Traditional spike detection and reversal confirmation
            if score >= self.p.min_score_threshold and spike_data:
                self.extreme_spike_detected = True
                self.spike_direction = spike_data['direction']
                self.spike_size = spike_data['size']
                
                if spike_data['direction'] == 'up':
                    self.spike_high = spike_data['extreme_price']
                else:
                    self.spike_low = spike_data['extreme_price']
                    
                self.spike_bar = len(self)
                return  # Wait for next bar to check reversal
            
            # Method 3: Check for reversal pattern after detected spike
            if self.extreme_spike_detected and len(self) - self.spike_bar <= self.p.reversal_confirmation_bars:
                pattern_detected, pattern_type = self.detect_reversal_pattern(self.spike_direction)
                divergence_confirmed = self.check_rsi_divergence(self.spike_direction)
                
                # Enter if we have pattern OR divergence (more flexible)
                if pattern_detected or divergence_confirmed:
                    position_size = self.calculate_position_size(current_price)
                    
                    if position_size > 0:
                        spike_data = {
                            'direction': self.spike_direction,
                            'extreme_price': self.spike_high if self.spike_direction == 'up' else self.spike_low
                        }
                        self._enter_contrarian_trade(spike_data, position_size, current_price, "reversal_pattern")
                        
                        # Reset spike detection
                        self.extreme_spike_detected = False
                        self.spike_direction = None
            
            # Method 4: Alternative entry on moderate BB breach with volume + RSI
            elif not self.extreme_spike_detected:
                bb_upper = self.bb.lines.top[0]
                bb_lower = self.bb.lines.bot[0]
                rsi_value = self.rsi[0]
                current_volume = self.data.volume[0]
                avg_volume = self.volume_sma[0]
                
                # Moderate overbought condition
                if (current_price > bb_upper and 
                    rsi_value >= self.p.rsi_overbought_moderate and
                    current_volume > avg_volume * 1.5):  # 1.5x volume instead of 2x
                    
                    position_size = self.calculate_position_size(current_price)
                    if position_size > 0:
                        spike_data = {'direction': 'up', 'extreme_price': current_price}
                        self._enter_contrarian_trade(spike_data, position_size, current_price, "moderate_overbought")
                
                # Moderate oversold condition        
                elif (current_price < bb_lower and 
                      rsi_value <= self.p.rsi_oversold_moderate and
                      current_volume > avg_volume * 1.5):
                    
                    position_size = self.calculate_position_size(current_price)
                    if position_size > 0:
                        spike_data = {'direction': 'down', 'extreme_price': current_price}
                        self._enter_contrarian_trade(spike_data, position_size, current_price, "moderate_oversold")
            
            # Reset spike detection if too much time has passed
            elif self.extreme_spike_detected and len(self) - self.spike_bar > self.p.reversal_confirmation_bars:
                self.extreme_spike_detected = False
                self.spike_direction = None
                
        else:
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions()
            
            if should_exit:
                if exit_reason in ["first_target_long", "first_target_short"]:
                    # Take partial profits
                    pos = self.getposition()
                    partial_size = abs(pos.size) * self.p.partial_exit_pct
                    
                    if pos.size > 0:
                        self.sell(size=partial_size)
                    else:
                        self.buy(size=partial_size)
                else:
                    # Exit completely
                    self.close()
                    self.closing = True
                    self.entry_bar = None
                    self.entry_side = None
                    self.entry_price = None

    def _enter_contrarian_trade(self, spike_data, position_size, current_price, entry_reason):
        """Helper method to enter contrarian trades with proper stop and target setup"""
        try:
            if spike_data['direction'] == 'up':  # Fade the up-spike (go short)
                stop_price = spike_data['extreme_price'] * (1 + self.p.stop_buffer_pct)
                target_1, target_2 = self.calculate_retracement_targets(spike_data)
                
                # Use market order for immediate entry, manual stop/target management
                self.order = self.sell(size=position_size, exectype=bt.Order.Market)
                self.entry_bar = len(self)
                self.entry_side = 'short'
                self.entry_price = current_price
                self.first_target_hit = False
                
            elif spike_data['direction'] == 'down':  # Fade the down-spike (go long)
                stop_price = spike_data['extreme_price'] * (1 - self.p.stop_buffer_pct)
                target_1, target_2 = self.calculate_retracement_targets(spike_data)
                
                # Use market order for immediate entry, manual stop/target management
                self.order = self.buy(size=position_size, exectype=bt.Order.Market)
                self.entry_bar = len(self)
                self.entry_side = 'long'
                self.entry_price = current_price
                self.first_target_hit = False
                
        except Exception as e:
            print(f"Error entering contrarian trade: {e}")
            self.order = None

    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            self.trade_dir[trade.ref] = 'long' if trade.size > 0 else 'short'
        if not trade.isclosed:
            return

        try:
            # Get entry and exit prices
            entry_price = trade.price
            exit_price = trade.history[-1].price if trade.history else self.data.close[0]
            pnl = trade.pnl
            
            # Track consecutive losses for risk management
            if pnl < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= self.p.max_consecutive_losses:
                    self.trading_enabled = False
                    print(f"Trading disabled after {self.consecutive_losses} consecutive losses")
            else:
                self.consecutive_losses = 0  # Reset on winning trade
            
            # Store trade exit information for visualization
            direction = self.trade_dir.get(trade.ref, None)
            trade_type = f'{direction}_exit' if direction in ('long', 'short') else 'unknown_exit'
            self.trade_exits.append({
                'entry_time': trade.dtopen,
                'exit_time': trade.dtclose,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': trade_type,
                'pnl': pnl
            })
            
            if trade.ref in self.trade_dir:
                del self.trade_dir[trade.ref]

        except Exception as e:
            print(f"Warning: Could not process trade: {str(e)}")
            print(f"Trade info - Status: {trade.status}, Size: {trade.size}, "
                  f"Price: {trade.price}, PnL: {trade.pnl}")

    def notify_order(self, order):
        # Only clear self.order if this is the parent (entry) order and matches entry side
        if self.order and order.ref == self.order.ref and order.parent is None and (
            (self.entry_side == 'long' and order.isbuy()) or (self.entry_side == 'short' and order.issell())
        ) and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None
            if hasattr(self, 'order_parent_ref') and order.ref == self.order_parent_ref:
                self.order_parent_ref = None
        
        # Clear closing flag if a close order is done
        if getattr(self, 'closing', False) and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.closing = False
            
        if order.status == order.Completed:
            if not order.parent:  # This is an entry order
                # Record trade start
                self.active_trades.append({
                    'entry_time': self.data.datetime.datetime(0),
                    'entry_price': order.executed.price,
                    'type': 'long' if order.isbuy() else 'short',
                    'size': order.executed.size
                })
            else:  # This is an exit order
                if self.active_trades:
                    trade = self.active_trades.pop()
                    # Record trade exit
                    self.trade_exits.append({
                        'entry_time': trade['entry_time'],
                        'entry_price': trade['entry_price'],
                        'exit_time': self.data.datetime.datetime(0),
                        'exit_price': order.executed.price,
                        'type': f"{trade['type']}_exit",
                        'pnl': (order.executed.price - trade['entry_price']) * trade['size'] if trade['type'] == 'long' 
                              else (trade['entry_price'] - order.executed.price) * trade['size']
                    })

def run_backtest(data, verbose=True, leverage=LEVERAGE, **kwargs):
    # Avoid zero-range bars: ensure High > Low on every bar
    mask = data["High"] <= data["Low"]
    if mask.any():
        data.loc[mask, "High"] = data.loc[mask, "Low"] + 1e-8
        
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(
        dataname=data,
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        datetime="datetime",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=None,
    )
    cerebro.adddata(feed)
    
    strategy_params = {
        "bb_period": kwargs.get("bb_period", 20),
        "bb_std": kwargs.get("bb_std", 2.0),
        "bb_extreme_std": kwargs.get("bb_extreme_std", 2.5),
        "atr_period": kwargs.get("atr_period", 14),
        "atr_extreme_multiplier": kwargs.get("atr_extreme_multiplier", 2.5),
        "rsi_period": kwargs.get("rsi_period", 7),
        "rsi_overbought_extreme": kwargs.get("rsi_overbought_extreme", 80),
        "rsi_oversold_extreme": kwargs.get("rsi_oversold_extreme", 20),
        "rsi_overbought_moderate": kwargs.get("rsi_overbought_moderate", 75),
        "rsi_oversold_moderate": kwargs.get("rsi_oversold_moderate", 25),
        "volume_climax_multiplier": kwargs.get("volume_climax_multiplier", 2.0),
        "volume_avg_period": kwargs.get("volume_avg_period", 20),
        "ema_period": kwargs.get("ema_period", 20),
        "reversal_confirmation_bars": kwargs.get("reversal_confirmation_bars", 3),
        "fibonacci_retracement_1": kwargs.get("fibonacci_retracement_1", 0.382),
        "fibonacci_retracement_2": kwargs.get("fibonacci_retracement_2", 0.5),
        "max_hold_bars": kwargs.get("max_hold_bars", 8),
        "stop_buffer_pct": kwargs.get("stop_buffer_pct", 0.002),
        "min_spike_size_pct": kwargs.get("min_spike_size_pct", 0.005),
        "position_size_reduction": kwargs.get("position_size_reduction", 0.7),
        "partial_exit_pct": kwargs.get("partial_exit_pct", 0.5),
        "max_consecutive_losses": kwargs.get("max_consecutive_losses", 4),
        "min_score_threshold": kwargs.get("min_score_threshold", 3),
        "immediate_entry_score": kwargs.get("immediate_entry_score", 5),
    }
    cerebro.addstrategy(Volatility_Reversal_Scalp_Strategy, **strategy_params)
    
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        margin=1.0 / leverage,
        commtype=bt.CommInfoBase.COMM_PERC
    )
    cerebro.broker.set_slippage_perc(0.0001)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(DetailedDrawdownAnalyzer, _name="detailed_drawdown")
    cerebro.addanalyzer(TradeRecorder, _name='trade_recorder')
    cerebro.addanalyzer(SQNAnalyzer, _name='sqn')
    
    results = cerebro.run()
    if len(results) > 0:
        strat = results[0][0] if isinstance(results[0], (list, tuple)) else results[0]
    else:
        raise ValueError("No results returned from backtest")

    trades = strat.analyzers.trade_recorder.get_analysis()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    total_trades = len(trades_df)
    win_trades = trades_df[trades_df['pnl'] > 0] if not trades_df.empty else pd.DataFrame()
    loss_trades = trades_df[trades_df['pnl'] < 0] if not trades_df.empty else pd.DataFrame()
    
    winrate = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0
    avg_trade = trades_df['pnl'].mean() if not trades_df.empty else 0
    best_trade = trades_df['pnl'].max() if not trades_df.empty else 0
    worst_trade = trades_df['pnl'].min() if not trades_df.empty else 0
    
    max_drawdown = 0
    avg_drawdown = 0
    try:
        dd = strat.analyzers.detailed_drawdown.get_analysis()
        max_drawdown = dd.get('max_drawdown', 0)
        avg_drawdown = dd.get('avg_drawdown', 0)
    except Exception as e:
        print(f"Error accessing drawdown analysis: {e}")

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    try:
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except Exception:
        sharpe_ratio = 0.0

    profit_factor = (win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())) if not loss_trades.empty else 0
    
    try:
        sqn = strat.analyzers.sqn.get_analysis()['sqn']
    except (AttributeError, KeyError):
        sqn = 0.0

    formatted_results = {
        "Start": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "End": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "Duration": f"{(data.datetime.iloc[-1] - data.datetime.iloc[0]).days} days",
        "Equity Final [$]": final_value,
        "Return [%]": total_return,
        "# Trades": total_trades,
        "Win Rate [%]": winrate,
        "Avg. Trade": avg_trade,
        "Best Trade": best_trade,
        "Worst Trade": worst_trade,
        "Max. Drawdown [%]": max_drawdown,
        "Avg. Drawdown [%]": avg_drawdown,
        "Sharpe Ratio": float(sharpe_ratio),
        "Profit Factor": profit_factor,
        "SQN": sqn,
    }
    
    if verbose:
        print("\n=== Strategy Performance Report ===")
        print(f"\nPeriod: {formatted_results['Start']} - {formatted_results['End']} ({formatted_results['Duration']})")
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"Final Capital: ${float(formatted_results['Equity Final [$]']):,.2f}")
        print(f"Total Return: {float(formatted_results['Return [%]']):,.2f}%")
        print(f"\nTotal Trades: {int(formatted_results['# Trades'])}")
        print(f"Win Rate: {float(formatted_results['Win Rate [%]']):.2f}%")
        print(f"Best Trade: {float(formatted_results['Best Trade']):.2f}")
        print(f"Worst Trade: {float(formatted_results['Worst Trade']):.2f}")
        print(f"Avg. Trade: {float(formatted_results['Avg. Trade']):.2f}")
        print(f"\nMax Drawdown: {float(formatted_results['Max. Drawdown [%]']):.2f}%")
        print(f"Sharpe Ratio: {float(formatted_results['Sharpe Ratio']):.2f}")
        print(f"Profit Factor: {float(formatted_results['Profit Factor']):.2f}")
    
    return formatted_results

def process_file(args):
    filename, data_folder, leverage = args
    data_path = os.path.join(data_folder, filename)
    
    try:
        parts = filename.split('-')
        symbol = parts[1]
        timeframe = parts[2]
    except (IndexError, ValueError) as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return (None, filename)

    print(f"\nTesting {symbol} {timeframe} with leverage {leverage}...")
    
    try:
        data_df = pd.read_csv(data_path)
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    except (IOError, ValueError) as e:
        print(f"Error reading or parsing data for {filename}: {str(e)}")
        return (None, filename)

    global LEVERAGE
    LEVERAGE = leverage
    
    results = run_backtest(
        data_df,
        verbose=False,
        bb_period=20,
        bb_std=2.0,
        bb_extreme_std=2.5,
        atr_period=14,
        atr_extreme_multiplier=2.5,
        rsi_period=7,
        rsi_overbought_extreme=80,
        rsi_oversold_extreme=20,
        rsi_overbought_moderate=75,
        rsi_oversold_moderate=25,
        volume_climax_multiplier=2.0,
        volume_avg_period=20,
        ema_period=20,
        reversal_confirmation_bars=3,
        fibonacci_retracement_1=0.382,
        fibonacci_retracement_2=0.5,
        max_hold_bars=8,
        stop_buffer_pct=0.002,
        min_spike_size_pct=0.005,
        position_size_reduction=0.7,
        partial_exit_pct=0.5,
        max_consecutive_losses=4,
        min_score_threshold=3,
        immediate_entry_score=5,
        leverage=leverage
    )
    
    log_result(
        strategy="Volatility_Reversal_Scalp",
        coinpair=symbol,
        timeframe=timeframe,
        leverage=leverage,
        results=results
    )
        
    summary = {
        'symbol': symbol,
        'timeframe': timeframe,
        'leverage': leverage,
        'winrate': results.get('Win Rate [%]', 0),
        'final_equity': results.get('Equity Final [$]', 0),
        'total_trades': results.get('# Trades', 0),
        'max_drawdown': results.get('Max. Drawdown [%]', 0)
    }
    return (summary, filename)

if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    data_folder = os.path.abspath(data_folder)
    
    try:
        files = [f for f in os.listdir(data_folder) if f.startswith('bybit-') and f.endswith('.csv')]
    except (OSError, IOError) as e:
        print(f"Error listing files in {data_folder}: {str(e)}")
        files = []

    all_results = []
    failed_files = []
    leverages = [1, 5, 10, 15, 25, 50]

    try:
        for leverage in leverages:
            print(f"\n==============================\nRunning all backtests for LEVERAGE = {leverage}\n==============================")
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(process_file, [(f, data_folder, leverage) for f in files]))
                
                for summary, fname in results:
                    if summary is not None:
                        all_results.append(summary)
                    else:
                        failed_files.append((fname, leverage))

            # Show top results for this leverage
            leverage_results = [r for r in all_results if r['leverage'] == leverage]
            sorted_results = sorted(leverage_results, key=lambda x: x['winrate'], reverse=True)[:3]
            
            print(f"\n=== Top 3 Results by Win Rate for LEVERAGE {leverage} ===")
            for i, result in enumerate(sorted_results, 1):
                print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
                print(f"Win Rate: {result['winrate']:.2f}%")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Final Equity: {result['final_equity']}")
                print(f"Max Drawdown: {result['max_drawdown']:.2f}%")

        if failed_files:
            print("\nThe following files failed to process:")
            for fname, lev in failed_files:
                print(f"- {fname} (leverage {lev})")

    except Exception as e:
        print("\nException occurred during processing:")
        print(str(e))
        print(traceback.format_exc())
        
        if all_results:
            try:
                # Show partial results for each leverage
                for leverage in leverages:
                    leverage_results = [r for r in all_results if r['leverage'] == leverage]
                    sorted_results = sorted(leverage_results, key=lambda x: x['winrate'], reverse=True)[:3]
                    
                    print(f"\n=== Top 3 Results by Win Rate (Partial) for LEVERAGE {leverage} ===")
                    for i, result in enumerate(sorted_results, 1):
                        print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
                        print(f"Win Rate: {result['winrate']:.2f}%")
                        print(f"Total Trades: {result['total_trades']}")
                        print(f"Final Equity: {result['final_equity']}")
                        print(f"Max Drawdown: {result['max_drawdown']:.2f}%")

                if failed_files:
                    print("\nThe following files failed to process:")
                    for fname, lev in failed_files:
                        print(f"- {fname} (leverage {lev})")

                if all_results:
                    pd.DataFrame(all_results).to_csv("ema_adx_backtest_results.csv", index=False)
                    
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc())