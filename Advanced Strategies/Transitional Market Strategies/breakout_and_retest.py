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

class Breakout_And_Retest_Strategy(bt.Strategy):
    params = (
        ("sr_lookback_period", 30),  # Bars to look back for S/R levels
        ("sr_min_touches", 3),  # Minimum touches to confirm S/R level
        ("sr_tolerance_pct", 0.002),  # 0.2% tolerance for S/R level detection
        ("breakout_min_pct", 0.003),  # Minimum 0.3% breakout to be valid
        ("volume_breakout_multiplier", 1.5),  # Volume must be 1.5x average for breakout
        ("volume_avg_period", 20),  # Period for volume average
        ("ema_trend_period", 50),  # EMA period for trend bias
        ("use_trend_filter", True),  # Whether to use trend filter
        ("rsi_period", 14),  # RSI period for momentum filter
        ("rsi_pullback_min", 40),  # RSI must stay above 40 in uptrend pullback
        ("rsi_pullback_max", 60),  # RSI must stay below 60 in downtrend pullback
        ("retest_timeout_bars", 15),  # Max bars to wait for retest
        ("retest_tolerance_pct", 0.002),  # 0.2% tolerance for retest level
        ("reversal_confirmation_bars", 2),  # Bars needed for reversal confirmation
        ("engulfing_min_ratio", 1.2),  # Min ratio for engulfing pattern
        ("hammer_ratio", 2.0),  # Body to wick ratio for hammer/doji patterns
        ("stop_loss_buffer_pct", 0.001),  # 0.1% buffer beyond retest level for stop
        ("first_target_pct", 0.005),  # 0.5% first target
        ("measured_move_multiplier", 1.0),  # 1x range height for measured move
        ("partial_exit_pct", 0.5),  # Take 50% profit at first target
        ("trail_stop_period", 10),  # EMA period for trailing stop
        ("trail_stop_buffer_pct", 0.002),  # 0.2% buffer for trailing stop
        ("max_hold_bars", 100),  # Maximum bars to hold position
        ("position_size_factor", 1.0),  # Normal position size for this strategy
        ("min_breakout_volume_decline", 0.7),  # Pullback volume should be <70% of breakout volume
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        self.ema_trend = bt.indicators.EMA(self.data.close, period=self.p.ema_trend_period) if self.p.use_trend_filter else None
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.trail_ema = bt.indicators.EMA(self.data.close, period=self.p.trail_stop_period)
        
        # Support/Resistance tracking
        self.support_levels = []
        self.resistance_levels = []
        self.last_sr_update = 0
        
        # Breakout tracking
        self.breakout_detected = False
        self.breakout_direction = None
        self.breakout_level = 0
        self.breakout_price = 0
        self.breakout_bar = None
        self.breakout_volume = 0
        self.range_height = 0
        
        # Retest tracking
        self.waiting_for_retest = False
        self.retest_level = 0
        self.retest_confirmed = False
        self.reversal_start_bar = None
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.first_target_hit = False
        self.trail_stop_active = False
        
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
            position_value *= self.p.position_size_factor
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def update_support_resistance_levels(self):
        """Update support and resistance levels based on recent price action"""
        if len(self) < self.p.sr_lookback_period or len(self) - self.last_sr_update < 10:
            return
            
        self.last_sr_update = len(self)
        
        # Get recent highs and lows
        recent_highs = []
        recent_lows = []
        
        for i in range(self.p.sr_lookback_period):
            recent_highs.append(self.data.high[-i])
            recent_lows.append(self.data.low[-i])
        
        # Find resistance levels (cluster of highs)
        self.resistance_levels = self._find_levels(recent_highs, 'resistance')
        
        # Find support levels (cluster of lows)
        self.support_levels = self._find_levels(recent_lows, 'support')

    def _find_levels(self, prices, level_type):
        """Find support or resistance levels from price clusters"""
        levels = []
        price_clusters = {}
        
        for price in prices:
            found_cluster = False
            for cluster_price in price_clusters:
                if abs(price - cluster_price) / cluster_price < self.p.sr_tolerance_pct:
                    price_clusters[cluster_price] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                price_clusters[price] = 1
        
        # Filter levels with minimum touches
        for price, touches in price_clusters.items():
            if touches >= self.p.sr_min_touches:
                levels.append({'level': price, 'touches': touches, 'strength': touches})
        
        # Sort by strength (most touches first)
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return levels[:5]  # Keep top 5 levels

    def detect_breakout(self):
        """Detect breakout of key support/resistance levels"""
        if not (self.support_levels or self.resistance_levels):
            return False, None, None
            
        current_price = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        # Check volume confirmation
        if current_volume < avg_volume * self.p.volume_breakout_multiplier:
            return False, None, None
        
        # Check resistance breakouts (bullish)
        for resistance in self.resistance_levels:
            level = resistance['level']
            if (current_high > level * (1 + self.p.breakout_min_pct) and 
                current_price > level):
                
                # Calculate range height for measured moves
                range_height = self._calculate_range_height(level, 'resistance')
                return True, 'long', {'level': level, 'range_height': range_height}
        
        # Check support breakdowns (bearish)
        for support in self.support_levels:
            level = support['level']
            if (current_low < level * (1 - self.p.breakout_min_pct) and 
                current_price < level):
                
                # Calculate range height for measured moves
                range_height = self._calculate_range_height(level, 'support')
                return True, 'short', {'level': level, 'range_height': range_height}
        
        return False, None, None

    def _calculate_range_height(self, broken_level, level_type):
        """Calculate the height of the range for measured move targets"""
        if level_type == 'resistance':
            # Find nearest support below
            nearest_support = 0
            for support in self.support_levels:
                if support['level'] < broken_level:
                    nearest_support = max(nearest_support, support['level'])
            return broken_level - nearest_support if nearest_support > 0 else broken_level * 0.02
        else:
            # Find nearest resistance above
            nearest_resistance = float('inf')
            for resistance in self.resistance_levels:
                if resistance['level'] > broken_level:
                    nearest_resistance = min(nearest_resistance, resistance['level'])
            return nearest_resistance - broken_level if nearest_resistance < float('inf') else broken_level * 0.02

    def check_trend_bias(self, direction):
        """Check if breakout aligns with trend bias"""
        if not self.p.use_trend_filter or self.ema_trend is None:
            return True
            
        current_price = self.data.close[0]
        ema_value = self.ema_trend[0]
        
        if direction == 'long':
            return current_price > ema_value  # Price above EMA for bullish bias
        else:
            return current_price < ema_value  # Price below EMA for bearish bias

    def detect_retest(self):
        """Detect retest of the broken level"""
        if not self.waiting_for_retest:
            return False
            
        current_price = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        # Check if price has returned to retest level
        if self.breakout_direction == 'long':
            # For bullish breakout, retest from above
            retest_occurring = (current_low <= self.retest_level * (1 + self.p.retest_tolerance_pct) and
                              current_price >= self.retest_level * (1 - self.p.retest_tolerance_pct))
        else:
            # For bearish breakout, retest from below
            retest_occurring = (current_high >= self.retest_level * (1 - self.p.retest_tolerance_pct) and
                              current_price <= self.retest_level * (1 + self.p.retest_tolerance_pct))
        
        if retest_occurring:
            # Check volume characteristics during pullback
            current_volume = self.data.volume[0]
            volume_ok = current_volume < self.breakout_volume * self.p.min_breakout_volume_decline
            
            # Check RSI during pullback
            rsi_ok = self._check_rsi_pullback()
            
            return volume_ok and rsi_ok
        
        return False

    def _check_rsi_pullback(self):
        """Check RSI behavior during pullback"""
        if len(self.rsi) == 0:
            return True
            
        current_rsi = self.rsi[0]
        
        if self.breakout_direction == 'long':
            # In bullish retest, RSI should stay above 40
            return current_rsi > self.p.rsi_pullback_min
        else:
            # In bearish retest, RSI should stay below 60
            return current_rsi < self.p.rsi_pullback_max

    def detect_reversal_confirmation(self):
        """Detect reversal patterns at retest level"""
        if not self.retest_confirmed:
            return False
            
        # Check for engulfing pattern
        if self._is_engulfing_pattern():
            return True
            
        # Check for hammer/doji pattern
        if self._is_hammer_or_doji():
            return True
            
        # Check for multiple bar confirmation
        if self._is_multi_bar_reversal():
            return True
        
        return False

    def _is_engulfing_pattern(self):
        """Check for engulfing candlestick pattern"""
        if len(self) < 2:
            return False
            
        prev_open = self.data.open[-1]
        prev_close = self.data.close[-1]
        curr_open = self.data.open[0]
        curr_close = self.data.close[0]
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        
        if curr_body < prev_body * self.p.engulfing_min_ratio:
            return False
        
        if self.breakout_direction == 'long':
            # Bullish engulfing: prev red, curr green, curr engulfs prev
            return (prev_close < prev_open and curr_close > curr_open and
                    curr_open < prev_close and curr_close > prev_open)
        else:
            # Bearish engulfing: prev green, curr red, curr engulfs prev
            return (prev_close > prev_open and curr_close < curr_open and
                    curr_open > prev_close and curr_close < prev_open)

    def _is_hammer_or_doji(self):
        """Check for hammer or doji patterns"""
        curr_open = self.data.open[0]
        curr_close = self.data.close[0]
        curr_high = self.data.high[0]
        curr_low = self.data.low[0]
        
        body_size = abs(curr_close - curr_open)
        total_range = curr_high - curr_low
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        
        if self.breakout_direction == 'long':
            # Hammer: small body, long lower wick
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)
            return (body_ratio < 0.3 and lower_wick > body_size * self.p.hammer_ratio and
                    upper_wick < body_size)
        else:
            # Inverted hammer: small body, long upper wick
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)
            return (body_ratio < 0.3 and upper_wick > body_size * self.p.hammer_ratio and
                    lower_wick < body_size)

    def _is_multi_bar_reversal(self):
        """Check for multiple bar reversal confirmation"""
        if self.reversal_start_bar is None:
            return False
            
        bars_since_reversal = len(self) - self.reversal_start_bar
        
        if bars_since_reversal >= self.p.reversal_confirmation_bars:
            current_price = self.data.close[0]
            
            if self.breakout_direction == 'long':
                # Price should be moving up from retest level
                return current_price > self.retest_level
            else:
                # Price should be moving down from retest level
                return current_price < self.retest_level
        
        return False

    def calculate_targets_and_stops(self, direction, entry_price):
        """Calculate profit targets and stop loss levels"""
        if direction == 'long':
            # Stop loss below retest level
            stop_price = self.retest_level * (1 - self.p.stop_loss_buffer_pct)
            
            # First target
            first_target = entry_price * (1 + self.p.first_target_pct)
            
            # Measured move target
            measured_target = entry_price + (self.range_height * self.p.measured_move_multiplier)
            
            # Use closer target for bracket order
            main_target = min(first_target, measured_target)
            
        else:  # short
            # Stop loss above retest level
            stop_price = self.retest_level * (1 + self.p.stop_loss_buffer_pct)
            
            # First target
            first_target = entry_price * (1 - self.p.first_target_pct)
            
            # Measured move target
            measured_target = entry_price - (self.range_height * self.p.measured_move_multiplier)
            
            # Use closer target for bracket order
            main_target = max(first_target, measured_target)
        
        return stop_price, main_target

    def check_exit_conditions(self):
        """Check for exit conditions"""
        if not self.position:
            return False, None
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Time-based exit
        if bars_held >= self.p.max_hold_bars:
            return True, "time_exit"
        
        # Calculate current profit/loss
        if self.entry_price:
            if pos.size > 0:  # Long position
                profit_pct = (current_price - self.entry_price) / self.entry_price
                
                # First target
                if profit_pct >= self.p.first_target_pct and not self.first_target_hit:
                    self.first_target_hit = True
                    return True, "first_target_long"
                
                # Trailing stop after first target
                if self.first_target_hit and self.trail_stop_active:
                    trail_level = self.trail_ema[0] * (1 - self.p.trail_stop_buffer_pct)
                    if current_price < trail_level:
                        return True, "trail_stop_long"
                
                # Measured move target
                measured_profit = (self.range_height * self.p.measured_move_multiplier) / self.entry_price
                if profit_pct >= measured_profit:
                    return True, "measured_target_long"
                    
            elif pos.size < 0:  # Short position
                profit_pct = (self.entry_price - current_price) / self.entry_price
                
                # First target
                if profit_pct >= self.p.first_target_pct and not self.first_target_hit:
                    self.first_target_hit = True
                    return True, "first_target_short"
                
                # Trailing stop after first target
                if self.first_target_hit and self.trail_stop_active:
                    trail_level = self.trail_ema[0] * (1 + self.p.trail_stop_buffer_pct)
                    if current_price > trail_level:
                        return True, "trail_stop_short"
                
                # Measured move target
                measured_profit = (self.range_height * self.p.measured_move_multiplier) / self.entry_price
                if profit_pct >= measured_profit:
                    return True, "measured_target_short"
        
        return False, None

    def reset_breakout_state(self):
        """Reset breakout and retest state variables"""
        self.breakout_detected = False
        self.breakout_direction = None
        self.waiting_for_retest = False
        self.retest_confirmed = False
        self.reversal_start_bar = None

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.sr_lookback_period, self.p.volume_avg_period, 
                      self.p.rsi_period, self.p.trail_stop_period)
        if self.p.use_trend_filter and self.ema_trend is not None:
            min_bars = max(min_bars, self.p.ema_trend_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Update support/resistance levels
        self.update_support_resistance_levels()
        
        if not self.position:  # If no position is open
            
            if not self.waiting_for_retest:
                # Step 1: Look for breakouts
                breakout_occurred, direction, breakout_info = self.detect_breakout()
                
                if breakout_occurred and self.check_trend_bias(direction):
                    self.breakout_detected = True
                    self.breakout_direction = direction
                    self.breakout_level = breakout_info['level']
                    self.breakout_price = current_price
                    self.breakout_bar = len(self)
                    self.breakout_volume = self.data.volume[0]
                    self.range_height = breakout_info['range_height']
                    
                    # Set up retest waiting
                    self.waiting_for_retest = True
                    self.retest_level = self.breakout_level
                    
            else:
                # Step 2: Check for retest
                if self.detect_retest():
                    self.retest_confirmed = True
                    self.reversal_start_bar = len(self)
                    
                # Step 3: Look for reversal confirmation
                if self.retest_confirmed and self.detect_reversal_confirmation():
                    # Enter position on confirmed reversal
                    position_size = self.calculate_position_size(current_price)
                    if position_size > 0:
                        self.enter_position(self.breakout_direction, current_price)
                
                # Timeout retest waiting
                if (self.breakout_bar is not None and 
                    len(self) - self.breakout_bar > self.p.retest_timeout_bars):
                    self.reset_breakout_state()
                    
        else:
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions()
            
            if should_exit:
                if exit_reason in ["first_target_long", "first_target_short"]:
                    # Take partial profits and activate trailing stop
                    pos = self.getposition()
                    partial_size = abs(pos.size) * self.p.partial_exit_pct
                    
                    if pos.size > 0:
                        self.sell(size=partial_size)
                    else:
                        self.buy(size=partial_size)
                    
                    # Activate trailing stop for remaining position
                    self.trail_stop_active = True
                    
                else:
                    # Exit completely
                    self.close()
                    self.closing = True
                    self.entry_bar = None
                    self.entry_side = None
                    self.entry_price = None
                    self.first_target_hit = False
                    self.trail_stop_active = False
                    self.reset_breakout_state()

    def enter_position(self, direction, entry_price):
        """Enter position with proper risk management"""
        position_size = self.calculate_position_size(entry_price)
        if position_size <= 0:
            return
        
        stop_price, target_price = self.calculate_targets_and_stops(direction, entry_price)
        
        if direction == 'long':
            parent, stop, limit = self.buy_bracket(
                size=position_size,
                exectype=bt.Order.Market,
                stopprice=stop_price,
                limitprice=target_price,
            )
        else:
            parent, stop, limit = self.sell_bracket(
                size=position_size,
                exectype=bt.Order.Market,
                stopprice=stop_price,
                limitprice=target_price,
            )
        
        self.order = parent
        self.order_parent_ref = parent.ref
        self.entry_bar = len(self)
        self.entry_side = direction
        self.entry_price = entry_price
        self.first_target_hit = False
        self.trail_stop_active = False
        
        # Reset states after entry
        self.reset_breakout_state()

    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            self.trade_dir[trade.ref] = 'long' if trade.size > 0 else 'short'
        if not trade.isclosed:
            return

        try:
            entry_price = trade.price
            exit_price = trade.history[-1].price if trade.history else self.data.close[0]
            pnl = trade.pnl
            
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

    def notify_order(self, order):
        if self.order and order.ref == self.order.ref and order.parent is None and (
            (self.entry_side == 'long' and order.isbuy()) or (self.entry_side == 'short' and order.issell())
        ) and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None
            if hasattr(self, 'order_parent_ref') and order.ref == self.order_parent_ref:
                self.order_parent_ref = None
        
        if getattr(self, 'closing', False) and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.closing = False
            
        if order.status == order.Completed:
            if not order.parent:
                self.active_trades.append({
                    'entry_time': self.data.datetime.datetime(0),
                    'entry_price': order.executed.price,
                    'type': 'long' if order.isbuy() else 'short',
                    'size': order.executed.size
                })
            else:
                if self.active_trades:
                    trade = self.active_trades.pop()
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
    
    # Extract strategy parameters properly
    strategy_params = {
        "sr_lookback_period": kwargs.get("sr_lookback_period", 30),
        "sr_min_touches": kwargs.get("sr_min_touches", 3),
        "sr_tolerance_pct": kwargs.get("sr_tolerance_pct", 0.002),
        "breakout_min_pct": kwargs.get("breakout_min_pct", 0.003),
        "volume_breakout_multiplier": kwargs.get("volume_breakout_multiplier", 1.5),
        "volume_avg_period": kwargs.get("volume_avg_period", 20),
        "ema_trend_period": kwargs.get("ema_trend_period", 50),
        "use_trend_filter": kwargs.get("use_trend_filter", True),
        "rsi_period": kwargs.get("rsi_period", 14),
        "rsi_pullback_min": kwargs.get("rsi_pullback_min", 40),
        "rsi_pullback_max": kwargs.get("rsi_pullback_max", 60),
        "retest_timeout_bars": kwargs.get("retest_timeout_bars", 15),
        "retest_tolerance_pct": kwargs.get("retest_tolerance_pct", 0.002),
        "reversal_confirmation_bars": kwargs.get("reversal_confirmation_bars", 2),
        "engulfing_min_ratio": kwargs.get("engulfing_min_ratio", 1.2),
        "hammer_ratio": kwargs.get("hammer_ratio", 2.0),
        "stop_loss_buffer_pct": kwargs.get("stop_loss_buffer_pct", 0.001),
        "first_target_pct": kwargs.get("first_target_pct", 0.005),
        "measured_move_multiplier": kwargs.get("measured_move_multiplier", 1.0),
        "partial_exit_pct": kwargs.get("partial_exit_pct", 0.5),
        "trail_stop_period": kwargs.get("trail_stop_period", 10),
        "trail_stop_buffer_pct": kwargs.get("trail_stop_buffer_pct", 0.002),
        "max_hold_bars": kwargs.get("max_hold_bars", 100),
        "position_size_factor": kwargs.get("position_size_factor", 1.0),
        "min_breakout_volume_decline": kwargs.get("min_breakout_volume_decline", 0.7),
    }
    
    cerebro.addstrategy(Breakout_And_Retest_Strategy, **strategy_params)
    
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
    strat = results[0][0] if isinstance(results[0], (list, tuple)) else results[0]

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
    
    results = run_backtest(data_df, verbose=False, leverage=leverage)
    
    log_result(
        strategy="Breakout_And_Retest",
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

            leverage_results = [r for r in all_results if r['leverage'] == leverage]
            sorted_results = sorted(leverage_results, key=lambda x: x['winrate'], reverse=True)[:3]
            
            print(f"\n=== Top 3 Results by Win Rate for LEVERAGE {leverage} ===")
            for i, result in enumerate(sorted_results, 1):
                print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
                print(f"Win Rate: {result['winrate']:.2f}%")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Final Equity: {result['final_equity']}")
                print(f"Max Drawdown: {result['max_drawdown']:.2f}%")

        if all_results:
            results_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
            os.makedirs(results_folder, exist_ok=True)
            results_path = os.path.join(results_folder, "Breakout_And_Retest.csv")
            pd.DataFrame(all_results).to_csv(results_path, index=False)

    except Exception as e:
        print(f"\nException occurred: {str(e)}")
        if all_results:
            results_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
            os.makedirs(results_folder, exist_ok=True)
            results_path = os.path.join(results_folder, "Breakout_And_Retest.csv")
            pd.DataFrame(all_results).to_csv(results_path, index=False)
