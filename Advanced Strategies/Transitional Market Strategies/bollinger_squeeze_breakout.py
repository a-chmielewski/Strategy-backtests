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

class Bollinger_Squeeze_Breakout_Strategy(bt.Strategy):
    params = (
        ("bb_period", 20),
        ("bb_std", 2.0),
        ("bb_squeeze_threshold", 0.015),  # BB width must be below 1.5% for squeeze detection
        ("bb_squeeze_bars", 8),  # Minimum bars in squeeze state
        ("adx_period", 14),
        ("adx_use_filter", True),  # Whether to use ADX as filter
        ("adx_low_threshold", 25),  # ADX below 25 indicates consolidation
        ("atr_period", 14),
        ("volume_avg_period", 20),
        ("volume_breakout_multiplier", 1.5),  # 1.5x volume spike for confirmation
        ("entry_mode", "conservative"),  # "aggressive" or "conservative"
        ("confirmation_bars", 1),  # Bars to wait for confirmation in conservative mode
        ("retest_enabled", False),  # Enable retest entry logic
        ("retest_tolerance_pct", 0.003),  # 0.3% tolerance for retest
        ("breakout_min_pct", 0.002),  # Minimum 0.2% breakout to consider valid
        ("stop_loss_pct", 0.01),  # 1% stop loss
        ("first_target_pct", 0.005),  # 0.5% first target
        ("range_projection_multiplier", 1.0),  # 1x range height projection
        ("atr_target_multiplier", 2.0),  # 2x ATR target for runner
        ("trail_stop_period", 5),  # EMA period for trailing stop
        ("trail_stop_buffer_pct", 0.001),  # 0.1% buffer below trailing EMA
        ("partial_exit_pct", 0.5),  # Take 50% profit at first target
        ("max_hold_bars", 50),  # Maximum bars to hold position
        ("fake_breakout_exit_bars", 3),  # Quick exit if reversal within 3 bars
        ("head_fake_protection", True),  # Enable head-fake protection
        ("position_size_factor", 0.9),  # Use 90% normal size initially
        ("pyramid_enabled", False),  # Enable pyramiding on trend confirmation
        ("squeeze_lookback_bars", 50),  # Bars to look back for squeeze history
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
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period) if self.p.adx_use_filter else None
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        self.trail_ema = bt.indicators.EMA(self.data.close, period=self.p.trail_stop_period)
        
        # BB width calculation
        self.bb_width = (self.bb.lines.top - self.bb.lines.bot) / self.bb.lines.mid
        
        # Squeeze detection state
        self.squeeze_detected = False
        self.squeeze_start_bar = None
        self.squeeze_bars_count = 0
        self.squeeze_low = 0
        self.squeeze_high = 0
        self.squeeze_range = 0
        
        # Breakout state
        self.breakout_detected = False
        self.breakout_direction = None
        self.breakout_price = 0
        self.breakout_bar = None
        self.waiting_for_confirmation = False
        self.waiting_for_retest = False
        self.retest_triggered = False
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.first_target_hit = False
        self.trail_stop_active = False
        self.head_fake_detected = False
        
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

            # Adjust position size based on strategy settings
            position_value *= self.p.position_size_factor

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def is_bollinger_squeeze(self):
        """Detect Bollinger Band squeeze conditions"""
        if len(self.bb_width) == 0:
            return False
            
        current_bb_width = self.bb_width[0]
        
        # Check if current BB width is below threshold
        bb_squeeze = current_bb_width < self.p.bb_squeeze_threshold
        
        # Optional ADX filter for consolidation
        adx_ok = True
        if self.p.adx_use_filter and self.adx is not None and len(self.adx) > 0:
            adx_ok = self.adx[0] < self.p.adx_low_threshold
        
        return bb_squeeze and adx_ok

    def update_squeeze_range(self):
        """Update squeeze range during consolidation"""
        if not self.squeeze_detected:
            return
            
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        if self.squeeze_bars_count == 1:
            # Initialize range on first squeeze bar
            self.squeeze_high = current_high
            self.squeeze_low = current_low
        else:
            # Update range during squeeze
            self.squeeze_high = max(self.squeeze_high, current_high)
            self.squeeze_low = min(self.squeeze_low, current_low)
        
        self.squeeze_range = self.squeeze_high - self.squeeze_low

    def detect_breakout(self):
        """Detect breakout from Bollinger Band squeeze"""
        if not self.squeeze_detected or self.squeeze_bars_count < self.p.bb_squeeze_bars:
            return False, None
            
        current_price = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        bb_upper = self.bb.lines.top[0]
        bb_lower = self.bb.lines.bot[0]
        
        # Check for breakout above upper band
        if (current_high > bb_upper and 
            current_price > self.squeeze_high * (1 + self.p.breakout_min_pct)):
            return True, 'long'
        
        # Check for breakout below lower band
        elif (current_low < bb_lower and 
              current_price < self.squeeze_low * (1 - self.p.breakout_min_pct)):
            return True, 'short'
        
        return False, None

    def confirm_breakout(self, direction):
        """Confirm breakout with volume and momentum"""
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        # Volume confirmation
        volume_spike = current_volume >= (avg_volume * self.p.volume_breakout_multiplier)
        
        # Momentum confirmation (strong candle)
        current_open = self.data.open[0]
        current_close = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        candle_range = current_high - current_low
        candle_body = abs(current_close - current_open)
        
        # Strong momentum candle (body > 60% of range)
        momentum_ok = candle_body / candle_range > 0.6 if candle_range > 0 else False
        
        if direction == 'long':
            directional_ok = current_close > current_open  # Green candle
        else:
            directional_ok = current_close < current_open  # Red candle
        
        return volume_spike and momentum_ok and directional_ok

    def check_retest_opportunity(self, direction):
        """Check for retest entry opportunity"""
        if not self.p.retest_enabled or not self.waiting_for_retest:
            return False
            
        current_price = self.data.close[0]
        bb_upper = self.bb.lines.top[0]
        bb_lower = self.bb.lines.bot[0]
        
        if direction == 'long':
            # For long, check if price retests upper band from above
            retest_level = bb_upper
            return (abs(current_price - retest_level) / retest_level < self.p.retest_tolerance_pct and
                    current_price > retest_level)
        else:
            # For short, check if price retests lower band from below
            retest_level = bb_lower
            return (abs(current_price - retest_level) / retest_level < self.p.retest_tolerance_pct and
                    current_price < retest_level)

    def detect_head_fake(self):
        """Detect potential head-fake pattern"""
        if not self.p.head_fake_protection or not self.breakout_detected:
            return False
            
        if len(self) - self.breakout_bar > 5:  # Check within 5 bars of breakout
            return False
            
        current_price = self.data.close[0]
        bb_upper = self.bb.lines.top[0]
        bb_lower = self.bb.lines.bot[0]
        
        # Check if price has reversed back into the bands
        if self.breakout_direction == 'long':
            return current_price < bb_upper
        else:
            return current_price > bb_lower

    def calculate_targets_and_stops(self, direction, entry_price):
        """Calculate profit targets and stop loss levels"""
        if direction == 'long':
            # Stop loss below squeeze range
            stop_price = self.squeeze_low * (1 - self.p.stop_loss_pct)
            
            # First target
            first_target = entry_price * (1 + self.p.first_target_pct)
            
            # Range projection target
            range_target = entry_price + (self.squeeze_range * self.p.range_projection_multiplier)
            
            # ATR target for runner
            atr_target = entry_price + (self.atr[0] * self.p.atr_target_multiplier)
            
            # Use closer target for main profit
            main_target = min(range_target, atr_target)
            
        else:  # short
            # Stop loss above squeeze range
            stop_price = self.squeeze_high * (1 + self.p.stop_loss_pct)
            
            # First target
            first_target = entry_price * (1 - self.p.first_target_pct)
            
            # Range projection target
            range_target = entry_price - (self.squeeze_range * self.p.range_projection_multiplier)
            
            # ATR target for runner
            atr_target = entry_price - (self.atr[0] * self.p.atr_target_multiplier)
            
            # Use closer target for main profit
            main_target = max(range_target, atr_target)
        
        return stop_price, first_target, main_target

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
        
        # Head-fake protection
        if self.detect_head_fake():
            self.head_fake_detected = True
            return True, "head_fake_exit"
        
        # Quick fake breakout exit
        if (bars_held <= self.p.fake_breakout_exit_bars and 
            self.squeeze_low < current_price < self.squeeze_high):
            return True, "fake_breakout"
        
        # Calculate current profit/loss
        if self.entry_price:
            if pos.size > 0:  # Long position
                profit_pct = (current_price - self.entry_price) / self.entry_price
                
                # First target
                if profit_pct >= self.p.first_target_pct and not self.first_target_hit:
                    self.first_target_hit = True
                    return True, "first_target_long"
                
                # Trailing stop logic after first target
                if self.first_target_hit and self.trail_stop_active:
                    trail_level = self.trail_ema[0] * (1 - self.p.trail_stop_buffer_pct)
                    if current_price < trail_level:
                        return True, "trail_stop_long"
                
                # Range/ATR target
                range_profit = (self.squeeze_range * self.p.range_projection_multiplier) / self.entry_price
                atr_profit = (self.atr[0] * self.p.atr_target_multiplier) / self.entry_price
                target_profit = min(range_profit, atr_profit)
                
                if profit_pct >= target_profit:
                    return True, "main_target_long"
                    
            elif pos.size < 0:  # Short position
                profit_pct = (self.entry_price - current_price) / self.entry_price
                
                # First target
                if profit_pct >= self.p.first_target_pct and not self.first_target_hit:
                    self.first_target_hit = True
                    return True, "first_target_short"
                
                # Trailing stop logic after first target
                if self.first_target_hit and self.trail_stop_active:
                    trail_level = self.trail_ema[0] * (1 + self.p.trail_stop_buffer_pct)
                    if current_price > trail_level:
                        return True, "trail_stop_short"
                
                # Range/ATR target
                range_profit = (self.squeeze_range * self.p.range_projection_multiplier) / self.entry_price
                atr_profit = (self.atr[0] * self.p.atr_target_multiplier) / self.entry_price
                target_profit = min(range_profit, atr_profit)
                
                if profit_pct >= target_profit:
                    return True, "main_target_short"
        
        return False, None

    def reset_squeeze_state(self):
        """Reset squeeze and breakout state variables"""
        self.squeeze_detected = False
        self.squeeze_start_bar = None
        self.squeeze_bars_count = 0
        self.breakout_detected = False
        self.breakout_direction = None
        self.waiting_for_confirmation = False
        self.waiting_for_retest = False
        self.retest_triggered = False
        self.head_fake_detected = False

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.bb_period, self.p.atr_period, self.p.volume_avg_period, 
                      self.p.trail_stop_period)
        if self.p.adx_use_filter and self.adx is not None:
            min_bars = max(min_bars, self.p.adx_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        if not self.position:  # If no position is open
            # Step 1: Detect Bollinger Band squeeze
            if self.is_bollinger_squeeze():
                if not self.squeeze_detected:
                    self.squeeze_detected = True
                    self.squeeze_start_bar = len(self)
                    self.squeeze_bars_count = 1
                else:
                    self.squeeze_bars_count += 1
                
                # Update squeeze range
                self.update_squeeze_range()
                    
            else:
                # Check for breakout when squeeze ends
                if self.squeeze_detected and self.squeeze_bars_count >= self.p.bb_squeeze_bars:
                    breakout_occurred, direction = self.detect_breakout()
                    
                    if breakout_occurred:
                        self.breakout_detected = True
                        self.breakout_direction = direction
                        self.breakout_price = current_price
                        self.breakout_bar = len(self)
                        
                        # Entry logic based on mode
                        if self.p.entry_mode == "aggressive":
                            # Aggressive entry: Enter immediately on breakout
                            if self.confirm_breakout(direction):
                                self.enter_position(direction, current_price)
                            else:
                                # Wait for retest if confirmation fails
                                if self.p.retest_enabled:
                                    self.waiting_for_retest = True
                                else:
                                    self.reset_squeeze_state()
                                    
                        elif self.p.entry_mode == "conservative":
                            # Conservative entry: Wait for confirmation
                            if self.confirm_breakout(direction):
                                self.waiting_for_confirmation = True
                            else:
                                self.reset_squeeze_state()
                    else:
                        # No breakout, reset squeeze state
                        self.reset_squeeze_state()
                else:
                    # No squeeze or insufficient bars, reset state
                    if self.squeeze_detected:
                        self.reset_squeeze_state()
            
            # Handle confirmation waiting (conservative mode)
            if self.waiting_for_confirmation and self.breakout_detected:
                bars_since_breakout = len(self) - self.breakout_bar
                if bars_since_breakout >= self.p.confirmation_bars:
                    # Check if trend continues in breakout direction
                    if self.breakout_direction == 'long' and current_price > self.breakout_price:
                        self.enter_position(self.breakout_direction, current_price)
                    elif self.breakout_direction == 'short' and current_price < self.breakout_price:
                        self.enter_position(self.breakout_direction, current_price)
                    else:
                        # Confirmation failed, check for retest
                        if self.p.retest_enabled:
                            self.waiting_for_retest = True
                            self.waiting_for_confirmation = False
                        else:
                            self.reset_squeeze_state()
            
            # Handle retest waiting
            if self.waiting_for_retest and self.breakout_detected:
                if self.check_retest_opportunity(self.breakout_direction):
                    self.enter_position(self.breakout_direction, current_price)
                elif len(self) - self.breakout_bar > 10:  # Timeout retest waiting
                    self.reset_squeeze_state()
                    
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
                    self.reset_squeeze_state()

    def enter_position(self, direction, entry_price):
        """Enter position with proper risk management"""
        position_size = self.calculate_position_size(entry_price)
        if position_size <= 0:
            return
        
        stop_price, first_target, main_target = self.calculate_targets_and_stops(direction, entry_price)
        
        if direction == 'long':
            parent, stop, limit = self.buy_bracket(
                size=position_size,
                exectype=bt.Order.Market,
                stopprice=stop_price,
                limitprice=first_target,
            )
        else:
            parent, stop, limit = self.sell_bracket(
                size=position_size,
                exectype=bt.Order.Market,
                stopprice=stop_price,
                limitprice=first_target,
            )
        
        self.order = parent
        self.order_parent_ref = parent.ref
        self.entry_bar = len(self)
        self.entry_side = direction
        self.entry_price = entry_price
        self.first_target_hit = False
        self.trail_stop_active = False
        
        # Reset states after entry
        self.waiting_for_confirmation = False
        self.waiting_for_retest = False
        self.reset_squeeze_state()

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
        "bb_squeeze_threshold": kwargs.get("bb_squeeze_threshold", 0.015),
        "bb_squeeze_bars": kwargs.get("bb_squeeze_bars", 8),
        "adx_period": kwargs.get("adx_period", 14),
        "adx_use_filter": kwargs.get("adx_use_filter", True),
        "adx_low_threshold": kwargs.get("adx_low_threshold", 25),
        "atr_period": kwargs.get("atr_period", 14),
        "volume_avg_period": kwargs.get("volume_avg_period", 20),
        "volume_breakout_multiplier": kwargs.get("volume_breakout_multiplier", 1.5),
        "entry_mode": kwargs.get("entry_mode", "conservative"),
        "confirmation_bars": kwargs.get("confirmation_bars", 1),
        "retest_enabled": kwargs.get("retest_enabled", False),
        "retest_tolerance_pct": kwargs.get("retest_tolerance_pct", 0.003),
        "breakout_min_pct": kwargs.get("breakout_min_pct", 0.002),
        "stop_loss_pct": kwargs.get("stop_loss_pct", 0.01),
        "first_target_pct": kwargs.get("first_target_pct", 0.005),
        "range_projection_multiplier": kwargs.get("range_projection_multiplier", 1.0),
        "atr_target_multiplier": kwargs.get("atr_target_multiplier", 2.0),
        "trail_stop_period": kwargs.get("trail_stop_period", 5),
        "trail_stop_buffer_pct": kwargs.get("trail_stop_buffer_pct", 0.001),
        "partial_exit_pct": kwargs.get("partial_exit_pct", 0.5),
        "max_hold_bars": kwargs.get("max_hold_bars", 50),
        "fake_breakout_exit_bars": kwargs.get("fake_breakout_exit_bars", 3),
        "head_fake_protection": kwargs.get("head_fake_protection", True),
        "position_size_factor": kwargs.get("position_size_factor", 0.9),
        "pyramid_enabled": kwargs.get("pyramid_enabled", False),
        "squeeze_lookback_bars": kwargs.get("squeeze_lookback_bars", 50),
    }
    cerebro.addstrategy(Bollinger_Squeeze_Breakout_Strategy, **strategy_params)
    
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
        bb_squeeze_threshold=0.015,
        bb_squeeze_bars=8,
        adx_period=14,
        adx_use_filter=True,
        adx_low_threshold=25,
        atr_period=14,
        volume_avg_period=20,
        volume_breakout_multiplier=1.5,
        entry_mode="conservative",
        confirmation_bars=1,
        retest_enabled=False,
        retest_tolerance_pct=0.003,
        breakout_min_pct=0.002,
        stop_loss_pct=0.01,
        first_target_pct=0.005,
        range_projection_multiplier=1.0,
        atr_target_multiplier=2.0,
        trail_stop_period=5,
        trail_stop_buffer_pct=0.001,
        partial_exit_pct=0.5,
        max_hold_bars=50,
        fake_breakout_exit_bars=3,
        head_fake_protection=True,
        position_size_factor=0.9,
        pyramid_enabled=False,
        squeeze_lookback_bars=50,
        leverage=leverage
    )
    
    log_result(
        strategy="Bollinger_Squeeze_Breakout",
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

        if all_results:
            results_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
            os.makedirs(results_folder, exist_ok=True)
            results_path = os.path.join(results_folder, "Bollinger_Squeeze_Breakout.csv")
            pd.DataFrame(all_results).to_csv(results_path, index=False)

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
                    results_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
                    os.makedirs(results_folder, exist_ok=True)
                    results_path = os.path.join(results_folder, "Bollinger_Squeeze_Breakout.csv")
                    pd.DataFrame(all_results).to_csv(results_path, index=False)
                    
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc()) 
