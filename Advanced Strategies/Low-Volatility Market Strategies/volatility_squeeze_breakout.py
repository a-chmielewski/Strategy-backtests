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

class Volatility_Squeeze_Breakout_Strategy(bt.Strategy):
    params = (
        ("bb_period", 20),
        ("bb_std", 2.0),
        ("bb_squeeze_threshold", 0.01),  # BB width must be below 1% for squeeze detection
        ("bb_squeeze_bars", 10),  # Minimum bars in squeeze state
        ("adx_period", 14),
        ("adx_low_threshold", 20),  # ADX below 20 indicates low volatility
        ("atr_period", 14),
        ("volume_avg_period", 20),
        ("volume_low_threshold", 0.7),  # Volume below 70% average during squeeze
        ("volume_breakout_multiplier", 2.0),  # 2x volume spike for breakout confirmation
        ("range_detection_bars", 30),  # Bars to look back for range detection
        ("range_touch_tolerance", 0.0015),  # 0.15% tolerance for support/resistance
        ("min_range_touches", 3),  # Minimum touches of S/R levels
        ("breakout_buffer_pct", 0.002),  # 0.2% beyond range for breakout orders
        ("confirmation_required", True),  # Require volume + candle confirmation
        ("candle_close_pct", 0.7),  # Candle must close 70% through its range
        ("initial_stop_buffer_pct", 0.001),  # 0.1% stop inside range
        ("quick_profit_pct", 0.005),  # 0.5% quick profit target
        ("range_projection_multiplier", 1.0),  # 1x range height projection
        ("atr_target_multiplier", 2.0),  # 2x ATR target
        ("trail_stop_pct", 0.003),  # 0.3% trailing stop
        ("partial_exit_pct", 0.5),  # Take 50% profit at first target
        ("max_hold_bars", 30),  # Maximum bars to hold position
        ("fake_breakout_exit_bars", 3),  # Exit if no follow-through in 3 bars
        ("position_size_reduction", 0.8),  # Use 80% normal size for breakout trades
        ("max_squeeze_bars", 100),  # Maximum bars to wait in squeeze
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
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        
        # Squeeze detection state
        self.squeeze_detected = False
        self.squeeze_start_bar = None
        self.squeeze_bars_count = 0
        
        # Range detection variables
        self.range_support = 0
        self.range_resistance = 0
        self.range_middle = 0
        self.range_height = 0
        self.range_confirmed = False
        
        # Breakout management
        self.pending_breakout_long = False
        self.pending_breakout_short = False
        self.breakout_long_price = 0
        self.breakout_short_price = 0
        self.breakout_direction = None
        self.breakout_confirmed = False
        self.first_target_hit = False
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.initial_atr = 0
        
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

            # Reduce position size for breakout trades (more conservative)
            position_value *= self.p.position_size_reduction

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def is_bollinger_squeeze(self):
        """Detect Bollinger Band squeeze conditions"""
        if len(self.bb) == 0 or len(self.adx) == 0 or len(self.volume_sma) == 0:
            return False
            
        current_price = self.data.close[0]
        bb_upper = self.bb.lines.top[0]
        bb_lower = self.bb.lines.bot[0]
        adx_value = self.adx[0]
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        # Calculate Bollinger Band width as percentage
        bb_width_pct = (bb_upper - bb_lower) / current_price
        
        # Check squeeze criteria
        bb_squeeze = bb_width_pct < self.p.bb_squeeze_threshold
        adx_low = adx_value < self.p.adx_low_threshold
        volume_low = current_volume < (avg_volume * self.p.volume_low_threshold)
        
        # All criteria must be met for squeeze
        return bb_squeeze and adx_low and volume_low

    def detect_support_resistance_range(self):
        """Detect support and resistance levels during squeeze"""
        if len(self) < self.p.range_detection_bars:
            return False
            
        # Look back at recent price action during squeeze period
        lookback_bars = min(self.p.range_detection_bars, self.squeeze_bars_count)
        if lookback_bars < 10:  # Need minimum data
            return False
            
        highs = []
        lows = []
        
        for i in range(lookback_bars):
            highs.append(self.data.high[-i])
            lows.append(self.data.low[-i])
        
        # Find resistance level (cluster of highs)
        high_clusters = {}
        for high in highs:
            found_cluster = False
            for cluster_level in high_clusters:
                if abs(high - cluster_level) / cluster_level < self.p.range_touch_tolerance:
                    high_clusters[cluster_level] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                high_clusters[high] = 1
        
        # Find support level (cluster of lows)
        low_clusters = {}
        for low in lows:
            found_cluster = False
            for cluster_level in low_clusters:
                if abs(low - cluster_level) / cluster_level < self.p.range_touch_tolerance:
                    low_clusters[cluster_level] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                low_clusters[low] = 1
        
        # Get most significant levels
        resistance_candidates = [(level, count) for level, count in high_clusters.items() 
                               if count >= self.p.min_range_touches]
        support_candidates = [(level, count) for level, count in low_clusters.items() 
                            if count >= self.p.min_range_touches]
        
        if not resistance_candidates or not support_candidates:
            return False
        
        # Choose levels with most touches
        resistance = max(resistance_candidates, key=lambda x: x[1])[0]
        support = min(support_candidates, key=lambda x: x[1])[0]
        
        # Validate range makes sense
        if resistance <= support or (resistance - support) / support < 0.005:  # Minimum 0.5% range
            return False
        
        self.range_resistance = resistance
        self.range_support = support
        self.range_middle = (resistance + support) / 2
        self.range_height = resistance - support
        
        return True

    def setup_breakout_levels(self):
        """Calculate breakout trigger levels"""
        if not self.range_confirmed:
            return False
            
        # Set breakout levels with buffer
        self.breakout_long_price = self.range_resistance * (1 + self.p.breakout_buffer_pct)
        self.breakout_short_price = self.range_support * (1 - self.p.breakout_buffer_pct)
        
        self.pending_breakout_long = True
        self.pending_breakout_short = True
        
        return True

    def check_breakout_trigger(self):
        """Check if price has triggered a breakout"""
        if not (self.pending_breakout_long or self.pending_breakout_short):
            return False, None
            
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        # Check long breakout
        if self.pending_breakout_long and current_high >= self.breakout_long_price:
            return True, 'long'
        
        # Check short breakout
        if self.pending_breakout_short and current_low <= self.breakout_short_price:
            return True, 'short'
        
        return False, None

    def confirm_breakout(self, direction):
        """Confirm breakout with volume and candle analysis"""
        if not self.p.confirmation_required:
            return True
            
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        current_open = self.data.open[0]
        current_close = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        # Volume confirmation
        volume_spike = current_volume >= (avg_volume * self.p.volume_breakout_multiplier)
        
        # Candle confirmation
        candle_range = current_high - current_low
        if candle_range == 0:
            return False
            
        if direction == 'long':
            # For long breakout, candle should close in upper part of its range
            close_position = (current_close - current_low) / candle_range
            candle_confirmation = (close_position >= self.p.candle_close_pct and 
                                 current_close > current_open)
        else:
            # For short breakout, candle should close in lower part of its range
            close_position = (current_high - current_close) / candle_range
            candle_confirmation = (close_position >= self.p.candle_close_pct and 
                                 current_close < current_open)
        
        return volume_spike and candle_confirmation

    def calculate_targets_and_stops(self, direction, entry_price):
        """Calculate profit targets and stop loss levels"""
        # Stop loss just inside the range
        if direction == 'long':
            stop_price = self.range_resistance * (1 - self.p.initial_stop_buffer_pct)
        else:
            stop_price = self.range_support * (1 + self.p.initial_stop_buffer_pct)
        
        # Profit targets
        quick_target = entry_price * (1 + self.p.quick_profit_pct) if direction == 'long' else entry_price * (1 - self.p.quick_profit_pct)
        
        # Range projection target
        range_target = entry_price + self.range_height if direction == 'long' else entry_price - self.range_height
        
        # ATR target
        atr_target = entry_price + (self.initial_atr * self.p.atr_target_multiplier) if direction == 'long' else entry_price - (self.initial_atr * self.p.atr_target_multiplier)
        
        # Use the closer of range or ATR target as main target
        main_target = min(range_target, atr_target) if direction == 'long' else max(range_target, atr_target)
        
        return stop_price, quick_target, main_target

    def check_exit_conditions(self):
        """Check for breakout position exit conditions"""
        if not self.position:
            return False, None
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Time-based exit
        if bars_held >= self.p.max_hold_bars:
            return True, "time_exit"
        
        # Check for fake breakout (quick reversal back into range)
        if (bars_held <= self.p.fake_breakout_exit_bars and 
            self.range_support < current_price < self.range_resistance):
            return True, "fake_breakout"
        
        # Calculate current profit/loss
        if self.entry_price:
            if pos.size > 0:  # Long position
                profit_pct = (current_price - self.entry_price) / self.entry_price
                
                # Quick profit target
                if profit_pct >= self.p.quick_profit_pct and not self.first_target_hit:
                    self.first_target_hit = True
                    return True, "quick_profit_long"
                
                # Range projection or ATR target
                range_profit = self.range_height / self.entry_price
                atr_profit = (self.initial_atr * self.p.atr_target_multiplier) / self.entry_price
                target_profit = min(range_profit, atr_profit)
                
                if profit_pct >= target_profit:
                    return True, "main_target_long"
                
                # Trailing stop after first target
                if self.first_target_hit and profit_pct > 0:
                    trail_level = self.entry_price * (1 + profit_pct - self.p.trail_stop_pct)
                    if current_price < trail_level:
                        return True, "trail_stop_long"
                        
            elif pos.size < 0:  # Short position
                profit_pct = (self.entry_price - current_price) / self.entry_price
                
                # Quick profit target
                if profit_pct >= self.p.quick_profit_pct and not self.first_target_hit:
                    self.first_target_hit = True
                    return True, "quick_profit_short"
                
                # Range projection or ATR target
                range_profit = self.range_height / self.entry_price
                atr_profit = (self.initial_atr * self.p.atr_target_multiplier) / self.entry_price
                target_profit = min(range_profit, atr_profit)
                
                if profit_pct >= target_profit:
                    return True, "main_target_short"
                
                # Trailing stop after first target
                if self.first_target_hit and profit_pct > 0:
                    trail_level = self.entry_price * (1 - profit_pct + self.p.trail_stop_pct)
                    if current_price > trail_level:
                        return True, "trail_stop_short"
        
        return False, None

    def reset_squeeze_state(self):
        """Reset all squeeze and breakout state variables"""
        self.squeeze_detected = False
        self.squeeze_start_bar = None
        self.squeeze_bars_count = 0
        self.range_confirmed = False
        self.pending_breakout_long = False
        self.pending_breakout_short = False
        self.breakout_direction = None
        self.breakout_confirmed = False

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.bb_period, self.p.adx_period, self.p.atr_period, 
                      self.p.volume_avg_period, self.p.range_detection_bars)
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
                    
                # Check if squeeze has lasted long enough and not too long
                if (self.squeeze_bars_count >= self.p.bb_squeeze_bars and 
                    self.squeeze_bars_count <= self.p.max_squeeze_bars):
                    
                    # Step 2: Detect support/resistance range
                    if not self.range_confirmed:
                        if self.detect_support_resistance_range():
                            self.range_confirmed = True
                            self.setup_breakout_levels()
                    
                    # Step 3: Check for breakout triggers
                    if self.range_confirmed:
                        breakout_triggered, direction = self.check_breakout_trigger()
                        
                        if breakout_triggered:
                            # Step 4: Confirm breakout
                            if self.confirm_breakout(direction):
                                position_size = self.calculate_position_size(current_price)
                                if position_size <= 0:
                                    return
                                
                                self.initial_atr = self.atr[0]
                                
                                # Cancel the opposite direction (OCO behavior)
                                if direction == 'long':
                                    self.pending_breakout_short = False
                                    
                                    stop_price, quick_target, main_target = self.calculate_targets_and_stops(
                                        'long', current_price
                                    )
                                    
                                    parent, stop, limit = self.buy_bracket(
                                        size=position_size,
                                        exectype=bt.Order.Market,
                                        stopprice=stop_price,
                                        limitprice=quick_target,
                                    )
                                    self.order = parent
                                    self.order_parent_ref = parent.ref
                                    self.entry_bar = len(self)
                                    self.entry_side = 'long'
                                    self.entry_price = current_price
                                    self.breakout_direction = 'long'
                                    self.first_target_hit = False
                                    
                                elif direction == 'short':
                                    self.pending_breakout_long = False
                                    
                                    stop_price, quick_target, main_target = self.calculate_targets_and_stops(
                                        'short', current_price
                                    )
                                    
                                    parent, stop, limit = self.sell_bracket(
                                        size=position_size,
                                        exectype=bt.Order.Market,
                                        stopprice=stop_price,
                                        limitprice=quick_target,
                                    )
                                    self.order = parent
                                    self.order_parent_ref = parent.ref
                                    self.entry_bar = len(self)
                                    self.entry_side = 'short'
                                    self.entry_price = current_price
                                    self.breakout_direction = 'short'
                                    self.first_target_hit = False
                                
                                # Reset squeeze state after breakout
                                self.reset_squeeze_state()
                            else:
                                # Breakout not confirmed, reset trigger for this direction
                                if direction == 'long':
                                    self.pending_breakout_long = False
                                else:
                                    self.pending_breakout_short = False
                                    
            else:
                # No longer in squeeze, reset state
                if self.squeeze_detected:
                    self.reset_squeeze_state()
                    
        else:
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions()
            
            if should_exit:
                if exit_reason in ["quick_profit_long", "quick_profit_short"]:
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
                    self.breakout_direction = None
                    self.first_target_hit = False

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
        "bb_squeeze_threshold": kwargs.get("bb_squeeze_threshold", 0.01),
        "bb_squeeze_bars": kwargs.get("bb_squeeze_bars", 10),
        "adx_period": kwargs.get("adx_period", 14),
        "adx_low_threshold": kwargs.get("adx_low_threshold", 20),
        "atr_period": kwargs.get("atr_period", 14),
        "volume_avg_period": kwargs.get("volume_avg_period", 20),
        "volume_low_threshold": kwargs.get("volume_low_threshold", 0.7),
        "volume_breakout_multiplier": kwargs.get("volume_breakout_multiplier", 2.0),
        "range_detection_bars": kwargs.get("range_detection_bars", 30),
        "range_touch_tolerance": kwargs.get("range_touch_tolerance", 0.0015),
        "min_range_touches": kwargs.get("min_range_touches", 3),
        "breakout_buffer_pct": kwargs.get("breakout_buffer_pct", 0.002),
        "confirmation_required": kwargs.get("confirmation_required", True),
        "candle_close_pct": kwargs.get("candle_close_pct", 0.7),
        "initial_stop_buffer_pct": kwargs.get("initial_stop_buffer_pct", 0.001),
        "quick_profit_pct": kwargs.get("quick_profit_pct", 0.005),
        "range_projection_multiplier": kwargs.get("range_projection_multiplier", 1.0),
        "atr_target_multiplier": kwargs.get("atr_target_multiplier", 2.0),
        "trail_stop_pct": kwargs.get("trail_stop_pct", 0.003),
        "partial_exit_pct": kwargs.get("partial_exit_pct", 0.5),
        "max_hold_bars": kwargs.get("max_hold_bars", 30),
        "fake_breakout_exit_bars": kwargs.get("fake_breakout_exit_bars", 3),
        "position_size_reduction": kwargs.get("position_size_reduction", 0.8),
        "max_squeeze_bars": kwargs.get("max_squeeze_bars", 100),
    }
    cerebro.addstrategy(Volatility_Squeeze_Breakout_Strategy, **strategy_params)
    
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
        bb_squeeze_threshold=0.01,
        bb_squeeze_bars=10,
        adx_period=14,
        adx_low_threshold=20,
        atr_period=14,
        volume_avg_period=20,
        volume_low_threshold=0.7,
        volume_breakout_multiplier=2.0,
        range_detection_bars=30,
        range_touch_tolerance=0.0015,
        min_range_touches=3,
        breakout_buffer_pct=0.002,
        confirmation_required=True,
        candle_close_pct=0.7,
        initial_stop_buffer_pct=0.001,
        quick_profit_pct=0.005,
        range_projection_multiplier=1.0,
        atr_target_multiplier=2.0,
        trail_stop_pct=0.003,
        partial_exit_pct=0.5,
        max_hold_bars=30,
        fake_breakout_exit_bars=3,
        position_size_reduction=0.8,
        max_squeeze_bars=100,
        leverage=leverage
    )
    
    log_result(
        strategy="Volatility_Squeeze_Breakout",
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
            results_path = os.path.join(results_folder, "Volatility_Squeeze_Breakout.csv")
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
                    results_path = os.path.join(results_folder, "Volatility_Squeeze_Breakout.csv")
                    pd.DataFrame(all_results).to_csv(results_path, index=False)
                    
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc()) 
