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

class Micro_Range_Scalp_Strategy(bt.Strategy):
    params = (
        ("atr_period", 14),
        ("atr_low_threshold", 0.002),  # ATR must be below 0.2% for low-vol detection
        ("bb_period", 20),
        ("bb_std", 2.0),
        ("bb_squeeze_threshold", 0.005),  # BB width must be below 0.5% for squeeze
        ("rsi_period", 7),  # Short RSI for responsiveness in small moves
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("stoch_k_period", 14),
        ("stoch_d_period", 3),
        ("stoch_oversold", 20),
        ("stoch_overbought", 80),
        ("volume_avg_period", 20),
        ("volume_decline_threshold", 0.8),  # Volume must be below 80% of average
        ("range_detection_bars", 20),  # Bars to look back for range detection
        ("min_range_touches", 2),  # Minimum touches of support/resistance
        ("range_tolerance_pct", 0.001),  # 0.1% tolerance for range levels
        ("micro_range_max_pct", 0.003),  # Maximum 0.3% range width for micro-range
        ("stop_loss_buffer_pct", 0.0005),  # 0.05% buffer beyond range for stops
        ("take_profit_pct", 0.0015),  # 0.15% take profit target
        ("max_hold_bars", 10),  # Maximum bars to hold micro-scalp
        ("min_range_bars", 6),  # Minimum bars to confirm a range
        ("position_size_reduction", 0.7),  # Use 70% normal size for micro-scalping
        ("break_even_buffer_pct", 0.0003),  # Move stop to break-even after 0.03% profit
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.p.bb_period, 
            devfactor=self.p.bb_std
        )
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
        self.stoch = bt.indicators.Stochastic(
            self.data, 
            period=self.p.stoch_k_period,
            period_dfast=self.p.stoch_d_period
        )
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        
        # Range detection variables
        self.micro_range_detected = False
        self.range_support = 0
        self.range_resistance = 0
        self.range_middle = 0
        self.range_width = 0
        self.range_touches_support = 0
        self.range_touches_resistance = 0
        self.last_range_update = 0
        
        # Low volatility state
        self.low_vol_confirmed = False
        self.low_vol_bars_count = 0
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.break_even_moved = False
        
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

            # Reduce position size for micro-scalping
            position_value *= self.p.position_size_reduction

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def is_low_volatility_environment(self):
        """Detect low-volatility conditions using multiple criteria"""
        if (len(self.atr) == 0 or len(self.bb) == 0 or 
            len(self.volume_sma) == 0):
            return False
            
        current_price = self.data.close[0]
        atr_value = self.atr[0]
        bb_upper = self.bb.lines.top[0]
        bb_lower = self.bb.lines.bot[0]
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        # Calculate ATR as percentage of price (with safety check)
        atr_pct = atr_value / current_price if current_price > 0 else 0
        
        # Calculate Bollinger Band width as percentage (with safety check)
        bb_width_pct = (bb_upper - bb_lower) / current_price if current_price > 0 else 0
        
        # Check low volatility criteria
        atr_low = atr_pct < self.p.atr_low_threshold
        bb_squeeze = bb_width_pct < self.p.bb_squeeze_threshold
        volume_declining = current_volume < (avg_volume * self.p.volume_decline_threshold)
        
        # All criteria must be met for low-vol confirmation
        return atr_low and bb_squeeze and volume_declining

    def detect_micro_range(self):
        """Detect micro-range with tight support and resistance levels"""
        if len(self) < self.p.range_detection_bars:
            return False
            
        # Look back at recent price action
        lookback_highs = []
        lookback_lows = []
        
        for i in range(self.p.range_detection_bars):
            lookback_highs.append(self.data.high[-i])
            lookback_lows.append(self.data.low[-i])
        
        # Find potential resistance (recent highs cluster)
        high_levels = []
        for i, high in enumerate(lookback_highs):
            touches = 0
            for other_high in lookback_highs:
                if high > 0 and abs(other_high - high) / high < self.p.range_tolerance_pct:
                    touches += 1
            if touches >= self.p.min_range_touches:
                high_levels.append(high)
        
        # Find potential support (recent lows cluster)
        low_levels = []
        for i, low in enumerate(lookback_lows):
            touches = 0
            for other_low in lookback_lows:
                if low > 0 and abs(other_low - low) / low < self.p.range_tolerance_pct:
                    touches += 1
            if touches >= self.p.min_range_touches:
                low_levels.append(low)
        
        if not high_levels or not low_levels:
            return False
        
        # Use the most common levels
        resistance = max(set(high_levels), key=high_levels.count)
        support = min(set(low_levels), key=low_levels.count)
        
        # Check if it's a micro-range (very tight) with safety check
        range_width_pct = (resistance - support) / support if support > 0 else 0
        
        if range_width_pct <= self.p.micro_range_max_pct and resistance > support:
            self.range_resistance = resistance
            self.range_support = support
            self.range_middle = (resistance + support) / 2
            self.range_width = resistance - support
            self.last_range_update = len(self)
            
            # Count actual touches in recent bars
            self.range_touches_support = sum(1 for low in lookback_lows 
                                           if support > 0 and abs(low - support) / support < self.p.range_tolerance_pct)
            self.range_touches_resistance = sum(1 for high in lookback_highs 
                                              if resistance > 0 and abs(high - resistance) / resistance < self.p.range_tolerance_pct)
            
            return True
        
        return False

    def is_near_support(self, price):
        """Check if price is near support level"""
        if not self.micro_range_detected or self.range_support <= 0:
            return False
        return abs(price - self.range_support) / self.range_support < self.p.range_tolerance_pct

    def is_near_resistance(self, price):
        """Check if price is near resistance level"""
        if not self.micro_range_detected or self.range_resistance <= 0:
            return False
        return abs(price - self.range_resistance) / self.range_resistance < self.p.range_tolerance_pct

    def check_oscillator_oversold(self):
        """Check if oscillators indicate oversold conditions"""
        if len(self.rsi) == 0 or len(self.stoch) == 0:
            return False
            
        rsi_oversold = self.rsi[0] < self.p.rsi_oversold
        stoch_oversold = self.stoch.lines.percK[0] < self.p.stoch_oversold
        
        return rsi_oversold or stoch_oversold

    def check_oscillator_overbought(self):
        """Check if oscillators indicate overbought conditions"""
        if len(self.rsi) == 0 or len(self.stoch) == 0:
            return False
            
        rsi_overbought = self.rsi[0] > self.p.rsi_overbought
        stoch_overbought = self.stoch.lines.percK[0] > self.p.stoch_overbought
        
        return rsi_overbought or stoch_overbought

    def check_range_breakout(self):
        """Check if the micro-range has been broken"""
        if not self.micro_range_detected:
            return False, None
            
        current_price = self.data.close[0]
        
        if current_price > self.range_resistance * (1 + self.p.range_tolerance_pct):
            return True, 'upside'
        elif current_price < self.range_support * (1 - self.p.range_tolerance_pct):
            return True, 'downside'
            
        return False, None

    def check_exit_conditions(self):
        """Check for micro-scalp exit conditions"""
        if not self.position:
            return False, None
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Time-based exit
        if bars_held >= self.p.max_hold_bars:
            return True, "time_exit"
        
        # Check for range breakout (exit immediately)
        breakout, direction = self.check_range_breakout()
        if breakout:
            return True, f"range_breakout_{direction}"
        
        # Profit target exits
        if self.entry_price and self.entry_price > 0:
            profit_pct = abs(current_price - self.entry_price) / self.entry_price
            
            if pos.size > 0:  # Long position
                # Take profit target or resistance approached
                if (current_price >= self.entry_price * (1 + self.p.take_profit_pct) or
                    self.is_near_resistance(current_price) or
                    self.check_oscillator_overbought()):
                    return True, "profit_target_long"
                    
                # Move to break-even
                if (not self.break_even_moved and 
                    current_price >= self.entry_price * (1 + self.p.break_even_buffer_pct)):
                    return True, "break_even_long"
                    
            elif pos.size < 0:  # Short position
                # Take profit target or support approached
                if (current_price <= self.entry_price * (1 - self.p.take_profit_pct) or
                    self.is_near_support(current_price) or
                    self.check_oscillator_oversold()):
                    return True, "profit_target_short"
                    
                # Move to break-even
                if (not self.break_even_moved and 
                    current_price <= self.entry_price * (1 - self.p.break_even_buffer_pct)):
                    return True, "break_even_short"
        
        return False, None

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.atr_period, self.p.bb_period, self.p.rsi_period, 
                      self.p.stoch_k_period, self.p.volume_avg_period, self.p.range_detection_bars)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Step 1: Detect low volatility environment
        if self.is_low_volatility_environment():
            self.low_vol_confirmed = True
            self.low_vol_bars_count += 1
        else:
            self.low_vol_confirmed = False
            self.low_vol_bars_count = 0
            self.micro_range_detected = False
        
        # Step 2: Detect micro-range only in low-vol environment
        if self.low_vol_confirmed and self.low_vol_bars_count >= self.p.min_range_bars:
            if self.detect_micro_range():
                self.micro_range_detected = True
        
        # Reset range if it's been too long since last update
        if (self.micro_range_detected and 
            len(self) - self.last_range_update > self.p.range_detection_bars):
            self.micro_range_detected = False
        
        if not self.position:  # If no position is open
            # Only trade if we have confirmed micro-range in low-vol environment
            if not (self.low_vol_confirmed and self.micro_range_detected):
                return
                
            position_size = self.calculate_position_size(current_price)
            if position_size <= 0:
                return
            
            # Entry Condition 1: Long near micro-support
            if (self.is_near_support(current_price) and 
                self.check_oscillator_oversold()):
                
                stop_price = self.range_support * (1 - self.p.stop_loss_buffer_pct)
                take_profit = min(self.range_middle, 
                                current_price * (1 + self.p.take_profit_pct))
                
                parent, stop, limit = self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_price,
                    limitprice=take_profit,
                )
                self.order = parent
                self.order_parent_ref = parent.ref
                self.entry_bar = len(self)
                self.entry_side = 'long'
                self.entry_price = current_price
                self.break_even_moved = False
                
            # Entry Condition 2: Short near micro-resistance
            elif (self.is_near_resistance(current_price) and 
                  self.check_oscillator_overbought()):
                
                stop_price = self.range_resistance * (1 + self.p.stop_loss_buffer_pct)
                take_profit = max(self.range_middle, 
                                current_price * (1 - self.p.take_profit_pct))
                
                parent, stop, limit = self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_price,
                    limitprice=take_profit,
                )
                self.order = parent
                self.order_parent_ref = parent.ref
                self.entry_bar = len(self)
                self.entry_side = 'short'
                self.entry_price = current_price
                self.break_even_moved = False
                
        else:
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions()
            
            if should_exit:
                if exit_reason in ["break_even_long", "break_even_short"]:
                    # Move stop to break-even instead of closing
                    self.break_even_moved = True
                    # Note: In a real implementation, you'd modify the stop order here
                    # For this backtest, we'll just track the flag
                else:
                    # Close position
                    self.close()
                    self.closing = True
                    self.entry_bar = None
                    self.entry_side = None
                    self.entry_price = None
                    self.break_even_moved = False

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
        "atr_period": kwargs.get("atr_period", 14),
        "atr_low_threshold": kwargs.get("atr_low_threshold", 0.002),
        "bb_period": kwargs.get("bb_period", 20),
        "bb_std": kwargs.get("bb_std", 2.0),
        "bb_squeeze_threshold": kwargs.get("bb_squeeze_threshold", 0.005),
        "rsi_period": kwargs.get("rsi_period", 7),
        "rsi_oversold": kwargs.get("rsi_oversold", 30),
        "rsi_overbought": kwargs.get("rsi_overbought", 70),
        "stoch_k_period": kwargs.get("stoch_k_period", 14),
        "stoch_d_period": kwargs.get("stoch_d_period", 3),
        "stoch_oversold": kwargs.get("stoch_oversold", 20),
        "stoch_overbought": kwargs.get("stoch_overbought", 80),
        "volume_avg_period": kwargs.get("volume_avg_period", 20),
        "volume_decline_threshold": kwargs.get("volume_decline_threshold", 0.8),
        "range_detection_bars": kwargs.get("range_detection_bars", 20),
        "min_range_touches": kwargs.get("min_range_touches", 2),
        "range_tolerance_pct": kwargs.get("range_tolerance_pct", 0.001),
        "micro_range_max_pct": kwargs.get("micro_range_max_pct", 0.003),
        "stop_loss_buffer_pct": kwargs.get("stop_loss_buffer_pct", 0.0005),
        "take_profit_pct": kwargs.get("take_profit_pct", 0.0015),
        "max_hold_bars": kwargs.get("max_hold_bars", 10),
        "min_range_bars": kwargs.get("min_range_bars", 6),
        "position_size_reduction": kwargs.get("position_size_reduction", 0.7),
        "break_even_buffer_pct": kwargs.get("break_even_buffer_pct", 0.0003),
    }
    cerebro.addstrategy(Micro_Range_Scalp_Strategy, **strategy_params)
    
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
        atr_period=14,
        atr_low_threshold=0.002,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.005,
        rsi_period=7,
        rsi_oversold=30,
        rsi_overbought=70,
        stoch_k_period=14,
        stoch_d_period=3,
        stoch_oversold=20,
        stoch_overbought=80,
        volume_avg_period=20,
        volume_decline_threshold=0.8,
        range_detection_bars=20,
        min_range_touches=2,
        range_tolerance_pct=0.001,
        micro_range_max_pct=0.003,
        stop_loss_buffer_pct=0.0005,
        take_profit_pct=0.0015,
        max_hold_bars=10,
        min_range_bars=6,
        position_size_reduction=0.7,
        break_even_buffer_pct=0.0003,
        leverage=leverage
    )
    
    log_result(
        strategy="Micro_Range_Scalp",
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

        # if all_results:
        #     pd.DataFrame(all_results).to_csv("results/ema_adx_backtest_results.csv", index=False)

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

