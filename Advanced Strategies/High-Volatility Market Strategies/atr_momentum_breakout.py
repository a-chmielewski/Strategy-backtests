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

class ATR_Momentum_Breakout_Strategy(bt.Strategy):
    params = (
        ("atr_period", 14),
        ("atr_volatility_threshold", 1.5),  # ATR must be 1.5x recent average
        ("range_multiplier", 2.0),  # Current candle range must be 2x recent average
        ("donchian_period", 20),  # Period for breakout level detection
        ("consolidation_period", 5),  # Minutes to define consolidation
        ("volume_surge_multiplier", 2.5),  # Volume must be 2.5x average for confirmation
        ("volume_avg_period", 10),  # Bars for volume average
        ("trend_ema_period", 21),  # Fast EMA for trend alignment
        ("scalp_target_pct", 0.005),  # 0.5% quick scalp target
        ("scalp_time_limit", 3),  # 3 bars max for scalp target
        ("failed_breakout_time", 3),  # Exit after 3 bars if no immediate profit
        ("atr_stop_multiplier", 1.5),  # Stop loss = 1.5x ATR below entry
        ("atr_target_multiplier", 2.0),  # Take profit = 2x ATR above entry
        ("trailing_atr_multiplier", 1.0),  # Trailing stop = 1x ATR
        ("min_volatility_threshold", 0.002),  # Minimum 0.2% volatility to trade
        ("partial_profit_pct", 0.7),  # Take 70% profit on scalp target hit
        ("max_position_time", 10),  # Maximum bars to hold position
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.atr_sma = bt.indicators.SMA(self.atr, period=self.p.atr_period)
        
        # Donchian Channels for breakout levels
        self.donchian_high = bt.indicators.Highest(self.data.high, period=self.p.donchian_period)
        self.donchian_low = bt.indicators.Lowest(self.data.low, period=self.p.donchian_period)
        
        # Trend filter
        self.trend_ema = bt.indicators.EMA(self.data.close, period=self.p.trend_ema_period)
        
        # Volume indicators
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.volume_avg_period)
        
        # Range tracking for volatility
        self.true_range = []
        self.recent_ranges = []
        
        # Consolidation and breakout tracking
        self.consolidation_high = 0
        self.consolidation_low = 0
        self.consolidation_start = 0
        self.is_in_consolidation = False
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.scalp_target_hit = False
        self.trailing_stop_price = None
        self.high_water_mark = None
        self.low_water_mark = None
        
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

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def is_high_volatility_regime(self):
        """Check if we're in a high volatility environment"""
        if len(self.atr) < self.p.atr_period or len(self.atr_sma) == 0:
            return False
            
        current_atr = self.atr[0]
        avg_atr = self.atr_sma[0]
        
        # ATR must be above threshold compared to recent average
        atr_condition = current_atr > avg_atr * self.p.atr_volatility_threshold
        
        # Current candle range must be significantly larger than recent average
        current_range = self.data.high[0] - self.data.low[0]
        
        # Calculate recent average range
        if len(self.recent_ranges) >= 10:
            avg_range = sum(self.recent_ranges[-10:]) / len(self.recent_ranges[-10:])
            range_condition = current_range > avg_range * self.p.range_multiplier
        else:
            range_condition = True  # Not enough data, assume ok
        
        # Minimum volatility threshold
        volatility_condition = current_atr / self.data.close[0] > self.p.min_volatility_threshold
        
        return atr_condition and range_condition and volatility_condition

    def detect_consolidation(self):
        """Detect if price has been in consolidation before potential breakout"""
        if len(self) < self.p.consolidation_period:
            return False
            
        # Look at recent highs and lows
        recent_highs = [self.data.high[-i] for i in range(self.p.consolidation_period)]
        recent_lows = [self.data.low[-i] for i in range(self.p.consolidation_period)]
        
        consolidation_high = max(recent_highs)
        consolidation_low = min(recent_lows)
        
        # Check if range is tight relative to ATR
        range_size = consolidation_high - consolidation_low
        atr_value = self.atr[0] if len(self.atr) > 0 else range_size
        
        # Consolidation if range is less than 1.5x ATR
        if range_size < atr_value * 1.5:
            self.consolidation_high = consolidation_high
            self.consolidation_low = consolidation_low
            self.is_in_consolidation = True
            return True
            
        self.is_in_consolidation = False
        return False

    def is_volume_surge(self):
        """Check for volume surge confirmation"""
        if not hasattr(self.data, 'volume') or len(self.volume_sma) == 0:
            return True  # Assume volume is ok if no volume data
            
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        return current_volume > avg_volume * self.p.volume_surge_multiplier

    def is_trend_aligned(self, direction):
        """Check if breakout direction aligns with trend"""
        if len(self.trend_ema) == 0:
            return True  # Assume trend is ok if no EMA data
            
        current_price = self.data.close[0]
        ema_value = self.trend_ema[0]
        
        if direction == 'long':
            return current_price > ema_value
        elif direction == 'short':
            return current_price < ema_value
            
        return False

    def detect_breakout(self):
        """Detect clean breakout with all confirmations"""
        current_price = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        
        # Check for breakout above recent highs
        if len(self.donchian_high) > 0:
            resistance_level = self.donchian_high[-1]  # Previous bar's resistance
            
            # Bullish breakout conditions
            if (current_high > resistance_level and 
                current_price > resistance_level and
                self.is_volume_surge() and
                self.is_trend_aligned('long')):
                return 'long', resistance_level
        
        # Check for breakdown below recent lows
        if len(self.donchian_low) > 0:
            support_level = self.donchian_low[-1]  # Previous bar's support
            
            # Bearish breakdown conditions
            if (current_low < support_level and 
                current_price < support_level and
                self.is_volume_surge() and
                self.is_trend_aligned('short')):
                return 'short', support_level
                
        return None, None

    def update_trailing_stop(self):
        """Update trailing stop based on ATR"""
        if not self.position:
            return
            
        current_price = self.data.close[0]
        pos = self.getposition()
        atr_value = self.atr[0] if len(self.atr) > 0 else 0
        
        if pos.size > 0:  # Long position
            # Update high water mark
            if self.high_water_mark is None or current_price > self.high_water_mark:
                self.high_water_mark = current_price
                
            # Calculate trailing stop
            new_trailing_stop = self.high_water_mark - (atr_value * self.p.trailing_atr_multiplier)
            
            # Update trailing stop if it's higher than current
            if self.trailing_stop_price is None or new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
                
        elif pos.size < 0:  # Short position
            # Update low water mark
            if self.low_water_mark is None or current_price < self.low_water_mark:
                self.low_water_mark = current_price
                
            # Calculate trailing stop
            new_trailing_stop = self.low_water_mark + (atr_value * self.p.trailing_atr_multiplier)
            
            # Update trailing stop if it's lower than current
            if self.trailing_stop_price is None or new_trailing_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop

    def check_scalp_target(self):
        """Check if quick scalp target is hit"""
        if not self.position or self.entry_price is None:
            return False
            
        current_price = self.data.close[0]
        pos = self.getposition()
        
        if pos.size > 0:  # Long position
            target_price = self.entry_price * (1 + self.p.scalp_target_pct)
            return current_price >= target_price
        elif pos.size < 0:  # Short position
            target_price = self.entry_price * (1 - self.p.scalp_target_pct)
            return current_price <= target_price
            
        return False

    def check_exit_conditions(self):
        """Check various exit conditions"""
        if not self.position:
            return False, None
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Time-based exit for failed breakouts
        if bars_held >= self.p.failed_breakout_time and not self.scalp_target_hit:
            # Check if trade is not profitable
            if pos.size > 0 and current_price <= self.entry_price:
                return True, "failed_breakout_long"
            elif pos.size < 0 and current_price >= self.entry_price:
                return True, "failed_breakout_short"
        
        # Maximum position time
        if bars_held >= self.p.max_position_time:
            return True, "max_time"
            
        # Quick scalp target hit
        if self.check_scalp_target() and not self.scalp_target_hit:
            self.scalp_target_hit = True
            return True, "scalp_target"
            
        # Trailing stop hit (after scalp target achieved)
        if self.scalp_target_hit and self.trailing_stop_price is not None:
            if pos.size > 0 and current_price <= self.trailing_stop_price:
                return True, "trailing_stop_long"
            elif pos.size < 0 and current_price >= self.trailing_stop_price:
                return True, "trailing_stop_short"
                
        return False, None

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.atr_period, self.p.donchian_period, self.p.trend_ema_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Update range tracking
        current_range = self.data.high[0] - self.data.low[0]
        self.recent_ranges.append(current_range)
        if len(self.recent_ranges) > 20:
            self.recent_ranges.pop(0)
        
        if not self.position:  # If no position is open
            # Step 1: Check for high volatility regime
            if not self.is_high_volatility_regime():
                return
                
            # Step 2: Detect consolidation (optional - helps identify better breakouts)
            self.detect_consolidation()
            
            # Step 3: Look for breakout signal
            breakout_direction, breakout_level = self.detect_breakout()
            
            if breakout_direction is not None:
                position_size = self.calculate_position_size(current_price)
                
                if position_size <= 0:
                    return
                
                # Calculate stops and targets based on ATR
                atr_value = self.atr[0] if len(self.atr) > 0 else current_price * 0.01
                
                if breakout_direction == 'long':
                    # Calculate stop and target for long
                    stop_price = current_price - (atr_value * self.p.atr_stop_multiplier)
                    take_profit = current_price + (atr_value * self.p.atr_target_multiplier)
                    
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
                    self.scalp_target_hit = False
                    self.trailing_stop_price = None
                    self.high_water_mark = None
                    
                elif breakout_direction == 'short':
                    # Calculate stop and target for short
                    stop_price = current_price + (atr_value * self.p.atr_stop_multiplier)
                    take_profit = current_price - (atr_value * self.p.atr_target_multiplier)
                    
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
                    self.scalp_target_hit = False
                    self.trailing_stop_price = None
                    self.low_water_mark = None
        else:
            # Update trailing stop
            self.update_trailing_stop()
            
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions()
            
            if should_exit:
                if exit_reason == "scalp_target":
                    # Take partial profits on scalp target
                    pos = self.getposition()
                    partial_size = abs(pos.size) * self.p.partial_profit_pct
                    
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
        "atr_volatility_threshold": kwargs.get("atr_volatility_threshold", 1.5),
        "range_multiplier": kwargs.get("range_multiplier", 2.0),
        "donchian_period": kwargs.get("donchian_period", 20),
        "consolidation_period": kwargs.get("consolidation_period", 5),
        "volume_surge_multiplier": kwargs.get("volume_surge_multiplier", 2.5),
        "volume_avg_period": kwargs.get("volume_avg_period", 10),
        "trend_ema_period": kwargs.get("trend_ema_period", 21),
        "scalp_target_pct": kwargs.get("scalp_target_pct", 0.005),
        "scalp_time_limit": kwargs.get("scalp_time_limit", 3),
        "failed_breakout_time": kwargs.get("failed_breakout_time", 3),
        "atr_stop_multiplier": kwargs.get("atr_stop_multiplier", 1.5),
        "atr_target_multiplier": kwargs.get("atr_target_multiplier", 2.0),
        "trailing_atr_multiplier": kwargs.get("trailing_atr_multiplier", 1.0),
        "min_volatility_threshold": kwargs.get("min_volatility_threshold", 0.002),
        "partial_profit_pct": kwargs.get("partial_profit_pct", 0.7),
        "max_position_time": kwargs.get("max_position_time", 10),
    }
    cerebro.addstrategy(ATR_Momentum_Breakout_Strategy, **strategy_params)
    
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
        atr_volatility_threshold=1.5,
        range_multiplier=2.0,
        donchian_period=20,
        consolidation_period=5,
        volume_surge_multiplier=2.5,
        volume_avg_period=10,
        trend_ema_period=21,
        scalp_target_pct=0.005,
        scalp_time_limit=3,
        failed_breakout_time=3,
        atr_stop_multiplier=1.5,
        atr_target_multiplier=2.0,
        trailing_atr_multiplier=1.0,
        min_volatility_threshold=0.002,
        partial_profit_pct=0.7,
        max_position_time=10,
        leverage=leverage
    )
    
    log_result(
        strategy="ATR_Momentum_Breakout",
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
        #     pd.DataFrame(all_results).to_csv("results/atr_momentum_breakout_results.csv", index=False)

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
                    pd.DataFrame(all_results).to_csv("atr_momentum_breakout_results.csv", index=False)
                    
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc())
