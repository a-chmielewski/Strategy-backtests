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

class RSI_Range_Scalp_Strategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("rsi_neutral_low", 40),
        ("rsi_neutral_high", 60),
        ("sma_reference", 100),
        ("range_lookback", 50),  # Bars to look back for range detection
        ("min_range_width", 0.008),  # Minimum 0.8% range width to trade
        ("support_resistance_tolerance", 0.002),  # 0.2% tolerance for S/R levels
        ("stop_loss_pct", 0.002),  # 0.2% tight stop outside S/R
        ("take_profit_ratio", 1.2),  # 1.2:1 reward/risk ratio
        ("time_stop_bars", 20),  # Quick exit for range scalping
        ("consecutive_stops_limit", 2),  # Stop trading after 2 consecutive stops
        ("volume_surge_threshold", 1.5),  # 1.5x average volume for breakout detection
        ("pattern_confirmation_bars", 2),  # Bars to confirm candlestick patterns
        ("range_middle_buffer", 0.3),  # Don't trade within 30% of range middle
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.sma_reference = bt.indicators.SMA(self.data.close, period=self.p.sma_reference)
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Range detection variables
        self.range_active = False
        self.support_level = 0
        self.resistance_level = 0
        self.range_middle = 0
        self.range_width = 0
        self.last_range_update = 0
        
        # Track range trading performance
        self.consecutive_stops = 0
        self.range_trading_enabled = True
        self.last_stop_bar = None
        
        # Candlestick pattern detection
        self.pattern_detected = False
        self.pattern_type = None  # 'bullish_engulfing', 'bearish_engulfing', etc.
        self.pattern_bar = None
        
        # Track recent highs/lows for range detection
        self.recent_highs = []
        self.recent_lows = []
        
        # Order and position tracking
        self.order = None
        self.order_parent_ref = None
        self.trade_dir = {}
        self.entry_bar = None
        self.entry_side = None
        self.closing = False
        self.entry_level = None  # Track entry support/resistance level

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

    def detect_range_levels(self):
        """Detect support and resistance levels from recent price action"""
        if len(self) < self.p.range_lookback:
            return False
            
        # Get recent highs and lows
        lookback_period = min(self.p.range_lookback, len(self))
        highs = [self.data.high[-i] for i in range(lookback_period)]
        lows = [self.data.low[-i] for i in range(lookback_period)]
        
        # Find significant levels by clustering highs and lows
        highs.sort(reverse=True)
        lows.sort()
        
        # Take top quartile of highs and bottom quartile of lows
        top_quartile = int(len(highs) * 0.25)
        bottom_quartile = int(len(lows) * 0.25)
        
        if top_quartile == 0:
            top_quartile = 1
        if bottom_quartile == 0:
            bottom_quartile = 1
            
        # Calculate potential resistance (average of top highs)
        potential_resistance = sum(highs[:top_quartile]) / top_quartile
        
        # Calculate potential support (average of bottom lows)
        potential_support = sum(lows[:bottom_quartile]) / bottom_quartile
        
        # Check if we have a valid range
        range_width = (potential_resistance - potential_support) / potential_support
        
        if range_width >= self.p.min_range_width:
            self.support_level = potential_support
            self.resistance_level = potential_resistance
            self.range_middle = (self.support_level + self.resistance_level) / 2
            self.range_width = range_width
            self.range_active = True
            self.last_range_update = len(self)
            return True
        
        return False

    def is_near_support(self, price):
        """Check if price is near support level"""
        if not self.range_active:
            return False
        tolerance = self.support_level * self.p.support_resistance_tolerance
        return self.support_level - tolerance <= price <= self.support_level + tolerance

    def is_near_resistance(self, price):
        """Check if price is near resistance level"""
        if not self.range_active:
            return False
        tolerance = self.resistance_level * self.p.support_resistance_tolerance
        return self.resistance_level - tolerance <= price <= self.resistance_level + tolerance

    def is_in_range_middle(self, price):
        """Check if price is in the middle zone (avoid trading here)"""
        if not self.range_active:
            return True
            
        range_size = self.resistance_level - self.support_level
        middle_buffer = range_size * self.p.range_middle_buffer
        
        middle_low = self.range_middle - middle_buffer
        middle_high = self.range_middle + middle_buffer
        
        return middle_low <= price <= middle_high

    def detect_candlestick_patterns(self):
        """Detect bullish and bearish candlestick patterns"""
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
        
        current_price = curr_close
        
        # Bullish patterns (for support bounce)
        if self.is_near_support(current_price):
            # Bullish engulfing
            if (prev_close < prev_open and  # Previous red candle
                curr_close > curr_open and  # Current green candle
                curr_open < prev_close and  # Current opens below prev close
                curr_close > prev_open):    # Current closes above prev open
                return True, 'bullish_engulfing'
            
            # Hammer/Pin bar at support
            body_size = abs(curr_close - curr_open)
            lower_shadow = curr_open - curr_low if curr_close > curr_open else curr_close - curr_low
            upper_shadow = curr_high - curr_close if curr_close > curr_open else curr_high - curr_open
            
            if (lower_shadow > 2 * body_size and  # Long lower shadow
                upper_shadow < body_size * 0.5):  # Small upper shadow
                return True, 'hammer'
        
        # Bearish patterns (for resistance rejection)
        elif self.is_near_resistance(current_price):
            # Bearish engulfing
            if (prev_close > prev_open and  # Previous green candle
                curr_close < curr_open and  # Current red candle
                curr_open > prev_close and  # Current opens above prev close
                curr_close < prev_open):    # Current closes below prev open
                return True, 'bearish_engulfing'
            
            # Shooting star at resistance
            body_size = abs(curr_close - curr_open)
            upper_shadow = curr_high - curr_close if curr_close < curr_open else curr_high - curr_open
            lower_shadow = curr_open - curr_low if curr_close < curr_open else curr_close - curr_low
            
            if (upper_shadow > 2 * body_size and  # Long upper shadow
                lower_shadow < body_size * 0.5):  # Small lower shadow
                return True, 'shooting_star'
        
        return False, None

    def check_rsi_divergence(self, direction):
        """Check for RSI divergence at support/resistance"""
        if len(self) < 10:
            return False
            
        current_price = self.data.close[0]
        current_rsi = self.rsi[0]
        
        # Look back for previous touch of same level
        for i in range(2, min(20, len(self))):
            past_price = self.data.close[-i]
            past_rsi = self.rsi[-i]
            
            if direction == 'bullish' and self.is_near_support(past_price):
                # Bullish divergence: lower low in price, higher low in RSI
                if current_price < past_price and current_rsi > past_rsi:
                    return True
            elif direction == 'bearish' and self.is_near_resistance(past_price):
                # Bearish divergence: higher high in price, lower high in RSI
                if current_price > past_price and current_rsi < past_rsi:
                    return True
                    
        return False

    def check_volume_breakout_signal(self):
        """Check for volume surge that might indicate breakout"""
        if not hasattr(self.data, 'volume') or len(self.volume_sma) == 0:
            return False
            
        current_volume = self.data.volume[0]
        avg_volume = self.volume_sma[0]
        
        return current_volume > avg_volume * self.p.volume_surge_threshold

    def check_range_exit_conditions(self):
        """Check for range scalping exit conditions"""
        if not self.position:
            return False
            
        current_price = self.data.close[0]
        pos = self.getposition()
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        # Time stop
        if bars_held >= self.p.time_stop_bars:
            return True
            
        if pos.size > 0:  # Long position
            # Exit when RSI reaches neutral zone
            if self.rsi[0] >= self.p.rsi_neutral_low:
                return True
            # Exit when approaching range middle
            if current_price >= self.range_middle * 0.95:
                return True
                
        elif pos.size < 0:  # Short position
            # Exit when RSI reaches neutral zone
            if self.rsi[0] <= self.p.rsi_neutral_high:
                return True
            # Exit when approaching range middle
            if current_price <= self.range_middle * 1.05:
                return True
        
        return False

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.rsi_period, self.p.sma_reference, self.p.range_lookback)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Step 1: Update range levels
        if not self.range_active or len(self) - self.last_range_update > 20:
            self.detect_range_levels()
        
        # Step 2: Check if range trading is disabled due to consecutive stops
        if not self.range_trading_enabled:
            # Re-enable after some bars have passed
            if self.last_stop_bar and len(self) - self.last_stop_bar > 50:
                self.range_trading_enabled = True
                self.consecutive_stops = 0
            return
        
        # Step 3: Check for volume breakout signal
        if self.check_volume_breakout_signal():
            if self.position:
                # Exit position on potential breakout
                self.close()
                self.closing = True
                self.entry_bar = None
                self.entry_side = None
            return
        
        if not self.position:  # If no position is open
            # Only trade if we have an active range and not in middle zone
            if not self.range_active or self.is_in_range_middle(current_price):
                return
            
            # Check for entry conditions
            pattern_detected, pattern_type = self.detect_candlestick_patterns()
            
            # Long entry conditions (near support)
            if (self.is_near_support(current_price) and 
                self.rsi[0] <= self.p.rsi_oversold and
                pattern_detected and pattern_type in ['bullish_engulfing', 'hammer']):
                
                # Additional confirmation with divergence
                divergence_ok = self.check_rsi_divergence('bullish') or True  # Divergence is bonus, not required
                
                if divergence_ok:
                    position_size = self.calculate_position_size(current_price)
                    
                    if position_size <= 0:
                        return
                    
                    # Calculate tight stop below support
                    stop_price = self.support_level * (1 - self.p.stop_loss_pct)
                    stop_distance = current_price - stop_price
                    take_profit = current_price + (stop_distance * self.p.take_profit_ratio)
                    
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
                    self.entry_level = self.support_level
            
            # Short entry conditions (near resistance)
            elif (self.is_near_resistance(current_price) and 
                  self.rsi[0] >= self.p.rsi_overbought and
                  pattern_detected and pattern_type in ['bearish_engulfing', 'shooting_star']):
                
                # Additional confirmation with divergence
                divergence_ok = self.check_rsi_divergence('bearish') or True  # Divergence is bonus, not required
                
                if divergence_ok:
                    position_size = self.calculate_position_size(current_price)
                    
                    if position_size <= 0:
                        return
                    
                    # Calculate tight stop above resistance
                    stop_price = self.resistance_level * (1 + self.p.stop_loss_pct)
                    stop_distance = stop_price - current_price
                    take_profit = current_price - (stop_distance * self.p.take_profit_ratio)
                    
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
                    self.entry_level = self.resistance_level
        else:
            # Check for early exit conditions
            should_exit = self.check_range_exit_conditions()
            
            if should_exit:
                self.close()
                self.closing = True
                self.entry_bar = None
                self.entry_side = None

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
            
            # Check if this was a stop loss
            if pnl < 0:
                self.consecutive_stops += 1
                self.last_stop_bar = len(self)
                
                # Disable range trading after consecutive stops
                if self.consecutive_stops >= self.p.consecutive_stops_limit:
                    self.range_trading_enabled = False
                    print(f"Range trading disabled after {self.consecutive_stops} consecutive stops")
            else:
                # Reset consecutive stops on winning trade
                self.consecutive_stops = 0
            
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
        "rsi_period": kwargs.get("rsi_period", 14),
        "rsi_oversold": kwargs.get("rsi_oversold", 30),
        "rsi_overbought": kwargs.get("rsi_overbought", 70),
        "rsi_neutral_low": kwargs.get("rsi_neutral_low", 40),
        "rsi_neutral_high": kwargs.get("rsi_neutral_high", 60),
        "sma_reference": kwargs.get("sma_reference", 100),
        "range_lookback": kwargs.get("range_lookback", 50),
        "min_range_width": kwargs.get("min_range_width", 0.008),
        "support_resistance_tolerance": kwargs.get("support_resistance_tolerance", 0.002),
        "stop_loss_pct": kwargs.get("stop_loss_pct", 0.002),
        "take_profit_ratio": kwargs.get("take_profit_ratio", 1.2),
        "time_stop_bars": kwargs.get("time_stop_bars", 20),
        "consecutive_stops_limit": kwargs.get("consecutive_stops_limit", 2),
        "volume_surge_threshold": kwargs.get("volume_surge_threshold", 1.5),
        "pattern_confirmation_bars": kwargs.get("pattern_confirmation_bars", 2),
        "range_middle_buffer": kwargs.get("range_middle_buffer", 0.3),
    }
    cerebro.addstrategy(RSI_Range_Scalp_Strategy, **strategy_params)
    
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
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        rsi_neutral_low=40,
        rsi_neutral_high=60,
        sma_reference=100,
        range_lookback=50,
        min_range_width=0.008,
        support_resistance_tolerance=0.002,
        stop_loss_pct=0.002,
        take_profit_ratio=1.2,
        time_stop_bars=20,
        consecutive_stops_limit=2,
        volume_surge_threshold=1.5,
        pattern_confirmation_bars=2,
        range_middle_buffer=0.3,
        leverage=leverage
    )
    
    log_result(
        strategy="RSI_Range_Scalp",
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