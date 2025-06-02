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

class MACD_Momentum_Strategy(bt.Strategy):
    params = (
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("rsi_period", 14),
        ("sma_trend", 50),
        ("rsi_bull_threshold", 50),
        ("rsi_bear_threshold", 50),
        ("rsi_overbought", 70),
        ("rsi_oversold", 30),
        ("stop_loss", 0.002),  # 0.2% tight stop for scalping
        ("take_profit_1", 0.004),  # 0.4% first target (1:2 RR)
        ("take_profit_2", 0.008),  # 0.8% second target for runners
        ("atr_period", 14),
        ("atr_stop_multiplier", 1.0),  # Tighter ATR multiplier for scalping
        ("time_stop_bars", 20),  # Quick exit for scalping
        ("min_volume_ratio", 1.2),  # Volume should be 20% above average
        ("momentum_confirmation_bars", 3),  # Bars to confirm momentum
        ("trail_stop_pct", 0.001),  # 0.1% trailing stop for runners
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.sma_trend = bt.indicators.SMA(self.data.close, period=self.p.sma_trend)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.p.atr_period)
        
        # MACD components for easier access
        self.macd_line = self.macd.macd
        self.macd_signal_line = self.macd.signal
        self.macd_histogram = self.macd_line - self.macd_signal_line
        
        # MACD crossover detection
        self.macd_cross = bt.indicators.CrossOver(self.macd_line, self.macd_signal_line)
        
        # Volume analysis
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Momentum state tracking
        self.trend_direction = 0  # 1 for bullish, -1 for bearish, 0 for neutral
        self.momentum_confirmed = False
        self.waiting_for_pullback = False
        self.pullback_detected = False
        self.momentum_turn_bar = None
        
        # Track recent highs/lows for breakout confirmation
        self.recent_high = 0
        self.recent_low = float('inf')
        self.consolidation_high = 0
        self.consolidation_low = float('inf')
        self.bars_in_consolidation = 0
        
        # Order and position tracking
        self.order = None
        self.order_parent_ref = None
        self.trade_dir = {}
        self.entry_bar = None
        self.entry_side = None
        self.closing = False
        self.partial_exit_done = False
        self.runner_stop_price = None

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

    def detect_trend_direction(self):
        """Determine overall trend direction using SMA"""
        current_price = self.data.close[0]
        
        if current_price > self.sma_trend[0]:
            return 1  # Bullish bias
        elif current_price < self.sma_trend[0]:
            return -1  # Bearish bias
        else:
            return 0  # Neutral

    def detect_momentum_pullback(self):
        """Detect minor pullback in momentum indicators"""
        current_price = self.data.close[0]
        
        # Update recent highs/lows for consolidation detection
        if self.trend_direction == 1:  # Bullish trend
            if current_price > self.recent_high:
                self.recent_high = current_price
                self.bars_in_consolidation = 0
                self.consolidation_high = current_price
                self.consolidation_low = current_price
            elif current_price < self.recent_low:
                self.recent_low = current_price
                
            # Check for minor pullback conditions
            macd_pullback = (self.macd_histogram[0] < self.macd_histogram[-1] or 
                           self.macd_cross[0] < 0)  # MACD turning down or crossing below signal
            rsi_pullback = self.rsi[0] < self.rsi[-1]  # RSI declining
            
            if macd_pullback and rsi_pullback:
                self.pullback_detected = True
                self.waiting_for_pullback = False
                return True
                
        elif self.trend_direction == -1:  # Bearish trend
            if current_price < self.recent_low:
                self.recent_low = current_price
                self.bars_in_consolidation = 0
                self.consolidation_high = current_price
                self.consolidation_low = current_price
            elif current_price > self.recent_high:
                self.recent_high = current_price
                
            # Check for minor pullback conditions
            macd_pullback = (self.macd_histogram[0] > self.macd_histogram[-1] or 
                           self.macd_cross[0] > 0)  # MACD turning up or crossing above signal
            rsi_pullback = self.rsi[0] > self.rsi[-1]  # RSI rising
            
            if macd_pullback and rsi_pullback:
                self.pullback_detected = True
                self.waiting_for_pullback = False
                return True
        
        # Track consolidation periods
        price_range = abs(current_price - self.data.close[-1]) / current_price
        if price_range < 0.001:  # Very small movement
            self.bars_in_consolidation += 1
            self.consolidation_high = max(self.consolidation_high, self.data.high[0])
            self.consolidation_low = min(self.consolidation_low, self.data.low[0])
        else:
            self.bars_in_consolidation = 0
            
        return False

    def check_momentum_resumption(self):
        """Check for momentum resumption after pullback"""
        if not self.pullback_detected:
            return False, None
            
        current_price = self.data.close[0]
        
        # Volume confirmation
        volume_ok = True
        if hasattr(self.data, 'volume') and len(self.volume_sma) > 0:
            volume_ok = self.data.volume[0] > self.volume_sma[0] * self.p.min_volume_ratio
        
        if self.trend_direction == 1:  # Bullish trend
            # Check for momentum resumption signals
            macd_resumption = (self.macd_cross[0] > 0 and  # MACD crossing above signal
                             self.macd_histogram[0] > 0)  # Histogram positive
            rsi_resumption = (self.rsi[0] > self.p.rsi_bull_threshold and
                            self.rsi[0] > self.rsi[-1])  # RSI above threshold and rising
            
            # Price breakout confirmation
            price_breakout = False
            if self.bars_in_consolidation >= 2:  # Had some consolidation
                price_breakout = current_price > self.consolidation_high
            else:
                price_breakout = current_price > self.recent_high * 0.999  # Small tolerance
                
            if macd_resumption and rsi_resumption and price_breakout and volume_ok:
                return True, 'long'
                
        elif self.trend_direction == -1:  # Bearish trend
            # Check for momentum resumption signals
            macd_resumption = (self.macd_cross[0] < 0 and  # MACD crossing below signal
                             self.macd_histogram[0] < 0)  # Histogram negative
            rsi_resumption = (self.rsi[0] < self.p.rsi_bear_threshold and
                            self.rsi[0] < self.rsi[-1])  # RSI below threshold and falling
            
            # Price breakout confirmation
            price_breakout = False
            if self.bars_in_consolidation >= 2:  # Had some consolidation
                price_breakout = current_price < self.consolidation_low
            else:
                price_breakout = current_price < self.recent_low * 1.001  # Small tolerance
                
            if macd_resumption and rsi_resumption and price_breakout and volume_ok:
                return True, 'short'
        
        return False, None

    def check_scalp_exit_conditions(self):
        """Check for quick scalping exit conditions"""
        current_price = self.data.close[0]
        pos = self.getposition()
        
        if pos.size == 0:
            return False
            
        bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
        
        if pos.size > 0:  # Long position
            # Exit conditions for long
            rsi_overbought = self.rsi[0] > self.p.rsi_overbought
            macd_weakening = (self.macd_histogram[0] < self.macd_histogram[-1] or
                            self.macd_cross[0] < 0)
            rsi_momentum_loss = self.rsi[0] < self.p.rsi_bull_threshold
            time_stop = bars_held >= self.p.time_stop_bars
            
            return rsi_overbought or macd_weakening or rsi_momentum_loss or time_stop
            
        elif pos.size < 0:  # Short position
            # Exit conditions for short
            rsi_oversold = self.rsi[0] < self.p.rsi_oversold
            macd_weakening = (self.macd_histogram[0] > self.macd_histogram[-1] or
                            self.macd_cross[0] > 0)
            rsi_momentum_loss = self.rsi[0] > self.p.rsi_bear_threshold
            time_stop = bars_held >= self.p.time_stop_bars
            
            return rsi_oversold or macd_weakening or rsi_momentum_loss or time_stop
        
        return False

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.macd_slow, self.p.rsi_period, self.p.sma_trend, self.p.atr_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Update trend direction
        self.trend_direction = self.detect_trend_direction()
        
        if not self.position:  # If no position is open
            # Look for momentum pullback
            if not self.pullback_detected and self.trend_direction != 0:
                self.detect_momentum_pullback()
            
            # Check for momentum resumption entry
            entry_signal, direction = self.check_momentum_resumption()
            
            if entry_signal and direction:
                position_size = self.calculate_position_size(current_price)
                
                if position_size <= 0:
                    return
                
                # Calculate stop loss and take profit for scalping
                atr_value = self.atr[0]
                
                if direction == 'long':
                    # Tight stop below recent consolidation low or fixed percentage
                    stop_price = min(
                        self.consolidation_low * (1 - self.p.stop_loss),
                        current_price - (atr_value * self.p.atr_stop_multiplier)
                    )
                    # First take profit target
                    take_profit = current_price * (1 + self.p.take_profit_1)
                    
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
                    self.partial_exit_done = False
                    self.runner_stop_price = stop_price
                    
                elif direction == 'short':
                    # Tight stop above recent consolidation high or fixed percentage
                    stop_price = max(
                        self.consolidation_high * (1 + self.p.stop_loss),
                        current_price + (atr_value * self.p.atr_stop_multiplier)
                    )
                    # First take profit target
                    take_profit = current_price * (1 - self.p.take_profit_1)
                    
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
                    self.partial_exit_done = False
                    self.runner_stop_price = stop_price
                
                # Reset pullback detection
                self.pullback_detected = False
                self.waiting_for_pullback = True
        else:
            # Check for scalping exit conditions
            should_exit = self.check_scalp_exit_conditions()
            
            if should_exit:
                self.close()
                self.closing = True
                self.entry_bar = None
                self.entry_side = None
                self.partial_exit_done = False
                self.runner_stop_price = None

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
        "macd_fast": kwargs.get("macd_fast", 12),
        "macd_slow": kwargs.get("macd_slow", 26),
        "macd_signal": kwargs.get("macd_signal", 9),
        "rsi_period": kwargs.get("rsi_period", 14),
        "sma_trend": kwargs.get("sma_trend", 50),
        "rsi_bull_threshold": kwargs.get("rsi_bull_threshold", 50),
        "rsi_bear_threshold": kwargs.get("rsi_bear_threshold", 50),
        "rsi_overbought": kwargs.get("rsi_overbought", 70),
        "rsi_oversold": kwargs.get("rsi_oversold", 30),
        "stop_loss": kwargs.get("stop_loss", 0.002),
        "take_profit_1": kwargs.get("take_profit_1", 0.004),
        "take_profit_2": kwargs.get("take_profit_2", 0.008),
        "atr_period": kwargs.get("atr_period", 14),
        "atr_stop_multiplier": kwargs.get("atr_stop_multiplier", 1.0),
        "time_stop_bars": kwargs.get("time_stop_bars", 20),
        "min_volume_ratio": kwargs.get("min_volume_ratio", 1.2),
        "momentum_confirmation_bars": kwargs.get("momentum_confirmation_bars", 3),
        "trail_stop_pct": kwargs.get("trail_stop_pct", 0.001),
    }
    cerebro.addstrategy(MACD_Momentum_Strategy, **strategy_params)
    
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
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_period=14,
        sma_trend=50,
        rsi_bull_threshold=50,
        rsi_bear_threshold=50,
        rsi_overbought=70,
        rsi_oversold=30,
        stop_loss=0.002,
        take_profit_1=0.004,
        take_profit_2=0.008,
        atr_period=14,
        atr_stop_multiplier=1.0,
        time_stop_bars=20,
        min_volume_ratio=1.2,
        momentum_confirmation_bars=3,
        trail_stop_pct=0.001,
        leverage=leverage
    )
    
    log_result(
        strategy="MACD_Momentum_Scalper",
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
            results_path = os.path.join(results_folder, "MACD_Momentum.csv")
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
                    results_path = os.path.join(results_folder, "MACD_Momentum.csv")
                    pd.DataFrame(all_results).to_csv(results_path, index=False)
                    
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc())
