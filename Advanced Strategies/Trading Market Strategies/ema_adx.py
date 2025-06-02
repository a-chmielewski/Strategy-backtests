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

class EMA_ADX_Strategy(bt.Strategy):
    params = (
        ("ema_fast", 20),
        ("ema_slow", 50),
        ("adx_period", 14),
        ("rsi_period", 14),
        ("adx_threshold", 25),
        ("adx_strong", 30),
        ("rsi_bull_threshold", 50),
        ("rsi_bear_threshold", 50),
        ("stop_loss", 0.005),  # 0.5% stop loss
        ("take_profit", 0.01),  # 1% take profit (2:1 reward/risk)
        ("atr_period", 14),
        ("atr_stop_multiplier", 2.0),
        ("atr_target_multiplier", 2.0),
        ("time_stop_bars", 50),
        ("pullback_bars", 5),  # Look back this many bars for pullback detection
        ("min_pullback_pct", 0.002),  # Minimum 0.2% pullback to consider valid
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []
        
        # Initialize indicators
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.data, period=self.p.adx_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.p.atr_period)
        
        # Trend detection
        self.ema_cross = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        
        # Track trend state and pullback detection
        self.trend_direction = 0  # 1 for bullish, -1 for bearish, 0 for neutral
        self.trend_confirmed = False
        self.waiting_for_pullback = False
        self.pullback_detected = False
        self.trend_start_bar = None
        
        # Track recent highs/lows for pullback detection
        self.recent_high = 0
        self.recent_low = float('inf')
        self.pullback_extreme = 0
        
        # Order and position tracking
        self.order = None
        self.order_parent_ref = None
        self.trade_dir = {}
        self.entry_bar = None
        self.entry_side = None
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

    def detect_trend_and_strength(self):
        """Detect trend direction and strength using EMA cross and ADX"""
        current_price = self.data.close[0]
        
        # Check for fresh EMA cross
        if self.ema_cross[0] > 0:  # Fast EMA crossed above slow EMA
            if self.adx[0] > self.p.adx_threshold:
                self.trend_direction = 1  # Bullish
                self.trend_confirmed = True
                self.waiting_for_pullback = True
                self.trend_start_bar = len(self)
                self.recent_high = current_price
                self.recent_low = current_price
                return True
        elif self.ema_cross[0] < 0:  # Fast EMA crossed below slow EMA
            if self.adx[0] > self.p.adx_threshold:
                self.trend_direction = -1  # Bearish
                self.trend_confirmed = True
                self.waiting_for_pullback = True
                self.trend_start_bar = len(self)
                self.recent_high = current_price
                self.recent_low = current_price
                return True
        
        # If we have an existing trend, check if it's still valid
        if self.trend_confirmed:
            # Trend remains valid if ADX is still strong and EMAs maintain order
            if self.adx[0] < self.p.adx_threshold - 5:  # Give some buffer
                self.trend_confirmed = False
                self.waiting_for_pullback = False
                self.trend_direction = 0
                return False
            
            # Check if EMA order is maintained
            if self.trend_direction == 1 and self.ema_fast[0] < self.ema_slow[0]:
                self.trend_confirmed = False
                self.waiting_for_pullback = False
                self.trend_direction = 0
                return False
            elif self.trend_direction == -1 and self.ema_fast[0] > self.ema_slow[0]:
                self.trend_confirmed = False
                self.waiting_for_pullback = False
                self.trend_direction = 0
                return False
        
        return self.trend_confirmed

    def detect_pullback(self):
        """Detect valid pullback for entry"""
        if not self.waiting_for_pullback:
            return False
            
        current_price = self.data.close[0]
        
        # Update recent highs and lows
        if self.trend_direction == 1:  # Bullish trend
            if current_price > self.recent_high:
                self.recent_high = current_price
            
            # Check for pullback (price dipping from recent high)
            pullback_pct = (self.recent_high - current_price) / self.recent_high
            if pullback_pct >= self.p.min_pullback_pct:
                # Valid pullback detected, now wait for bounce
                self.pullback_detected = True
                self.pullback_extreme = current_price
                return True
                
        elif self.trend_direction == -1:  # Bearish trend
            if current_price < self.recent_low:
                self.recent_low = current_price
            
            # Check for pullback (price bouncing from recent low)
            pullback_pct = (current_price - self.recent_low) / self.recent_low
            if pullback_pct >= self.p.min_pullback_pct:
                # Valid pullback detected, now wait for reversal
                self.pullback_detected = True
                self.pullback_extreme = current_price
                return True
        
        return False

    def check_entry_conditions(self):
        """Check if conditions are right for entry after pullback"""
        if not self.pullback_detected:
            return False, None
            
        current_price = self.data.close[0]
        
        if self.trend_direction == 1:  # Bullish trend
            # Look for bounce from pullback
            # Entry when price moves back up from pullback extreme
            if current_price > self.pullback_extreme * 1.001:  # 0.1% bounce
                # Additional confirmations
                rsi_ok = self.rsi[0] > self.p.rsi_bull_threshold
                price_above_fast_ema = current_price > self.ema_fast[0] * 0.999  # Allow small tolerance
                adx_strong = self.adx[0] > self.p.adx_strong
                
                if rsi_ok and (price_above_fast_ema or adx_strong):
                    return True, 'long'
                    
        elif self.trend_direction == -1:  # Bearish trend
            # Look for reversal from pullback
            # Entry when price moves back down from pullback extreme
            if current_price < self.pullback_extreme * 0.999:  # 0.1% reversal
                # Additional confirmations
                rsi_ok = self.rsi[0] < self.p.rsi_bear_threshold
                price_below_fast_ema = current_price < self.ema_fast[0] * 1.001  # Allow small tolerance
                adx_strong = self.adx[0] > self.p.adx_strong
                
                if rsi_ok and (price_below_fast_ema or adx_strong):
                    return True, 'short'
        
        return False, None

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
            
        # Need minimum bars for all indicators
        min_bars = max(self.p.ema_slow, self.p.adx_period, self.p.rsi_period, self.p.atr_period)
        if len(self) < min_bars:
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
        
        # Step 1: Detect trend and strength
        trend_valid = self.detect_trend_and_strength()
        
        # Step 2: If trend is valid, detect pullback
        if trend_valid:
            self.detect_pullback()
        
        if not self.position:  # If no position is open
            # Step 3: Check for entry after pullback
            entry_signal, direction = self.check_entry_conditions()
            
            if entry_signal and direction:
                position_size = self.calculate_position_size(current_price)
                
                if position_size <= 0:
                    return
                
                # Calculate stop loss and take profit using ATR
                atr_value = self.atr[0]
                
                if direction == 'long':
                    # Stop loss: below pullback extreme or using ATR
                    stop_price = min(
                        self.pullback_extreme * (1 - self.p.stop_loss),
                        current_price - (atr_value * self.p.atr_stop_multiplier)
                    )
                    # Take profit: using ATR or fixed percentage
                    take_profit = max(
                        current_price * (1 + self.p.take_profit),
                        current_price + (atr_value * self.p.atr_target_multiplier)
                    )
                    
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
                    
                    # Reset pullback detection
                    self.waiting_for_pullback = False
                    self.pullback_detected = False
                    
                elif direction == 'short':
                    # Stop loss: above pullback extreme or using ATR
                    stop_price = max(
                        self.pullback_extreme * (1 + self.p.stop_loss),
                        current_price + (atr_value * self.p.atr_stop_multiplier)
                    )
                    # Take profit: using ATR or fixed percentage
                    take_profit = min(
                        current_price * (1 - self.p.take_profit),
                        current_price - (atr_value * self.p.atr_target_multiplier)
                    )
                    
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
                    
                    # Reset pullback detection
                    self.waiting_for_pullback = False
                    self.pullback_detected = False
        else:
            # Exit conditions for open positions
            bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
            pos = self.getposition()
            
            if pos.size > 0:  # Long position
                # Exit if trend weakens or time stop
                if (bars_held >= self.p.time_stop_bars or 
                    self.adx[0] < self.p.adx_threshold - 5 or
                    current_price < self.ema_fast[0]):
                    self.close()
                    self.closing = True
                    self.entry_bar = None
                    self.entry_side = None
                    
            elif pos.size < 0:  # Short position
                # Exit if trend weakens or time stop
                if (bars_held >= self.p.time_stop_bars or
                    self.adx[0] < self.p.adx_threshold - 5 or
                    current_price > self.ema_fast[0]):
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
        "ema_fast": kwargs.get("ema_fast", 20),
        "ema_slow": kwargs.get("ema_slow", 50),
        "adx_period": kwargs.get("adx_period", 14),
        "rsi_period": kwargs.get("rsi_period", 14),
        "adx_threshold": kwargs.get("adx_threshold", 25),
        "adx_strong": kwargs.get("adx_strong", 30),
        "rsi_bull_threshold": kwargs.get("rsi_bull_threshold", 50),
        "rsi_bear_threshold": kwargs.get("rsi_bear_threshold", 50),
        "stop_loss": kwargs.get("stop_loss", 0.005),
        "take_profit": kwargs.get("take_profit", 0.01),
        "atr_period": kwargs.get("atr_period", 14),
        "atr_stop_multiplier": kwargs.get("atr_stop_multiplier", 2.0),
        "atr_target_multiplier": kwargs.get("atr_target_multiplier", 2.0),
        "time_stop_bars": kwargs.get("time_stop_bars", 50),
        "pullback_bars": kwargs.get("pullback_bars", 5),
        "min_pullback_pct": kwargs.get("min_pullback_pct", 0.002),
    }
    cerebro.addstrategy(EMA_ADX_Strategy, **strategy_params)
    
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
        ema_fast=20,
        ema_slow=50,
        adx_period=14,
        rsi_period=14,
        adx_threshold=25,
        adx_strong=30,
        rsi_bull_threshold=50,
        rsi_bear_threshold=50,
        stop_loss=0.005,
        take_profit=0.01,
        atr_period=14,
        atr_stop_multiplier=2.0,
        atr_target_multiplier=2.0,
        time_stop_bars=50,
        pullback_bars=5,
        min_pullback_pct=0.002,
        leverage=leverage
    )
    
    log_result(
        strategy="EMA_ADX_TrendRider",
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
            pd.DataFrame(all_results).to_csv("ema_adx_backtest_results.csv", index=False)

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
