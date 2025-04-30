import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer
from results_logger import log_result

LEVERAGE = 50

class DoubleEMA_StochOsc_Strategy(bt.Strategy):
    params = (
        ("ema_slow", 150),
        ("ema_fast", 50),
        ("stoch_k", 5),
        ("stoch_d", 3),
        ("slowing", 3),
        ("stop_loss", 0.01),
        ("take_profit", 0.02),
        ("stoch_overbought", 80),
        ("stoch_oversold", 20),
        ("time_stop_bars", 30),
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []  # To track ongoing trades for visualization
        
        # Initialize indicators
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.stoch = bt.indicators.StochasticFull(
            self.data,
            period=self.p.stoch_k,
            period_dfast=self.p.stoch_d,
            period_dslow=self.p.slowing,
            movav=bt.indicators.MovAv.Simple
        )
        
        # Use built-in CrossOver indicator for stochastic cross
        self.stoch_x = bt.ind.CrossOver(self.stoch.percK, self.stoch.percD)
        
        # Track running swing highs and lows since last entry
        self.swing_high = -float('inf')
        self.swing_low = float('inf')
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
            

    def next(self):
        """Define trading logic"""
        if self.order or getattr(self, 'order_parent_ref', None) is not None or getattr(self, 'closing', False):
            return
        if len(self) < max(self.p.ema_slow, self.p.ema_fast, self.p.stoch_k, self.p.stoch_d, self.p.slowing):
            return
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
            
        # Initialize swings with the first bar's high/low if needed
        if self.swing_high == -float('inf') and self.swing_low == float('inf'):
            self.swing_high = self.data.high[0]
            self.swing_low = self.data.low[0]
        
        # Update swing points
        self.swing_high = max(self.swing_high, self.data.high[0])
        self.swing_low = min(self.swing_low, self.data.low[0])
        
        # Use built-in CrossOver for stochastic cross
        stoch_crossing_up = self.stoch_x[0] > 0
        stoch_crossing_down = self.stoch_x[0] < 0
        
        if not self.position:  # If no position is open
            position_size = self.calculate_position_size(current_price)
            
            # Skip if size is too small
            if position_size <= 0:
                return
                
            # Buy conditions
            ema_uptrend = self.ema_fast[0] > self.ema_slow[0]
            price_near_fast_ema = abs(current_price - self.ema_fast[0]) < 0.003 * current_price
            price_at_ema = price_near_fast_ema  # Only require proximity to fast EMA
            stoch_oversold = self.stoch.percK[-1] < self.p.stoch_oversold and self.stoch.percK[0] > self.stoch.percD[0]
            
            # Sell conditions
            ema_downtrend = self.ema_fast[0] < self.ema_slow[0]
            stoch_overbought = self.stoch.percK[-1] > self.p.stoch_overbought and self.stoch.percK[0] < self.stoch.percD[0]
            
            if ema_uptrend and price_at_ema and stoch_oversold and stoch_crossing_up:
                # Buy with bracket order (stop loss at swing low)
                stop_price = self.swing_low * (1 - self.p.stop_loss)
                take_profit = current_price * (1 + self.p.take_profit)
                
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
                # Reset swing high/low after entry
                self.swing_high = -float('inf')
                self.swing_low = float('inf')
                
            elif ema_downtrend and price_at_ema and stoch_overbought and stoch_crossing_down:
                # Sell with bracket order (stop loss at swing high)
                stop_price = self.swing_high * (1 + self.p.stop_loss)
                take_profit = current_price * (1 - self.p.take_profit)
                
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
                # Reset swing high/low after entry
                self.swing_high = -float('inf')
                self.swing_low = float('inf')
        else:
            bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
            pos = self.getposition()
            if pos.size > 0:
                if bars_held >= self.p.time_stop_bars or current_price < self.ema_fast[0]:
                    self.close()
                    self.closing = True
                    self.entry_bar = None
                    self.entry_side = None
            elif pos.size < 0:
                if bars_held >= self.p.time_stop_bars or current_price > self.ema_fast[0]:
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

            # Reset swing high/low after a position closes
            self.swing_high, self.swing_low = -float('inf'), float('inf')
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

def run_backtest(data, verbose=True, **kwargs):
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
        "ema_slow": kwargs.get("ema_slow", 50),
        "ema_fast": kwargs.get("ema_fast", 150),
        "stoch_k": kwargs.get("stoch_k", 5),
        "stoch_d": kwargs.get("stoch_d", 3),
        "slowing": kwargs.get("slowing", 3),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.02),
        "stoch_overbought": kwargs.get("stoch_overbought", 80),
        "stoch_oversold": kwargs.get("stoch_oversold", 20),
        "time_stop_bars": kwargs.get("time_stop_bars", 30),
    }
    cerebro.addstrategy(DoubleEMA_StochOsc_Strategy, **strategy_params)
    initial_cash = 100.0
    leverage = 50
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
    filename, data_folder = args
    data_path = os.path.join(data_folder, filename)
    try:
        parts = filename.split('-')
        symbol = parts[1]
        timeframe = parts[2]
    except (IndexError, ValueError) as e:
        print(f"Error parsing filename {filename}: {str(e)}")
        return (None, filename)
    print(f"\nTesting {symbol} {timeframe}...")
    try:
        data_df = pd.read_csv(data_path)
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    except (IOError, ValueError) as e:
        print(f"Error reading or parsing data for {filename}: {str(e)}")
        return (None, filename)
    results = run_backtest(
        data_df,
        verbose=False,
        ema_slow=50,
        ema_fast=150,
        stoch_k=5,
        stoch_d=3,
        slowing=3,
        stop_loss=0.01,
        take_profit=0.02,
        stoch_overbought=80,
        stoch_oversold=20,
        time_stop_bars=30
    )
    log_result(
        strategy="DoubleEMA_StochOsc",
        coinpair=symbol,
        timeframe=timeframe,
        leverage=LEVERAGE,
        results=results
    )
    summary = {
        'symbol': symbol,
        'timeframe': timeframe,
        'winrate': results.get('Win Rate [%]', 0),
        'final_equity': results.get('Equity Final [$]', 0),
        'total_trades': results.get('# Trades', 0),
        'max_drawdown': results.get('Max. Drawdown [%]', 0)
    }
    return (summary, filename)

if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_folder = os.path.abspath(data_folder)
    try:
        files = [f for f in os.listdir(data_folder) if f.startswith('bybit-') and f.endswith('.csv')]
    except (OSError, IOError) as e:
        print(f"Error listing files in {data_folder}: {str(e)}")
        files = []
    all_results = []
    failed_files = []
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_file, [(f, data_folder) for f in files]))
            for summary, fname in results:
                if summary is not None:
                    all_results.append(summary)
                else:
                    failed_files.append(fname)
        sorted_results = sorted(all_results, key=lambda x: x['winrate'], reverse=True)[:3]
        print("\n=== Top 3 Results by Win Rate ===")
        for i, result in enumerate(sorted_results, 1):
            print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
            print(f"Win Rate: {result['winrate']:.2f}%")
            print(f"Total Trades: {result['total_trades']}")
            print(f"Final Equity: {result['final_equity']}")
            print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
        if failed_files:
            print("\nThe following files failed to process:")
            for fname in failed_files:
                print(f"- {fname}")
        if all_results:
            pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
    except Exception as e:
        print("\nException occurred during processing:")
        print(str(e))
        print(traceback.format_exc())
        if all_results:
            try:
                sorted_results = sorted(all_results, key=lambda x: x['winrate'], reverse=True)[:3]
                print("\n=== Top 3 Results by Win Rate (Partial) ===")
                for i, result in enumerate(sorted_results, 1):
                    print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
                    print(f"Win Rate: {result['winrate']:.2f}%")
                    print(f"Total Trades: {result['total_trades']}")
                    print(f"Final Equity: {result['final_equity']}")
                    print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
                if failed_files:
                    print("\nThe following files failed to process:")
                    for fname in failed_files:
                        print(f"- {fname}")
                pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc())