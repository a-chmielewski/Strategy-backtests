import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import concurrent.futures
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from results_logger import log_result
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer

LEVERAGE = 50

class BBStrategy(bt.Strategy):
    """Base template for creating trading strategies"""
    
    params = (
        # Add strategy parameters here
        ("period", 21),
        ("devfactor", 2.0),
        ("stop_k", 1.0),
        ("target_k", 2.0),
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize indicators
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.period, devfactor=self.params.devfactor)
        self.bb_upper = self.bollinger.top
        self.bb_lower = self.bollinger.bot
        self.bb_mid = self.bollinger.mid
        self.entry_bar = None

    def calculate_position_size(self, current_price):
        current_equity = self.broker.getvalue()
        if current_equity < 100:
            position_value = current_equity
        else:
            position_value = 100.0
        leverage = LEVERAGE
        try:
            position_size = (position_value * leverage) / current_price
        except ZeroDivisionError:
            print("Error in calculate_position_size: Division by zero")
            return 0
        return position_size

    def next(self):
        """Define trading logic"""
        # Only trade if enough data for slope
        if len(self) < self.params.period + 1:
            return
        # Exit logic: close if price touches mid-band or after 10 bars
        if self.position:
            if self.entry_bar is not None:
                bars_held = len(self) - self.entry_bar
                if (self.position.size > 0 and self.data.close[0] >= self.bb_mid[0]) or \
                   (self.position.size < 0 and self.data.close[0] <= self.bb_mid[0]) or \
                   (bars_held >= 10):
                    self.close()
                    self.entry_bar = None
                    return
        if not self.position:  # If we have no position
            position_size = self.calculate_position_size(self.data.close[0])
            # Calculate midline slope (current - previous)
            mid_slope = self.bb_mid[0] - self.bb_mid[-5]
            band_width = self.bb_upper[0] - self.bb_lower[0]
            stop_k = self.params.stop_k
            target_k = self.params.target_k
            # Fade lower band only if previous close was inside and current close is below band, and midline slope is flat or rising
            if self.data.close[-1] >= self.bb_lower[-1] and self.data.close[0] < self.bb_lower[0] and mid_slope >= 0:
                stop_loss = self.data.close[0] - stop_k * band_width
                take_profit = self.data.close[0] + target_k * band_width
                self.buy_bracket(size=position_size, exectype=bt.Order.Market,
                                 stopprice=stop_loss,
                                 limitprice=take_profit)
                self.entry_bar = len(self)
            # Fade upper band only if previous close was inside and current close is above band, and midline slope is flat or falling
            elif self.data.close[-1] <= self.bb_upper[-1] and self.data.close[0] > self.bb_upper[0] and mid_slope <= 0:
                stop_loss = self.data.close[0] + stop_k * band_width
                take_profit = self.data.close[0] - target_k * band_width
                self.sell_bracket(size=position_size, exectype=bt.Order.Market, 
                                stopprice=stop_loss,
                                limitprice=take_profit)
                self.entry_bar = len(self)
        

def run_backtest(data, verbose=True, **kwargs):
    """Run backtest with given parameters"""
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
        "period": kwargs.get("period", 21),
        "devfactor": kwargs.get("devfactor", 2.0),
        "stop_k": kwargs.get("stop_k", 1.0),
        "target_k": kwargs.get("target_k", 2.0),
    }
    cerebro.addstrategy(BBStrategy, **strategy_params)
    initial_cash = 100.0
    leverage = LEVERAGE
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
    # Vectorized trade metrics
    trades = strat.analyzers.trade_recorder.get_analysis()
    if trades:
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = pd.DataFrame()
    total_trades = len(trades_df)
    if not trades_df.empty:
        win_trades = trades_df[trades_df['pnl'] > 0]
        loss_trades = trades_df[trades_df['pnl'] < 0]
        winrate = (len(win_trades) / total_trades * 100) if total_trades > 0 else 0
        avg_trade = trades_df['pnl'].mean()
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()
    else:
        win_trades = pd.DataFrame()
        loss_trades = pd.DataFrame()
        winrate = 0
        avg_trade = 0
        best_trade = 0
        worst_trade = 0
    max_drawdown = 0
    avg_drawdown = 0
    try:
        dd = strat.analyzers.detailed_drawdown.get_analysis()
        max_drawdown = dd.get('max_drawdown', 0)
        avg_drawdown = dd.get('avg_drawdown', 0)
    except (AttributeError, KeyError) as e:
        print(f"Error accessing drawdown analysis: {e}")
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    try:
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except (AttributeError, KeyError):
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
        print(f"SQN: {float(formatted_results['SQN']):.2f}")
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
        period=21,
        devfactor=2.0,
        stop_k=1.0,
        target_k=2.0
    )
    log_result(
        strategy="BBStrategy",
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
    try:
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_folder = os.path.abspath(data_folder)
        files = [f for f in os.listdir(data_folder) if f.startswith('bybit-') and f.endswith('.csv')]
        all_results = []
        failed_files = []
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
        # Optionally write partial results to disk
        if all_results:
            pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
    except Exception as e:
        print("\nException occurred in main execution:")
        print(str(e))
        print(traceback.format_exc())
        # Print whatever results were collected so far
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
            if all_results:
                pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
        except Exception as e2:
            print("\nError printing partial results:")
            print(str(e2))
            print(traceback.format_exc())