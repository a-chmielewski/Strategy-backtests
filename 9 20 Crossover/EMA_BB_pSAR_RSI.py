import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import math
import traceback
from pathlib import Path
import json
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from results_logger import log_result
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer

LEVERAGE = 50

class EMA_BB_PSAR_RSI(bt.Strategy):
    """Base template for creating trading strategies"""

    params = (
        ("ema_short", 20),
        ("ema_long", 50),
        ("sar_step", 0.02), 
        ("sar_max", 0.2),
        ("rsi_period", 14),
        ("rsi_overbought", 70),
        ("rsi_oversold", 30),
        ("bb_period", 21),
        ("bb_devfactor", 2.0),
        ("stop_loss", 0.01),
        ("take_profit", 0.02),
        ("ema_gap_threshold", 0.0005),
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize indicators
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.sar = bt.indicators.ParabolicSAR(self.data, af=self.p.sar_step, afmax=self.p.sar_max)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.p.bb_period, devfactor=self.p.bb_devfactor)
        self.bb_upper = self.bb.top
        self.bb_lower = self.bb.bot

    def calculate_position_size(self, current_price):
        try:
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
        min_bars = max(self.p.ema_long, self.p.bb_period, self.p.rsi_period) + 1
        if len(self) < min_bars:
            return
            
        if self.getposition().size == 0:
            position_size = self.calculate_position_size(self.data.close[0])
            
            # Skip if size is too small
            if position_size <= 0:
                return
                
            ema_gap = abs(self.ema_short[0] - self.ema_long[0])
            price = self.data.close[0]
            gap_ok = ema_gap > (self.p.ema_gap_threshold * price)
            
            # Group conditions for better readability
            trend_buy = (self.ema_short > self.ema_long and self.data.close[0] > self.sar[0])
            momentum_buy = (
                self.rsi[0] <= self.p.rsi_oversold and
                self.data.close[0] < self.bb_lower[0] and
                self.data.close[-1] > self.bb_lower[-1]
            )
            trend_sell = (self.ema_short < self.ema_long and self.data.close[0] < self.sar[0])
            momentum_sell = (
                self.rsi[0] >= self.p.rsi_overbought and
                self.data.close[0] > self.bb_upper[0] and
                self.data.close[-1] < self.bb_upper[-1]
            )
                
            # BUY if either trend or momentum signal and EMA gap is sufficient
            if (trend_buy or momentum_buy) and gap_ok:
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=price * (1 + self.p.take_profit),
                    stopprice=price * (1 - self.p.stop_loss)
                )
                
            # SELL if either trend or momentum signal and EMA gap is sufficient
            elif (trend_sell or momentum_sell) and gap_ok:
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=price * (1 - self.p.take_profit),
                    stopprice=price * (1 + self.p.stop_loss)
                )

        # Manual exit on EMA trend reversal
        if self.position:
            if self.position.size > 0 and self.ema_short < self.ema_long:
                self.close()
                return
            elif self.position.size < 0 and self.ema_short > self.ema_long:
                self.close()
                return

def run_backtest(data, verbose=True, **kwargs):
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
        "ema_short": kwargs.get("ema_short", 9),
        "ema_long": kwargs.get("ema_long", 21),
        "sar_step": kwargs.get("sar_step", 0.02),
        "sar_max": kwargs.get("sar_max", 0.2),
        "rsi_period": kwargs.get("rsi_period", 14),
        "rsi_overbought": kwargs.get("rsi_overbought", 60),
        "rsi_oversold": kwargs.get("rsi_oversold", 40),
        "bb_period": kwargs.get("bb_period", 21),
        "bb_devfactor": kwargs.get("bb_devfactor", 2.0),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.01)
    }
    cerebro.addstrategy(EMA_BB_PSAR_RSI, **strategy_params)
    initial_cash = 100.0
    leverage = LEVERAGE
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        margin=1.0 / leverage,
        commtype=bt.CommInfoBase.COMM_PERC
    )
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
        ema_short=9,
        ema_long=21,
        sar_step=0.02,
        sar_max=0.2,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        bb_period=21,
        bb_devfactor=2.0,
        stop_loss=0.01,
        take_profit=0.02
    )
    log_result(
        strategy="EMA_BB_PSAR_RSI",
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
    try:
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_folder = os.path.abspath(data_folder)
        files = [f for f in os.listdir(data_folder) if f.startswith('bybit-') and f.endswith('.csv')]
        all_results = []
        failed_files = []
        leverages = [1, 5, 10, 15, 25, 50]
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
        if failed_files:
            print("\nThe following files failed to process:")
            for fname, lev in failed_files:
                print(f"- {fname} (leverage {lev})")
        if all_results:
            pd.DataFrame(all_results).to_csv("partial_ema_bb_psar_rsi_results.csv", index=False)
    except Exception as e:
        print("\nException occurred in main execution:")
        print(str(e))
        print(traceback.format_exc())
        try:
            for leverage in [1, 5, 10, 15, 25, 50]:
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
                pd.DataFrame(all_results).to_csv("partial_ema_bb_psar_rsi_results.csv", index=False)
        except Exception as e2:
            print("\nError printing partial results:")
            print(str(e2))
            print(traceback.format_exc()) 