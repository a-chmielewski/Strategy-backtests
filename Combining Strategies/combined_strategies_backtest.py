import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import random
from deap import base, creator, tools, algorithms
import multiprocessing
from functools import partial
import math
import traceback

# Import strategies from their respective files
from bollinger import BBStrategy
from EMA_BB_pSAR_RSI import EMA_BB_PSAR_RSI
from double_EMA_StochOsc import DoubleEMA_StochOsc_Strategy
from mean_reversion import VolatilityMeanReversionStrategy

class StrategyRunner:
    def __init__(self):
        self.strategies = {
            'Bollinger': {
                'class': BBStrategy,
                'symbol': 'SOLUSDT',
                'timeframe': '5m',
                'params': {}  # Add any specific parameters here
            },
            'EMA_BB_PSAR_RSI': {
                'class': EMA_BB_PSAR_RSI,
                'symbol': 'BTCUSDT',
                'timeframe': '5m',
                'params': {}
            },
            'Double_EMA_StochOsc': {
                'class': DoubleEMA_StochOsc_Strategy,
                'symbol': 'ETHUSDT',
                'timeframe': '5m',
                'params': {}
            },
            'Mean_Reversion': {
                'class': VolatilityMeanReversionStrategy,
                'symbol': 'XRPUSDT',
                'timeframe': '5m',
                'params': {}
            }
        }
        
    def run_combined_backtest(self, data_dict, initial_cash=400.0, plot=False, verbose=True):
        """Run all strategies simultaneously with shared capital"""
        cerebro = bt.Cerebro()
        
        # Add data feeds for each symbol and store them in a dict
        data_feeds = {}
        for symbol, data in data_dict.items():
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
            # Add data with name to be able to reference it
            cerebro.adddata(feed, name=symbol)
            data_feeds[symbol] = feed
        
        # Add strategies with their corresponding data feeds
        for strategy_name, strategy_info in self.strategies.items():
            symbol = strategy_info['symbol']
            if symbol in data_feeds:
                # Create a strategy-specific class that uses the correct data feed
                strategy_class = strategy_info['class']
                
                # Create a subclass that will use the correct data feed
                class StrategyWithData(strategy_class):
                    def __init__(self):
                        super().__init__()
                        # Use the data feed with matching symbol
                        self.data = self.getdatabyname(symbol)
                
                # Add the modified strategy
                cerebro.addstrategy(
                    StrategyWithData,
                    **strategy_info['params']
                )
        
        # Set up broker with shared capital
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(
            commission=0.0002,
            commtype=bt.CommInfoBase.COMM_PERC,
            margin=1.0/10  # For 10x leverage
        )
        cerebro.broker.set_slippage_perc(0.0001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
        
        if plot:
            cerebro.addobserver(bt.observers.BuySell)
            cerebro.addobserver(bt.observers.Value)
            cerebro.addobserver(bt.observers.DrawDown)
        
        # Run backtest
        results = cerebro.run()
        strat = results[0]
        
        # Calculate combined metrics
        final_value = cerebro.broker.getvalue()
        formatted_results = self.format_results(strat, None, initial_cash, final_value)
        formatted_results['strategy'] = 'Combined Strategies'
        
        if verbose:
            self.print_combined_results(formatted_results)
        
        return formatted_results

    def format_results(self, strat, data, initial_cash, final_value):
        # Safe getter function
        def safe_get(d, *keys, default=0):
            try:
                for key in keys:
                    if d is None:
                        return default
                    d = d[key]
                return d if d is not None else default
            except (KeyError, AttributeError, TypeError):
                return default

        trades_analysis = strat.analyzers.trades.get_analysis()
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
        
        return {
            "Initial Capital": initial_cash,
            "Final Capital": final_value,
            "Total Return %": ((final_value - initial_cash) / initial_cash) * 100,
            "# Trades": safe_get(trades_analysis, "total", "total"),
            "Win Rate %": (
                safe_get(trades_analysis, "won", "total", default=0)
                / max(safe_get(trades_analysis, "total", "total", default=1), 1)
                * 100
            ),
            "Max Drawdown %": safe_get(drawdown_analysis, "max", "drawdown"),
            "Sharpe Ratio": getattr(strat.analyzers.sharpe.get_analysis(), 'sharperatio', 0.0),
            "Profit Factor": (
                safe_get(trades_analysis, "won", "pnl", "total", default=0)
                / max(abs(safe_get(trades_analysis, "lost", "pnl", "total", default=1)), 1)
            )
        }

    def print_combined_results(self, results):
        print("\n=== Combined Strategies Performance ===")
        print(f"Initial Capital: ${results['Initial Capital']:.2f}")
        print(f"Final Capital: ${results['Final Capital']:.2f}")
        print(f"Total Return: {results['Total Return %']:.2f}%")
        print(f"Number of Trades: {results['# Trades']}")
        print(f"Win Rate: {results['Win Rate %']:.2f}%")
        print(f"Max Drawdown: {results['Max Drawdown %']:.2f}%")
        print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
        print(f"Profit Factor: {results['Profit Factor']:.2f}")
        print("\nNote: All strategies are running simultaneously with shared capital")

def main():
    # Data paths
    data_paths = {
        'SOLUSDT': r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-SOLUSDT-5m-20240929-to-20241128.csv",
        'BTCUSDT': r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv",
        'ETHUSDT': r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ETHUSDT-5m-20240929-to-20241128.csv",
        'XRPUSDT': r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-XRPUSDT-5m-20240929-to-20241128.csv"
    }
    
    # Initialize runner
    runner = StrategyRunner()
    
    try:
        # Load all data
        data_dict = {}
        for symbol, path in data_paths.items():
            data = pd.read_csv(path)
            data['datetime'] = pd.to_datetime(data['datetime'])
            data_dict[symbol] = data
        
        # Run combined backtest
        initial_cash = 100.0
        results = runner.run_combined_backtest(data_dict, initial_cash=initial_cash, verbose=True)
        
    except Exception as e:
        print(f"Error running combined backtest: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 