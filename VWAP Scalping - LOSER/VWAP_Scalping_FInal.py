import backtrader as bt
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
import math
import random
import json
from functools import partial
import traceback
from collections import deque

random.seed(42)
np.random.seed(42)


class VWAPScalping(bt.Strategy):
    params = (
        ("vwap_period", 16),
        ("rsi_period", 12),
        ("sar_step", 0.04861946623362669),
        ("sar_max", 0.29479654081926365),
        ("use_vwap", True),
        ("use_rsi", True),
        ("use_sar", True),
        ("take_profit", 0.02),
        ("stop_loss", 0.01),
    )

    def __init__(self):
        if self.p.use_vwap:
            # Calculate typical price more efficiently
            self.typical_price = (
                self.data.high + self.data.low + self.data.close
            ) / 3.0

            # Use MovingAverageSimple (SMA) with caching for cumulative calculations
            cumulative_tp_vol = self.typical_price * self.data.volume

            # Use built-in indicators for moving sums with automatic caching
            self.cum_tp_vol = bt.indicators.SumN(
                cumulative_tp_vol, period=self.p.vwap_period
            )
            self.cum_vol = bt.indicators.SumN(
                self.data.volume, period=self.p.vwap_period
            )

            # Calculate VWAP using the cached values
            self.vwap = self.cum_tp_vol / self.cum_vol

            # Use CrossOver indicator for signal generation
            self.crossover = bt.indicators.CrossOver(self.data.close, self.vwap)

        if self.p.use_rsi:
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        if self.p.use_sar:
            self.sar = bt.indicators.ParabolicSAR(
                self.data, af=self.p.sar_step, afmax=self.p.sar_max
            )

    def calculate_position_size(self, current_price):
        try:
            current_equity = self.broker.getvalue()

            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0

            leverage = self.broker.getcommissioninfo(self.data).get_leverage()

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0


    def next(self):
        if not self.p.use_vwap or len(self.vwap) < 1:
            return

        # Use cached crossover indicator instead of manual calculation
        long_conditions = [self.crossover > 0]
        short_conditions = [self.crossover < 0]
        if self.p.use_rsi:
            long_conditions.append(self.rsi[0] < 40)
            short_conditions.append(self.rsi[0] > 60)
        if self.p.use_sar:
            long_conditions.append(self.data.close[0] > self.sar[0])
            short_conditions.append(self.data.close[0] < self.sar[0])

        current_price = self.data.close[0]

        # Long Entry
        if all(long_conditions) and self.position.size <= 0:
            position_size = self.calculate_position_size(current_price)
            stop_loss = current_price - (current_price * self.p.stop_loss)
            take_profit = current_price + (current_price * self.p.take_profit)

            if position_size > 0:
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss,
                )

        # Short Entry
        elif all(short_conditions) and self.position.size >= 0:
            position_size = self.calculate_position_size(current_price)
            stop_loss = current_price + (current_price * self.p.stop_loss)
            take_profit = current_price - (current_price * self.p.take_profit)

            if position_size > 0:
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss,
                )

class TradeRecorder(bt.Analyzer):
    """Custom analyzer to record individual trade results for SQN calculation"""
    
    def __init__(self):
        super(TradeRecorder, self).__init__()
        self.trades = []
        self.current_trade = None

    def get_analysis(self):
        """Required method for Backtrader analyzers"""
        return self

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.trades.append({
                'pnl': trade.pnlcomm,  # PnL including commission
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'datetime': trade.data.datetime.datetime(0)
            })

def calculate_sqn(trades):
    """Calculate System Quality Number using individual trade data"""
    try:
        if not trades or len(trades) < 2:
            return 0.0
            
        pnl_list = [trade['pnl'] for trade in trades]
        avg_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list)
        
        if std_pnl == 0:
            return 0.0
            
        sqn = (avg_pnl / std_pnl) * math.sqrt(len(pnl_list))
        return max(min(sqn, 100), -100)  # Limit SQN to reasonable range
        
    except Exception as e:
        print(f"Error calculating SQN: {str(e)}")
        return 0.0


def run_backtest(data, plot=False, verbose=True, optimize=False, **kwargs):

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

    cerebro.addstrategy(
        VWAPScalping,
        vwap_period=kwargs.get("vwap_period", 20),
        rsi_period=kwargs.get("rsi_period", 14),
        use_rsi=kwargs.get("use_rsi", False),
        use_sar=kwargs.get("use_sar", False),
        sar_step=kwargs.get("sar_step", 0.02),
        sar_max=kwargs.get("sar_max", 0.2),
        take_profit=kwargs.get("take_profit", 0.02),
        stop_loss=kwargs.get("stop_loss", 0.01),
    )
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        # margin=1.0 / 50,
        leverage=50,
        commtype=bt.CommInfoBase.COMM_PERC
    )
    cerebro.broker.set_slippage_perc(0.0001)
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
    cerebro.addanalyzer(TradeRecorder, _name='trade_recorder')

    if plot:
        cerebro.addobserver(bt.observers.BuySell)  # Show buy/sell points
        cerebro.addobserver(bt.observers.Value)  # Show portfolio value
        cerebro.addobserver(bt.observers.DrawDown)  # Show drawdown

    # Run backtest
    results = cerebro.run()
    # Handle results (note that with parallel execution, results structure might be different)
    if len(results) > 0:
        if isinstance(results[0], (list, tuple)):
            strat = results[0][0]  # Get first strategy from first result set
        else:
            strat = results[0]
    else:
        raise ValueError("No results returned from backtest")


    try:
        trade_recorder = strat.analyzers.trade_recorder.get_analysis()
        sqn = calculate_sqn(trade_recorder.trades if hasattr(trade_recorder, 'trades') else [])
    except Exception as e:
        print(f"Error accessing trade recorder: {e}")
        print(f"full traceback: {traceback.format_exc()}")
        sqn = 0.0
    
    try:
        trades_analysis = strat.analyzers.trades.get_analysis()
    except Exception as e:
        print(f"Error accessing trades analysis: {e}")
        trades_analysis = {}

    try:
        drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    except Exception as e:
        print(f"Error accessing drawdown analysis: {e}")
        drawdown_analysis = {}

    # Calculate metrics with safe getters
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    # Safe getter function with better None handling
    def safe_get(d, *keys, default=0):
        try:
            for key in keys:
                if d is None:
                    return default
                d = d[key]
            return d if d is not None else default
        except (KeyError, AttributeError, TypeError):
            return default

    # Get Sharpe Ratio with proper None handling
    try:
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except:
        sharpe_ratio = 0.0

    # Format results with safe getters and explicit None handling
    formatted_results = {
        "Start": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "End": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "Duration": f"{(data.datetime.iloc[-1] - data.datetime.iloc[0]).days} days",
        "Exposure Time [%]": (
            safe_get(trades_analysis, "total", "total", default=0) / (len(data) / 60)
        )
        * 100,
        "Equity Final [$]": final_value,
        "Equity Peak [$]": final_value
        + (
            safe_get(drawdown_analysis, "max", "drawdown", default=0)
            * final_value
            / 100
        ),
        "Return [%]": total_return,
        "Buy & Hold Return [%]": (
            (data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100
        ),
        "Max. Drawdown [%]": safe_get(drawdown_analysis, "max", "drawdown", default=0),
        "Avg. Drawdown [%]": safe_get(
            drawdown_analysis, "average", "drawdown", default=0
        ),
        "Max. Drawdown Duration": safe_get(drawdown_analysis, "max", "len", default=0),
        "Avg. Drawdown Duration": safe_get(
            drawdown_analysis, "average", "len", default=0
        ),
        "# Trades": safe_get(trades_analysis, "total", "total", default=0),
        "Win Rate [%]": (
            safe_get(trades_analysis, "won", "total", default=0)
            / max(safe_get(trades_analysis, "total", "total", default=1), 1)
            * 100
        ),
        "Best Trade [%]": safe_get(trades_analysis, "won", "pnl", "max", default=0),
        "Worst Trade [%]": safe_get(trades_analysis, "lost", "pnl", "min", default=0),
        "Avg. Trade [%]": safe_get(trades_analysis, "pnl", "net", "average", default=0),
        "Max. Trade Duration": safe_get(trades_analysis, "len", "max", default=0),
        "Avg. Trade Duration": safe_get(trades_analysis, "len", "average", default=0),
        "Profit Factor": (
            safe_get(trades_analysis, "won", "pnl", "total", default=0)
            / max(abs(safe_get(trades_analysis, "lost", "pnl", "total", default=1)), 1)
        ),
        "Sharpe Ratio": float(sharpe_ratio),  # Ensure it's a float
        "SQN": sqn,  # ,
        # '_trades': strat._trades
    }

    # Print detailed statistics only if verbose is True
    if verbose:
        print("\n=== Strategy Performance Report ===")
        print(
            f"\nPeriod: {formatted_results['Start']} - {formatted_results['End']} ({formatted_results['Duration']})"
        )
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"Final Capital: ${float(formatted_results['Equity Final [$]']):,.2f}")
        print(f"Total Return: {float(formatted_results['Return [%]']):,.2f}%")
        print(
            f"Buy & Hold Return: {float(formatted_results['Buy & Hold Return [%]']):,.2f}%"
        )
        print(f"\nTotal Trades: {int(formatted_results['# Trades'])}")
        print(f"Win Rate: {float(formatted_results['Win Rate [%]']):,.2f}%")
        print(f"Best Trade: {float(formatted_results['Best Trade [%]']):,.2f}%")
        print(f"Worst Trade: {float(formatted_results['Worst Trade [%]']):,.2f}%")
        print(f"Avg. Trade: {float(formatted_results['Avg. Trade [%]']):,.2f}%")
        print(f"\nMax Drawdown: {float(formatted_results['Max. Drawdown [%]']):,.2f}%")
        print(f"Sharpe Ratio: {float(formatted_results['Sharpe Ratio']):,.2f}")
        print(f"Profit Factor: {float(formatted_results['Profit Factor']):,.2f}")
        print(f"SQN: {float(formatted_results['SQN']):,.2f}")

    return formatted_results

if __name__ == "__main__":
    data_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20240915-to-20241114.csv"
    # Load data with parallel processing
    print("Loading and preprocessing data...")
    data_df = pd.read_csv(data_path)
    data_df.dropna(inplace=True)
    columns = ["datetime", "Open", "High", "Low", "Close", "Volume"]
    data_df = data_df[columns]
    data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    # Ensure data is sorted by date/time if not already
    data_df = data_df.sort_values(by="datetime").reset_index(drop=True)

    # Determine the split index for 50%
    split_index = int(len(data_df) * 0.5)

    # Split the data into training and testing sets
    train_df = data_df.iloc[:split_index].reset_index(drop=True)
    test_df = data_df.iloc[split_index:].reset_index(drop=True)

    # fixed_params = {
    #     "vwap_period": 16,
    #     "rsi_period": 12,
    #     "use_rsi": True,
    #     "use_macd": False,
    #     "use_bb": False,
    #     "bb_period": 16,
    #     "bb_std_dev": 2.2574098289906153,
    #     "macd_fast": 12,
    #     "macd_slow": 36,
    #     "macd_signal": 5,
    #     "use_atr": False,
    #     "atr_period": 18,
    #     "use_sar": True,
    #     "sar_step": 0.04861946623362669,
    #     "sar_max": 0.29479654081926365,
    #     "take_profit": 0.02,
    #     "stop_loss": 0.01,
    # }


    print("Running backtest...")
    results_fixed = run_backtest(
        data_df, plot=False, verbose=True, optimize=False
    )
    print("\nBacktest results with fixed parameters:")
    print(results_fixed)