import backtrader as bt
import pandas as pd
import traceback
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer
from results_logger import log_result

LEVERAGE = 50

class TimeFilteredTrendStrategy(bt.Strategy):
    params = (
        ("ema_fast_period", 20),
        ("ema_slow_period", 50),
        ("atr_period", 14),
        ("stop_loss", 0.01),
        ("take_profit", 0.02),
        ("trailing_atr", 0.01),
        ("trading_hours", [14, 15, 16]),
        ("trading_days", [0, 1, 2]),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.order = None

    def is_trading_window(self):
        current_time = self.data.datetime.datetime(0)
        if current_time.hour not in self.p.trading_hours:
            return False
        if current_time.weekday() not in self.p.trading_days:
            return False
        return True

    def calculate_position_size(self, current_price):
        try:
            current_equity = self.broker.getvalue()
            position_value = current_equity if current_equity < 100 else 100.0
            leverage = LEVERAGE
            position_size = (position_value * leverage) / current_price
            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def next(self):
        if len(self) < self.p.ema_slow_period:
            return
        current_price = self.data.close[0]
        # Exit on EMA cross back (always, even outside trading window)
        if self.position:
            if (self.position.size > 0 and self.ema_fast[0] < self.ema_slow[0]) or (self.position.size < 0 and self.ema_fast[0] > self.ema_slow[0]):
                self.close()
                return
        if not self.is_trading_window():
            return
        if not self.position:
            position_size = self.calculate_position_size(current_price)
            stop_loss_distance = 0.01 * current_price
            take_profit_distance = 0.02 * current_price
            trail_amount = self.atr[0] * self.p.trailing_atr
            if current_price > self.ema_fast[0] and self.ema_fast[0] > self.ema_slow[0]:
                parent, stop, limit = self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=current_price + take_profit_distance,
                    trailamount=trail_amount
                )
                self.order = parent
            elif current_price < self.ema_fast[0] and self.ema_fast[0] < self.ema_slow[0]:
                parent, stop, limit = self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=current_price - take_profit_distance,
                    trailamount=trail_amount
                )
                self.order = parent

    def notify_order(self, order):
        if self.order and order.ref == self.order.ref and \
        order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

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
        "ema_fast_period": kwargs.get("ema_fast_period", 20),
        "ema_slow_period": kwargs.get("ema_slow_period", 50),
        "atr_period": kwargs.get("atr_period", 14),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.02),
        "trailing_atr": kwargs.get("trailing_atr", 0.01),
        "trading_hours": kwargs.get("trading_hours", [14, 15, 16]),
        "trading_days": kwargs.get("trading_days", [0, 1, 2]),
    }
    cerebro.addstrategy(TimeFilteredTrendStrategy, **strategy_params)
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
        ema_fast_period=20,
        ema_slow_period=50,
        atr_period=14,
        stop_loss=0.01,
        take_profit=0.02,
        trailing_atr=0.01,
        trading_hours=[14, 15, 16],
        trading_days=[0, 1, 2]
    )
    log_result(
            strategy="TimeFilteredTrendStrategy",
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
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
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
            pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
    except Exception as e:
        print("\nException occurred during processing:")
        print(str(e))
        print(traceback.format_exc())
        if all_results:
            try:
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
                pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
            except Exception as e2:
                print("\nError printing partial results:")
                print(str(e2))
                print(traceback.format_exc()) 