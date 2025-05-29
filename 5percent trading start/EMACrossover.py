import backtrader as bt
import pandas as pd
import traceback
import os
import concurrent.futures
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from results_logger import log_result
from analyzers import TradeRecorder, DetailedDrawdownAnalyzer, SQNAnalyzer

LEVERAGE = 1  # Default, will be overwritten in main

class EMACrossStrategy(bt.Strategy):
    """Base template for creating trading strategies"""
    
    params = (
        ("ema_short", 9),
        ("ema_long", 80),
        ("stochastic_period", 12),
        ("stochastic_pfast", 5),
        ("stochastic_pslow", 3),
        ("stop_loss", 0.01),
        ("take_profit", 0.02),
    )

    def __init__(self):
        """Initialize strategy components"""
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.stochastic = bt.indicators.Stochastic(
            self.data,
            period=self.p.stochastic_period,
            period_dfast=self.p.stochastic_pfast,
            period_dslow=self.p.stochastic_pslow,
            movav=bt.indicators.MovAv.Simple
        )
        self.atr = bt.indicators.ATR(self.data, period=14)

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
        if len(self) < max(self.p.ema_long, self.p.stochastic_period) + 1:
            return
        prev_ema_short = self.ema_short[-1]
        prev_ema_long = self.ema_long[-1]
        curr_ema_short = self.ema_short[0]
        curr_ema_long = self.ema_long[0]
        golden_cross_event = prev_ema_short <= prev_ema_long and curr_ema_short > curr_ema_long
        death_cross_event = prev_ema_short >= prev_ema_long and curr_ema_short < curr_ema_long
        # ATR-based price/EMA proximity filter
        atr = self.atr[0]
        price_near_ema = (
            abs(self.data.close[0] - curr_ema_short) < atr or
            abs(self.data.close[0] - curr_ema_long) < atr
        )
        stoch_above_20 = self.stochastic.lines.percK[0] > 20
        stoch_below_80 = self.stochastic.lines.percK[0] < 80
        stoch_prev_above_80 = self.stochastic.lines.percK[-1] > 80
        stoch_prev_below_20 = self.stochastic.lines.percK[-1] < 20
        position_size = self.calculate_position_size(self.data.close[0])
        current_price = self.data.close[0]

        # Stochastic entry crossovers
        stoch_entry_long = stoch_prev_below_20 and stoch_above_20
        stoch_entry_short = stoch_prev_above_80 and stoch_below_80

        # Exit logic for open positions
        if self.position:
            if self.position.size > 0:
                # Long: exit on death cross or Stoch reversal
                if death_cross_event or (stoch_prev_above_80 and stoch_below_80):
                    self.close()
                    return
            elif self.position.size < 0:
                # Short: exit on golden cross or Stoch reversal
                if golden_cross_event or (stoch_prev_below_20 and stoch_above_20):
                    self.close()
                    return

        # Entry logic if flat
        if not self.position:
            if golden_cross_event and price_near_ema and stoch_entry_long:
                stop_loss = current_price * (1 - self.p.stop_loss)
                take_profit = current_price * (1 + self.p.take_profit)
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_loss,
                    limitprice=take_profit,
                    stopexec=bt.Order.Stop,
                    limitexec=bt.Order.Limit,
                )
            elif death_cross_event and price_near_ema and stoch_entry_short:
                stop_loss = current_price * (1 + self.p.stop_loss)
                take_profit = current_price * (1 - self.p.take_profit)
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_loss,
                    limitprice=take_profit,
                    stopexec=bt.Order.Stop,
                    limitexec=bt.Order.Limit,
                )

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
        "ema_short": kwargs.get("ema_short", 9),
        "ema_long": kwargs.get("ema_long", 80),
        "stochastic_period": kwargs.get("stochastic_period", 12),
        "stochastic_pfast": kwargs.get("stochastic_pfast", 5),
        "stochastic_pslow": kwargs.get("stochastic_pslow", 3),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.01),
    }
    cerebro.addstrategy(EMACrossStrategy, **strategy_params)
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
        ema_long=80,
        stochastic_period=12,
        stochastic_pfast=5,
        stochastic_pslow=3,
        stop_loss=0.01,
        take_profit=0.02
    )
    log_result(
        strategy="EMACrossStrategy",
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
            # Print top 3 for this leverage
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
        # Optionally write partial results to disk
        if all_results:
            pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
    except Exception as e:
        print("\nException occurred in main execution:")
        print(str(e))
        print(traceback.format_exc())
        # Print whatever results were collected so far
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
                pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
        except Exception as e2:
            print("\nError printing partial results:")
            print(str(e2))
            print(traceback.format_exc())