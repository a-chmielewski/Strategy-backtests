import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import os
import concurrent.futures

class TimeFilteredTrendStrategy(bt.Strategy):
    params = (
        ("ema_fast_period", 20),
        ("ema_slow_period", 50),
        ("rsi_period", 14),
        ("rsi_overbought", 70),
        ("rsi_oversold", 30),
        ("atr_period", 14),
        ("stop_loss", 0.02),
        ("take_profit", 0.04),
        ("trailing_atr", 0.01),
        ("trading_hours", [14, 15, 16]),
        ("trading_days", [0, 1, 2]),
    )

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

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
            leverage = 50
            position_size = (position_value * leverage) / current_price
            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def next(self):
        if not self.is_trading_window():
            return
        current_price = self.data.close[0]
        if not self.position:
            position_size = self.calculate_position_size(current_price)
            stop_loss_distance = self.atr[0] * self.p.stop_loss
            take_profit_distance = self.atr[0] * self.p.take_profit
            trail_amount = self.atr[0] * self.p.trailing_atr
            if current_price > self.ema_fast[0] and self.ema_fast[0] > self.ema_slow[0]:
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=current_price - stop_loss_distance,
                    limitprice=current_price + take_profit_distance,
                    trailamount=trail_amount
                )
            elif current_price < self.ema_fast[0] and self.ema_fast[0] < self.ema_slow[0]:
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=current_price + stop_loss_distance,
                    limitprice=current_price - take_profit_distance,
                    trailamount=trail_amount
                )

def calculate_sqn(trades):
    try:
        if not trades or len(trades) < 2:
            return 0.0
        pnl_list = [trade['pnl'] for trade in trades]
        avg_pnl = np.mean(pnl_list)
        std_pnl = np.std(pnl_list)
        if std_pnl == 0:
            return 0.0
        sqn = (avg_pnl / std_pnl) * math.sqrt(len(pnl_list))
        return max(min(sqn, 100), -100)
    except Exception as e:
        print(f"Error calculating SQN: {str(e)}")
        return 0.0

class TradeRecorder(bt.Analyzer):
    def __init__(self):
        super(TradeRecorder, self).__init__()
        self.active_trades = {}
        self.trades = []
    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            trade_value = abs(trade.price * trade.size)
            self.active_trades[trade.ref] = {
                'entry_time': len(self.strategy),
                'entry_bar_datetime': self.strategy.datetime.datetime(),
                'entry_price': trade.price,
                'size': abs(trade.size),
                'value': trade_value
            }
        if trade.status == trade.Closed:
            entry_data = self.active_trades.pop(trade.ref, None)
            if entry_data is not None:
                entry_time = entry_data['entry_time']
                exit_time = len(self.strategy)
                bars_held = exit_time - entry_time
                exit_price = trade.price
                self.trades.append({
                    'datetime': self.strategy.datetime.datetime(),
                    'type': 'long' if trade.size > 0 else 'short',
                    'size': entry_data['size'],
                    'price': exit_price,
                    'value': entry_data['value'],
                    'pnl': float(trade.pnl),
                    'pnlcomm': float(trade.pnlcomm),
                    'commission': float(trade.commission),
                    'entry_price': entry_data['entry_price'],
                    'exit_price': exit_price,
                    'bars_held': bars_held
                })
    def get_analysis(self):
        return self.trades

class DetailedDrawdownAnalyzer(bt.Analyzer):
    def __init__(self):
        super(DetailedDrawdownAnalyzer, self).__init__()
        self.drawdowns = []
        self.current_drawdown = None
        self.peak = 0
        self.equity_curve = []
    def next(self):
        value = self.strategy.broker.getvalue()
        self.equity_curve.append(value)
        if value > self.peak:
            self.peak = value
            if self.current_drawdown is not None:
                self.drawdowns.append(self.current_drawdown)
                self.current_drawdown = None
        elif value < self.peak:
            dd_pct = (self.peak - value) / self.peak * 100
            if self.current_drawdown is None:
                self.current_drawdown = {'start': len(self), 'peak': self.peak, 'lowest': value, 'dd_pct': dd_pct}
            elif value < self.current_drawdown['lowest']:
                self.current_drawdown['lowest'] = value
                self.current_drawdown['dd_pct'] = dd_pct
    def stop(self):
        if self.current_drawdown is not None:
            self.drawdowns.append(self.current_drawdown)
    def get_analysis(self):
        if not self.drawdowns:
            return {'max_drawdown': 0, 'avg_drawdown': 0, 'drawdowns': []}
        max_dd = max(dd['dd_pct'] for dd in self.drawdowns)
        avg_dd = sum(dd['dd_pct'] for dd in self.drawdowns) / len(self.drawdowns)
        return {
            'drawdowns': self.drawdowns,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd
        }

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
        "rsi_period": kwargs.get("rsi_period", 14),
        "rsi_overbought": kwargs.get("rsi_overbought", 70),
        "rsi_oversold": kwargs.get("rsi_oversold", 30),
        "atr_period": kwargs.get("atr_period", 14),
        "stop_loss": kwargs.get("stop_loss", 0.02),
        "take_profit": kwargs.get("take_profit", 0.04),
        "trailing_atr": kwargs.get("trailing_atr", 0.01),
        "trading_hours": kwargs.get("trading_hours", [14, 15, 16]),
        "trading_days": kwargs.get("trading_days", [0, 1, 2]),
    }
    cerebro.addstrategy(TimeFilteredTrendStrategy, **strategy_params)
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
    except:
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
    filename, data_folder = args
    data_path = os.path.join(data_folder, filename)
    try:
        parts = filename.split('-')
        symbol = parts[1]
        timeframe = parts[2]
        print(f"\nTesting {symbol} {timeframe}...")
        data_df = pd.read_csv(data_path)
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
        results = run_backtest(
            data_df,
            verbose=False,
            symbol=symbol,
            timeframe=timeframe,
            data_source="Bybit",
            ema_fast_period=20,
            ema_slow_period=50,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            atr_period=14,
            stop_loss=0.02,
            take_profit=0.04,
            trailing_atr=0.01,
            trading_hours=[14, 15, 16],
            trading_days=[0, 1, 2]
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
    except Exception as e:
        print(f"Error processing {data_path}: {str(e)}")
        print(traceback.format_exc())
        return (None, filename)

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
        if all_results:
            pd.DataFrame(all_results).to_csv("partial_backtest_results.csv", index=False)
    except Exception as e:
        print("\nException occurred in main execution:")
        print(str(e))
        print(traceback.format_exc())
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