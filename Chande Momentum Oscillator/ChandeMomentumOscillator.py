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

class ChandeMomentumOscillator(bt.Indicator):
    """
    Chande Momentum Oscillator Implementation
    Formula: CMO = 100 * ((Su - Sd)/(Su + Sd))
    where:
    Su = Sum of higher closes over n periods
    Sd = Sum of lower closes over n periods
    """
    lines = ('cmo',)  # Define the lines (output) of the indicator
    params = (('period', 14),)  # Default period is 14

    def __init__(self):
        super(ChandeMomentumOscillator, self).__init__()
        
        # Calculate daily price changes
        self.data_diff = self.data - self.data(-1)
        
        # Separate positive and negative price changes
        self.up = bt.If(self.data_diff > 0, self.data_diff, 0.0)
        self.down = bt.If(self.data_diff < 0, -self.data_diff, 0.0)
        
        # Calculate rolling sums
        self.up_sum = bt.indicators.SumN(self.up, period=self.params.period)
        self.down_sum = bt.indicators.SumN(self.down, period=self.params.period)
        
        # Calculate CMO
        epsilon = 1e-8
        self.lines.cmo = 100 * (self.up_sum - self.down_sum) / (self.up_sum + self.down_sum + epsilon)

class ChandeMomentumOscillatorStrategy(bt.Strategy):
    params = (
        ("period", 14),
        ("overbought", 50),    # Overbought level
        ("oversold", -50),     # Oversold level
        ("stop_loss", 0.01),   # Static 1% stop loss
        ("take_profit", 0.02), # Static 1% take profit
        ("ma_period", 50),
        ("cmo_slope_atr_mult", 1.5),  # ATR multiplier for CMO slope threshold
        ("time_stop_bars", 30),       # Time stop in bars
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []  # To track ongoing trades for visualization
        
        # Initialize indicators
        self.cmo = ChandeMomentumOscillator(self.data, period=self.params.period)
        self.prev_cmo = 0
        self.was_oversold = False
        self.was_overbought = False
        self.ma = bt.indicators.SMA(self.data.close, period=self.params.ma_period)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.order = None
        self.trade_dir = {}  # Track trade direction by trade.ref
        self.entry_bar = None
        self.entry_side = None

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
        if self.order:
            return
        # Seed prev_cmo after period bars, before MA warm-up
        if len(self) == self.p.period:
            self.prev_cmo = self.cmo[0]
        if len(self) < self.p.ma_period:
            return
        is_bullish = self.data.close[0] > self.ma[0] * 1.002
        is_bearish = self.data.close[0] < self.ma[0] * 0.998
        cmo_direction = self.cmo[0] - self.prev_cmo
        self.prev_cmo = self.cmo[0]
        # ATR-based CMO slope threshold
        try:
            atr_percent = (self.atr[0] / self.data.close[0]) * 100 if self.data.close[0] != 0 else 0
            cmo_slope_threshold = self.p.cmo_slope_atr_mult * atr_percent
        except Exception as e:
            print(f"Error calculating ATR-based CMO slope threshold: {str(e)}")
            cmo_slope_threshold = 2.0
        if self.getposition().size == 0:
            position_size = self.calculate_position_size(self.data.close[0])
            if position_size <= 0:
                return
            if self.cmo[0] < self.p.oversold:
                self.was_oversold = True
            if self.was_oversold and cmo_direction > cmo_slope_threshold and self.cmo[0] > self.p.oversold and is_bullish:
                self.was_oversold = False
                stop_loss = self.data.close[0] * (1 - self.p.stop_loss)
                take_profit = self.data.close[0] * (1 + self.p.take_profit)
                parent, stop, limit = self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss
                )
                self.order = parent  # Track only the parent order
                self.entry_bar = len(self)
                self.entry_side = 'long'
            if self.cmo[0] > self.p.overbought:
                self.was_overbought = True
            if self.was_overbought and cmo_direction < -cmo_slope_threshold and self.cmo[0] < self.p.overbought and is_bearish:
                self.was_overbought = False
                stop_loss = self.data.close[0] * (1 + self.p.stop_loss)
                take_profit = self.data.close[0] * (1 - self.p.take_profit)
                parent, stop, limit = self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss
                )
                self.order = parent  # Track only the parent order
        else:
            # Active management: time-stop and MA cross exit
            bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
            pos = self.getposition()
            if pos.size > 0:
                # Long position
                if bars_held >= self.p.time_stop_bars or self.data.close[0] < self.ma[0]:
                    self.close()
                    self.entry_bar = None
                    self.entry_side = None
            elif pos.size < 0:
                # Short position
                if bars_held >= self.p.time_stop_bars or self.data.close[0] > self.ma[0]:
                    self.close()
                    self.entry_bar = None
                    self.entry_side = None

    def notify_trade(self, trade):
        if trade.isopen and trade.justopened:
            self.trade_dir[trade.ref] = 'long' if trade.size > 0 else 'short'
        if not trade.isclosed:
            return
        try:
            if trade.history:
                entry_price = trade.price
                exit_price = trade.history[-1].price
            else:
                entry_price = trade.price
                exit_price = trade.price
            pnl = trade.pnl
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
        if self.order and order.ref == self.order.ref and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None
        if order.status == order.Completed:
            if not order.parent:  # This is an entry order
                self.active_trades.append({
                    'entry_time': self.data.datetime.datetime(0),
                    'entry_price': order.executed.price,
                    'type': 'long' if order.isbuy() else 'short',
                    'size': order.executed.size,
                    'exit_orders': []
                })
            else:  # This is an exit order
                if self.active_trades:
                    trade = self.active_trades[-1]
                    trade['exit_orders'].append({
                        'exit_time': self.data.datetime.datetime(0),
                        'exit_price': order.executed.price,
                        'size': order.executed.size
                    })
                    total_exit_size = sum(exit_order['size'] for exit_order in trade['exit_orders'])
                    if abs(total_exit_size) >= abs(trade['size']):
                        trade = self.active_trades.pop()
                        for exit_order in trade['exit_orders']:
                            self.trade_exits.append({
                                'entry_time': trade['entry_time'],
                                'entry_price': trade['entry_price'],
                                'exit_time': exit_order['exit_time'],
                                'exit_price': exit_order['exit_price'],
                                'type': f"{trade['type']}_exit",
                                'pnl': (exit_order['exit_price'] - trade['entry_price']) * exit_order['size'] if trade['type'] == 'long' 
                                      else (trade['entry_price'] - exit_order['exit_price']) * exit_order['size']
                            })

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

    # Extract strategy parameters from kwargs or use defaults
    strategy_params = {
        "period": kwargs.get("period", 14),
        "overbought": kwargs.get("overbought", 50),
        "oversold": kwargs.get("oversold", -50),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.02),
        "ma_period": kwargs.get("ma_period", 50),
        "cmo_slope_atr_mult": kwargs.get("cmo_slope_atr_mult", 1.5),
        "time_stop_bars": kwargs.get("time_stop_bars", 30),
    }

    # Add strategy with parameters
    cerebro.addstrategy(ChandeMomentumOscillatorStrategy, **strategy_params)
    
    initial_cash = 100.0
    leverage = LEVERAGE  # Default leverage
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        margin=1.0 / leverage,
        commtype=bt.CommInfoBase.COMM_PERC
    )
    
    # Update analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(DetailedDrawdownAnalyzer, _name="detailed_drawdown")
    cerebro.addanalyzer(TradeRecorder, _name='trade_recorder')
    cerebro.addanalyzer(SQNAnalyzer, _name='sqn')

    if verbose:
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

    # Calculate metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100

    # Get Sharpe Ratio
    try:
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except:
        sharpe_ratio = 0.0

    profit_factor = (win_trades['pnl'].sum() / abs(loss_trades['pnl'].sum())) if not loss_trades.empty else 0
    try:
        sqn = strat.analyzers.sqn.get_analysis()['sqn']
    except (AttributeError, KeyError):
        sqn = 0.0

    # Format results
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

    # Print detailed statistics only if verbose is True
    if verbose:
        print("\n=== Strategy Performance Report ===")
        print(
            f"\nPeriod: {formatted_results['Start']} - {formatted_results['End']} ({formatted_results['Duration']})"
        )
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
        print(f"\nTesting {symbol} {timeframe}...")
        data_df = pd.read_csv(data_path)
        data_df["datetime"] = pd.to_datetime(data_df["datetime"])
        results = run_backtest(
            data_df,
            verbose=False,
            period=14,
            overbought=50,
            oversold=-50,
            stop_loss=0.01,
            take_profit=0.01,
            ma_period=50,
            cmo_slope_atr_mult=1.5,
            time_stop_bars=30
        )
        log_result(
            strategy="ChandeMomentumOscillator",
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
            pd.DataFrame(all_results).to_csv("partial_cmo_results.csv", index=False)
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
                pd.DataFrame(all_results).to_csv("partial_cmo_results.csv", index=False)
        except Exception as e2:
            print("\nError printing partial results:")
            print(str(e2))
            print(traceback.format_exc()) 