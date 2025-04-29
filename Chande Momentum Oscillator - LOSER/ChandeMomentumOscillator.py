import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import concurrent.futures
import os

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
        ("take_profit", 0.01), # Static 1% take profit
        ("ma_period", 50),
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

    def calculate_position_size(self, current_price):
        try:
            current_equity = self.broker.getvalue()

            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0

            leverage = 50

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0

    def next(self):
        """Define trading logic with trend filter"""
        if len(self.data) < max(2, self.params.ma_period):
            return

        # Determine the trend: bullish if price above MA, bearish otherwise
        is_bullish = self.data.close[0] > self.ma[0]
        is_bearish = self.data.close[0] < self.ma[0]

        # Calculate CMO direction
        cmo_direction = self.cmo[0] - self.prev_cmo
        self.prev_cmo = self.cmo[0]

        if not self.position:
            position_size = self.calculate_position_size(self.data.close[0])
            if position_size <= 0:
                return

            # Buy signal: CMO moves up after being oversold and trend is bullish
            if self.cmo[0] < self.params.oversold:
                self.was_oversold = True

            if self.was_oversold and cmo_direction > 0 and self.cmo[0] > self.params.oversold and is_bullish:
                self.was_oversold = False
                # Use static 1% levels for stop loss and take profit
                stop_loss = self.data.close[0] * (1 - self.params.stop_loss)
                take_profit = self.data.close[0] * (1 + self.params.take_profit)
                
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss
                )

        else:
            # Sell signal: CMO moves down after being overbought and trend is bearish
            if self.cmo[0] > self.params.overbought:
                self.was_overbought = True

            if self.was_overbought and cmo_direction < 0 and self.cmo[0] < self.params.overbought and is_bearish:
                self.was_overbought = False
                self.close()
                position_size = self.calculate_position_size(self.data.close[0])
                if position_size > 0:
                    # Use static 1% levels for stop loss and take profit
                    stop_loss = self.data.close[0] * (1 + self.params.stop_loss)
                    take_profit = self.data.close[0] * (1 - self.params.take_profit)
                    
                    self.sell_bracket(
                        size=position_size,
                        exectype=bt.Order.Market,
                        limitprice=take_profit,
                        stopprice=stop_loss
                    )

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        try:
            # Get entry and exit prices
            entry_price = trade.price
            exit_price = trade.history[-1].price if trade.history else self.data.close[0]
            pnl = trade.pnl
            
            # Store trade exit information for visualization
            self.trade_exits.append({
                'entry_time': trade.dtopen,  # Use trade's open datetime
                'exit_time': trade.dtclose,  # Use trade's close datetime
                'entry_price': entry_price,
                'exit_price': exit_price,
                'type': 'long_exit' if trade.size > 0 else 'short_exit',
                'pnl': pnl
            })
            
        except Exception as e:
            print(f"Warning: Could not process trade: {str(e)}")
            print(f"Trade info - Status: {trade.status}, Size: {trade.size}, "
                  f"Price: {trade.price}, PnL: {trade.pnl}")

    def notify_order(self, order):
        if order.status == order.Completed:
            if not order.parent:  # This is an entry order
                # Record trade start
                self.active_trades.append({
                    'entry_time': self.data.datetime.datetime(0),
                    'entry_price': order.executed.price,
                    'type': 'long' if order.isbuy() else 'short',
                    'size': order.executed.size,
                    'exit_orders': []  # Track multiple exit orders
                })
            else:  # This is an exit order
                if self.active_trades:
                    trade = self.active_trades[-1]  # Get current trade without removing it
                    trade['exit_orders'].append({
                        'exit_time': self.data.datetime.datetime(0),
                        'exit_price': order.executed.price,
                        'size': order.executed.size
                    })
                    
                    # If all size is closed, record the complete trade
                    total_exit_size = sum(exit_order['size'] for exit_order in trade['exit_orders'])
                    if abs(total_exit_size) >= abs(trade['size']):
                        trade = self.active_trades.pop()  # Now remove the trade
                        # Record each exit
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

class TradeRecorder(bt.Analyzer):
    """Custom analyzer to record individual trade results"""
    
    def __init__(self):
        super(TradeRecorder, self).__init__()
        self.active_trades = {}  # Holds data for open trades by trade.ref
        self.trades = []         # Holds final results for closed trades

    def notify_trade(self, trade):
        """Called by Backtrader when a trade is updated"""
        
        # 1) Trade Just Opened
        if trade.isopen and trade.justopened:
            # Compute approximate "value" = entry_price * size
            trade_value = abs(trade.price * trade.size)
            
            self.active_trades[trade.ref] = {
                'entry_time': len(self.strategy),
                'entry_bar_datetime': self.strategy.datetime.datetime(),
                'entry_price': trade.price,
                'size': abs(trade.size),
                'value': trade_value
            }

        # 2) Trade Closed
        if trade.status == trade.Closed:
            # Retrieve entry details
            entry_data = self.active_trades.pop(trade.ref, None)
            
            if entry_data is not None:
                # Calculate bars_held
                entry_time = entry_data['entry_time']
                exit_time = len(self.strategy)
                bars_held = exit_time - entry_time

                # Store final trade record
                self.trades.append({
                    'datetime': self.strategy.datetime.datetime(),
                    'type': 'long' if trade.size > 0 else 'short',
                    'size': entry_data['size'],
                    'price': trade.price,
                    'value': entry_data['value'],
                    'pnl': float(trade.pnl),
                    'pnlcomm': float(trade.pnlcomm),
                    'commission': float(trade.commission),
                    'entry_price': entry_data['entry_price'],
                    'exit_price': trade.price,
                    'bars_held': bars_held
                })

    def get_analysis(self):
        """Required method for Backtrader analyzers"""
        return self.trades


class DetailedDrawdownAnalyzer(bt.Analyzer):
    """Analyzer providing detailed drawdown statistics"""
    
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
        """Called when backtesting is finished"""
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

    # Extract strategy parameters from kwargs or use defaults
    strategy_params = {
        "period": kwargs.get("period", 14),
        "overbought": kwargs.get("overbought", 50),
        "oversold": kwargs.get("oversold", -50),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.01),
        "ma_period": kwargs.get("ma_period", 50)
    }

    # Add strategy with parameters
    cerebro.addstrategy(ChandeMomentumOscillatorStrategy, **strategy_params)
    
    initial_cash = 100.0
    leverage = 50  # Default leverage
    
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


    # Calculate metrics with safe getters
    try:
        trade_recorder = strat.analyzers.trade_recorder.get_analysis()
        sqn = calculate_sqn(trade_recorder if hasattr(trade_recorder, 'trades') else [])
    except Exception as e:
        print(f"Error accessing trade recorder: {e}")
        sqn = 0.0
    
    try:
        trades_analysis = strat.analyzers.trades.get_analysis()
    except Exception as e:
        print(f"Error accessing trades analysis: {e}")
        trades_analysis = {}

    try:
        drawdown_analysis = strat.analyzers.detailed_drawdown.get_analysis()
    except Exception as e:
        print(f"Error accessing drawdown analysis: {e}")
        drawdown_analysis = {}

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

    # Format results
    formatted_results = {
        "Start": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "End": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "Duration": f"{(data.datetime.iloc[-1] - data.datetime.iloc[0]).days} days",
        "Exposure Time [%]": (safe_get(trades_analysis, "total", "total", default=0) / len(data)) * 100,
        "Equity Final [$]": final_value,
        "Equity Peak [$]": final_value + (safe_get(drawdown_analysis, "max_drawdown", default=0) * final_value / 100),
        "Return [%]": total_return,
        "Buy & Hold Return [%]": ((data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100),
        "Max. Drawdown [%]": safe_get(drawdown_analysis, "max_drawdown", default=0),
        "Avg. Drawdown [%]": safe_get(drawdown_analysis, "avg_drawdown", default=0),
        "Max. Drawdown Duration": safe_get(drawdown_analysis, "max", "len", default=0),
        "Avg. Drawdown Duration": safe_get(drawdown_analysis, "average", "len", default=0),
        "# Trades": safe_get(trades_analysis, "total", "total", default=0),
        "Win Rate [%]": (safe_get(trades_analysis, "won", "total", default=0) / 
                        max(safe_get(trades_analysis, "total", "total", default=1), 1) * 100),
        "Best Trade [%]": safe_get(trades_analysis, "won", "pnl", "max", default=0),
        "Worst Trade [%]": safe_get(trades_analysis, "lost", "pnl", "min", default=0),
        "Avg. Trade [%]": safe_get(trades_analysis, "pnl", "net", "average", default=0),
        "Max. Trade Duration": safe_get(trades_analysis, "len", "max", default=0),
        "Avg. Trade Duration": safe_get(trades_analysis, "len", "average", default=0),
        "Profit Factor": (safe_get(trades_analysis, "won", "pnl", "total", default=0) / 
                         max(abs(safe_get(trades_analysis, "lost", "pnl", "total", default=1)), 1)),
        "Sharpe Ratio": float(sharpe_ratio),
        "SQN": sqn
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
            period=14,
            overbought=50,
            oversold=-50,
            stop_loss=0.01,
            take_profit=0.01,
            ma_period=50
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