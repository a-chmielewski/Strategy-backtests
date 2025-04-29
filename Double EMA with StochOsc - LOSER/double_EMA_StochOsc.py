import backtrader as bt
import pandas as pd
import numpy as np
import math
import traceback
import os
import concurrent.futures

class DoubleEMA_StochOsc_Strategy(bt.Strategy):
    params = (
        ("ema_slow", 50),
        ("ema_fast", 150),
        ("stoch_k", 5),
        ("stoch_d", 3),
        ("slowing", 3),
        ("stop_loss", 0.01),  # 1% stop loss
        ("take_profit", 0.01), # 1% take profit
        ("stoch_overbought", 80),
        ("stoch_oversold", 20),
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
        
        # Store previous values for crossover detection
        self.prev_stoch_k = 0
        self.prev_stoch_d = 0
        
        # Track swing highs and lows
        self.swing_high = self.data.high[0]
        self.swing_low = self.data.low[0]

    def calculate_position_size(self, current_price):
        try:
            if current_price is None or current_price == 0:
                return 0
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
        """Define trading logic"""
        if len(self.data) < 2:  # Need at least 2 bars
            return
            
        current_price = self.data.close[0]
        if current_price is None or current_price == 0:
            return
            
        # Update swing points
        self.swing_high = max(self.data.high[-1], self.data.high[0])
        self.swing_low = min(self.data.low[-1], self.data.low[0])
        
        # Calculate if stochastic is crossing
        stoch_k_direction = self.stoch.percK[0] - self.prev_stoch_k
        stoch_d_direction = self.stoch.percD[0] - self.prev_stoch_d
        
        # Store current values for next iteration
        self.prev_stoch_k = self.stoch.percK[0]
        self.prev_stoch_d = self.stoch.percD[0]
        
        if not self.position:  # If no position is open
            position_size = self.calculate_position_size(current_price)
            
            # Skip if size is too small
            if position_size <= 0:
                return
                
            # Buy conditions
            ema_uptrend = self.ema_slow[0] > self.ema_fast[0]
            price_at_ema = (current_price <= self.ema_fast[0] * 1.001 and 
                           current_price >= self.ema_fast[0] * 0.999)  # Within 0.1% of EMA
            stoch_oversold = (self.stoch.percK[0] <= self.p.stoch_oversold and 
                            self.stoch.percD[0] <= self.p.stoch_oversold)
            stoch_crossing_up = stoch_k_direction > 0 and stoch_d_direction > 0
            
            # Sell conditions
            ema_downtrend = self.ema_slow[0] < self.ema_fast[0]
            stoch_overbought = (self.stoch.percK[0] >= self.p.stoch_overbought and 
                              self.stoch.percD[0] >= self.p.stoch_overbought)
            stoch_crossing_down = stoch_k_direction < 0 and stoch_d_direction < 0
            
            if ema_uptrend and price_at_ema and stoch_oversold and stoch_crossing_up:
                # Buy with bracket order (stop loss at swing low)
                stop_price = self.swing_low * (1 - self.p.stop_loss)
                take_profit = current_price * (1 + self.p.take_profit)
                
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_price,
                    limitprice=take_profit,
                )
                
            elif ema_downtrend and price_at_ema and stoch_overbought and stoch_crossing_down:
                # Sell with bracket order (stop loss at swing high)
                stop_price = self.swing_high * (1 + self.p.stop_loss)
                take_profit = current_price * (1 - self.p.take_profit)
                
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_price,
                    limitprice=take_profit,
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
                'datetime': self.data.datetime.datetime(0),
                'price': exit_price,
                'type': 'long_exit' if trade.size > 0 else 'short_exit',
                'pnl': pnl,
                'entry_price': entry_price
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
        "ema_slow": kwargs.get("ema_slow", 50),
        "ema_fast": kwargs.get("ema_fast", 150),
        "stoch_k": kwargs.get("stoch_k", 5),
        "stoch_d": kwargs.get("stoch_d", 3),
        "slowing": kwargs.get("slowing", 3),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.01),
        "stoch_overbought": kwargs.get("stoch_overbought", 80),
        "stoch_oversold": kwargs.get("stoch_oversold", 20)
    }

    # Add strategy with parameters
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
    results = cerebro.run()
    if len(results) > 0:
        strat = results[0][0] if isinstance(results[0], (list, tuple)) else results[0]
    else:
        raise ValueError("No results returned from backtest")
    # Vectorized trade metrics
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
            ema_slow=50,
            ema_fast=150,
            stoch_k=5,
            stoch_d=3,
            slowing=3,
            stop_loss=0.01,
            take_profit=0.01,
            stoch_overbought=80,
            stoch_oversold=20
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