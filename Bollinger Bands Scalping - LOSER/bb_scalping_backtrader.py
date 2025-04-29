import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import math
import traceback
from pathlib import Path
import json
import concurrent.futures
import os

warnings.filterwarnings("ignore", category=FutureWarning)

class BollingerBandsStrategy(bt.Strategy):
    """Strategy implementing Bollinger Bands with additional indicators"""
    
    params = (
        # Core parameters
        ('n', 20),  # BB period
        ('ndev', 2.0),  # BB standard deviation
        ('bb_touch_threshold', 0.002),  # How close price needs to be to bands
        
        # Risk management
        ('sl_pct', 0.01),  # Stop loss percentage
        ('tp_pct', 0.01),  # Take profit percentage
        ('min_size', 1.0),  # Minimum position size
        
        # Indicator parameters
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        
        # Feature flags
        ('use_rsi', False),
        ('use_macd', True),
        ('use_candlestick', True),
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trades_list = []
        self.long_signals = 0
        self.short_signals = 0
        
        # Initialize indicators
        self.bb = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.p.n,
            devfactor=self.p.ndev
        )
        self.bb_top = self.bb.lines.top
        self.bb_bot = self.bb.lines.bot
        self.bb_mid = self.bb.lines.mid
        
        # Initialize RSI if used
        if self.p.use_rsi:
            self.rsi = bt.indicators.RSI(
                self.data.close,
                period=self.p.rsi_period
            )
        
        # Initialize MACD if used
        if self.p.use_macd:
            self.macd = bt.indicators.MACD(
                self.data.close,
                period_me1=self.p.macd_fast,
                period_me2=self.p.macd_slow,
                period_signal=self.p.macd_signal
            )
            self.macd_signal = self.macd.signal
            self.macd_hist = self.macd.macd - self.macd.signal

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
        """Define trading logic"""
        if self.position:
            return
            
        current_close = self.data.close[0]
        
        # Calculate distances to bands
        price_distance_to_lower = (current_close - self.bb_bot[0]) / self.bb_bot[0]
        price_distance_to_upper = (self.bb_top[0] - current_close) / self.bb_top[0]
        
        # Check band touches
        touching_lower = price_distance_to_lower <= self.p.bb_touch_threshold
        touching_upper = price_distance_to_upper <= self.p.bb_touch_threshold
        
        # Long entry conditions
        if touching_lower:
            long_condition = True
            
            if self.p.use_macd:
                long_condition = long_condition and self._check_macd_condition(True)
                
            if self.p.use_rsi:
                long_condition = long_condition and self._check_rsi_condition(True)
                
            if self.p.use_candlestick:
                long_condition = long_condition and self._is_hammer()
            
            if long_condition:
                self._execute_long_trade(current_close)
                
        # Short entry conditions
        elif touching_upper:
            short_condition = True
            
            if self.p.use_macd:
                short_condition = short_condition and self._check_macd_condition(False)
                
            if self.p.use_rsi:
                short_condition = short_condition and self._check_rsi_condition(False)
                
            if self.p.use_candlestick:
                short_condition = short_condition and self._is_shooting_star()
            
            if short_condition:
                self._execute_short_trade(current_close)

    def _execute_long_trade(self, current_close):
        """Execute long trade with bracket orders"""
        stop_loss = current_close * (1 - self.p.sl_pct)
        take_profit = current_close * (1 + self.p.tp_pct)
        
        size = self.calculate_position_size(current_close)
        
        self.buy_bracket(
            size=size,
            exectype=bt.Order.Market,
            stopprice=stop_loss,
            limitprice=take_profit,
        )
        
        self.long_signals += 1

    def _execute_short_trade(self, current_close):
        """Execute short trade with bracket orders"""
        stop_loss = current_close * (1 + self.p.sl_pct)
        take_profit = current_close * (1 - self.p.tp_pct)
        
        size = self.calculate_position_size(current_close)
        
        self.sell_bracket(
            size=size,
            exectype=bt.Order.Market,
            stopprice=stop_loss,
            limitprice=take_profit,
        )
        
        self.short_signals += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades_list.append({
                'pnl': trade.pnlcomm,
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'datetime': trade.data.datetime.datetime(0)
            })

    def _check_macd_condition(self, is_long):
        """
        Check MACD conditions for trade entry
        Args:
            is_long (bool): True for long trades, False for short trades
        """
        if is_long:
            # For long trades, check if MACD histogram is positive and increasing
            return (self.macd_hist[0] > self.macd_hist[-1] > 0 and 
                   self.macd.lines.macd[0] > self.macd.lines.signal[0])
        else:
            # For short trades, check if MACD histogram is negative and decreasing
            return (self.macd_hist[0] < self.macd_hist[-1] < 0 and 
                   self.macd.lines.macd[0] < self.macd.lines.signal[0])

    def _check_rsi_condition(self, is_long):
        """
        Check RSI conditions for trade entry
        Args:
            is_long (bool): True for long trades, False for short trades
        """
        if is_long:
            return self.rsi[0] < self.p.rsi_oversold
        else:
            return self.rsi[0] > self.p.rsi_overbought

    def _is_hammer(self):
        """Check if current candle is a hammer pattern (bullish)"""
        body = abs(self.data.close[0] - self.data.open[0])
        lower_wick = min(self.data.open[0], self.data.close[0]) - self.data.low[0]
        upper_wick = self.data.high[0] - max(self.data.open[0], self.data.close[0])
        
        # Hammer criteria
        return (lower_wick > 2 * body and  # Lower wick at least 2x body
                upper_wick < body and      # Small or no upper wick
                body > 0)                  # Real body exists

    def _is_shooting_star(self):
        """Check if current candle is a shooting star pattern (bearish)"""
        body = abs(self.data.close[0] - self.data.open[0])
        lower_wick = min(self.data.open[0], self.data.close[0]) - self.data.low[0]
        upper_wick = self.data.high[0] - max(self.data.open[0], self.data.close[0])
        
        # Shooting star criteria
        return (upper_wick > 2 * body and  # Upper wick at least 2x body
                lower_wick < body and      # Small or no lower wick
                body > 0)                  # Real body exists

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


class StrategyDataCollector:
    """Collects and stores comprehensive strategy backtest data"""
    
    def __init__(self, base_path="strategy_database/BB_Scalping"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_backtest_results(self, strategy_name, results, strategy_params, data_info, trades_data, 
                            initial_capital=100.0, leverage=10):
        """Save comprehensive backtest results to JSON file"""
        
        # Convert trades_data to list if it's not already
        trades_data = list(trades_data) if trades_data else []
        
        # Process trades
        processed_trades = []
        for trade in trades_data:
            processed_trade = {
                'datetime': trade['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'type': trade['type'],
                'size': float(trade['size']),
                'price': float(trade['price']),
                'value': float(trade['value']),
                'pnl': float(trade['pnl']),
                'pnlcomm': float(trade['pnlcomm']),
                'commission': float(trade['commission']),
                'entry_price': float(trade['entry_price']),
                'exit_price': float(trade['exit_price']),
                'bars_held': int(trade['bars_held'])
            }
            processed_trades.append(processed_trade)
        
        strategy_data = {
            "timestamp": self.current_timestamp,
            "strategy_name": strategy_name,
            "strategy_type": "Mean Reversion",
            "strategy_description": "Bollinger Bands Scalping Strategy with RSI and MACD Filters",
            "data_info": data_info,
            "parameters": {
                "n": strategy_params.get("n", 20),
                "ndev": strategy_params.get("ndev", 2.0),
                "bb_touch_threshold": strategy_params.get("bb_touch_threshold", 0.002),
                "sl_pct": strategy_params.get("sl_pct", 0.01),
                "tp_pct": strategy_params.get("tp_pct", 0.01),
                "rsi_period": strategy_params.get("rsi_period", 14),
                "rsi_oversold": strategy_params.get("rsi_oversold", 30),
                "rsi_overbought": strategy_params.get("rsi_overbought", 70),
                "macd_fast": strategy_params.get("macd_fast", 12),
                "macd_slow": strategy_params.get("macd_slow", 26),
                "macd_signal": strategy_params.get("macd_signal", 9)
            },
            "performance": {
                "initial_capital": initial_capital,
                "final_equity": results.get("Equity Final [$]", 0),
                "total_return_pct": results.get("Return [%]", 0),
                "buy_hold_return_pct": results.get("Buy & Hold Return [%]", 0),
                "max_drawdown_pct": results.get("Max. Drawdown [%]", 0),
                "avg_drawdown_pct": results.get("Avg. Drawdown [%]", 0),
                "sharpe_ratio": results.get("Sharpe Ratio", 0),
                "profit_factor": results.get("Profit Factor", 0),
                "sqn": results.get("SQN", 0),
                "win_rate_pct": results.get("Win Rate [%]", 0),
                "expectancy": self.calculate_expectancy(processed_trades),
                "avg_trade_pct": results.get("Avg. Trade [%]", 0),
                "max_trade_duration": results.get("Max. Trade Duration", 0),
                "avg_trade_duration": results.get("Avg. Trade Duration", 0),
                "total_trades": results.get("# Trades", 0),
                "exposure_time_pct": results.get("Exposure Time [%]", 0)
            },
            "risk_management": {
                "position_sizing": "Dynamic (5% per trade)",
                "leverage": leverage,
                "stop_loss_type": "Fixed percentage",
                "take_profit_type": "Fixed percentage",
                "max_position_size": "100% of equity"
            },
            "trades": processed_trades
        }
        
        # Save to file
        filename = f"BB_Scalping_{data_info['symbol']}_{data_info['timeframe']}_{self.current_timestamp}.json"
        filepath = self.base_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(strategy_data, f, indent=4)
        
        return filepath
    
    @staticmethod
    def calculate_expectancy(trades):
        """Calculate strategy expectancy"""
        if not trades:
            return 0
            
        win_sum = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        loss_sum = sum(abs(trade['pnl']) for trade in trades if trade['pnl'] < 0)
        wins = sum(1 for trade in trades if trade['pnl'] > 0)
        losses = sum(1 for trade in trades if trade['pnl'] < 0)
        
        total_trades = len(trades)
        if total_trades == 0:
            return 0
            
        win_rate = wins / total_trades if total_trades > 0 else 0
        loss_rate = losses / total_trades if total_trades > 0 else 0
        
        avg_win = win_sum / wins if wins > 0 else 0
        avg_loss = loss_sum / losses if losses > 0 else 0
        
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return expectancy

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
        "n": kwargs.get("n", 20),
        "ndev": kwargs.get("ndev", 2.0),
        "bb_touch_threshold": kwargs.get("bb_touch_threshold", 0.002),
        "sl_pct": kwargs.get("sl_pct", 0.01),
        "tp_pct": kwargs.get("tp_pct", 0.01),
        "rsi_period": kwargs.get("rsi_period", 14),
        "rsi_oversold": kwargs.get("rsi_oversold", 30),
        "rsi_overbought": kwargs.get("rsi_overbought", 70),
        "macd_fast": kwargs.get("macd_fast", 12),
        "macd_slow": kwargs.get("macd_slow", 26),
        "macd_signal": kwargs.get("macd_signal", 9),
        "use_rsi": kwargs.get("use_rsi", False),
        "use_macd": kwargs.get("use_macd", True),
        "use_candlestick": kwargs.get("use_candlestick", True)
    }
    cerebro.addstrategy(BollingerBandsStrategy, **strategy_params)
    initial_cash = 100.0
    leverage = 10
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        commtype=bt.CommInfoBase.COMM_PERC,
        margin=1.0 / leverage
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
            n=20,
            ndev=2.0,
            bb_touch_threshold=0.002,
            sl_pct=0.01,
            tp_pct=0.01,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            use_rsi=False,
            use_macd=True,
            use_candlestick=True
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
            pd.DataFrame(all_results).to_csv("partial_bb_scalping_results.csv", index=False)
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
                pd.DataFrame(all_results).to_csv("partial_bb_scalping_results.csv", index=False)
        except Exception as e2:
            print("\nError printing partial results:")
            print(str(e2))
            print(traceback.format_exc()) 