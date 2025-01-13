import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
import math
import traceback

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

            leverage = 10

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
    """Custom analyzer to record individual trade results for SQN calculation"""
    
    def __init__(self):
        super(TradeRecorder, self).__init__()
        self.trades = []

    def get_analysis(self):
        return self

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.trades.append({
                'pnl': trade.pnlcomm,
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

def run_backtest(data, plot=False, verbose=True, **kwargs):
    """Run backtest with the given data and parameters"""
    cerebro = bt.Cerebro()

    # Add data feed
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

    # Add strategy
    cerebro.addstrategy(BollingerBandsStrategy, **kwargs)
    
    # Set broker parameters
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        commtype=bt.CommInfoBase.COMM_PERC,
        margin=1.0/10
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
        cerebro.addobserver(bt.observers.BuySell)
        cerebro.addobserver(bt.observers.Value)
        cerebro.addobserver(bt.observers.DrawDown)

    # Run backtest and get results
    results = cerebro.run()
    strat = results[0]

    # Calculate metrics
    try:
        trade_recorder = strat.analyzers.trade_recorder.get_analysis()
        sqn = calculate_sqn(trade_recorder.trades if hasattr(trade_recorder, 'trades') else [])
    except Exception as e:
        print(f"Error accessing trade recorder: {e}")
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
        "Equity Peak [$]": final_value + (safe_get(drawdown_analysis, "max", "drawdown", default=0) * final_value / 100),
        "Return [%]": total_return,
        "Buy & Hold Return [%]": ((data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100),
        "Max. Drawdown [%]": safe_get(drawdown_analysis, "max", "drawdown", default=0),
        "Avg. Drawdown [%]": safe_get(drawdown_analysis, "average", "drawdown", default=0),
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

    # Add expectancy calculation
    win_rate = formatted_results['Win Rate [%]'] / 100
    loss_rate = 1 - win_rate
    avg_win = safe_get(trades_analysis, 'won', 'pnl', 'average', default=0)
    avg_loss = abs(safe_get(trades_analysis, 'lost', 'pnl', 'average', default=0))
    
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    formatted_results['Expectancy'] = expectancy

    if verbose:
        print("\n=== Strategy Performance Report ===")
        print(f"\nPeriod: {formatted_results['Start']} - {formatted_results['End']} ({formatted_results['Duration']})")
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"Final Capital: ${float(formatted_results['Equity Final [$]']):,.2f}")
        print(f"Total Return: {float(formatted_results['Return [%]']):,.2f}%")
        print(f"Buy & Hold Return: {float(formatted_results['Buy & Hold Return [%]']):,.2f}%")
        print(f"\nTotal Trades: {int(formatted_results['# Trades'])}")
        print(f"Expectancy: {float(formatted_results['Expectancy']):,.2f}")
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
    # Define all data paths
    data_paths = [
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-1000PEPEUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-1000PEPEUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ADAUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ADAUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-DOGEUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-DOGEUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ETHUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ETHUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-LINKUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-LINKUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-SOLUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-SOLUSDT-5m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-XRPUSDT-1m-20240929-to-20241128.csv",
        r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-XRPUSDT-5m-20240929-to-20241128.csv",
    ]

    # Store results for all datasets
    all_results = []

    # Test each dataset
    for data_path in data_paths:
        try:
            # Extract symbol and timeframe from path
            filename = data_path.split('\\')[-1]
            symbol = filename.split('-')[1]
            timeframe = filename.split('-')[2]
            
            print(f"\nTesting {symbol} {timeframe}...")
            
            # Load and process data
            data_df = pd.read_csv(data_path)
            data_df["datetime"] = pd.to_datetime(data_df["datetime"])
            
            # Run backtest
            results = run_backtest(data_df, verbose=False)
            
            # Add symbol and timeframe to results
            results['symbol'] = symbol
            results['timeframe'] = timeframe
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {data_path}: {str(e)}")
            continue

    # Sort results by win rate and get top 3
    sorted_results = sorted(all_results, key=lambda x: x['Win Rate [%]'], reverse=True)[:3]

    # Print top 3 results
    print("\n=== Top 3 Results by Win Rate ===")
    for i, result in enumerate(sorted_results, 1):
        print(f"\n{i}. {result['symbol']} ({result['timeframe']})")
        print(f"Win Rate: {result['Win Rate [%]']:.2f}%")
        print(f"Total Trades: {result['# Trades']}")
        print(f"Total Return: {result['Return [%]']:.2f}%")
        print(f"Final Equity: {result['Equity Final [$]']}")
        print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
        print(f"Profit Factor: {result['Profit Factor']:.2f}") 