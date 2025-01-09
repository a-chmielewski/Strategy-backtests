import backtrader as bt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import random
from deap import base, creator, tools, algorithms
import multiprocessing
from functools import partial
import math
import traceback
from typing import Optional
import operator

random.seed(42)

class VwapIntradayIndicator(bt.Indicator):
    """
    Volume Weighted Average Price (VWAP) indicator for intraday trading.
    Resets on a new day based on the date from the datetime field.
    """
    
    lines = ("vwap_intraday",)
    params = {}
    plotinfo = {"subplot": False}
    plotlines = {"vwap_intraday": {"color": "blue"}}

    def __init__(self) -> None:
        # Calculate the typical price
        self.hlc = (self.data.high + self.data.low + self.data.close) / 3.0

        # Initialize tracking variables
        self.current_date: Optional[datetime.date] = None
        self.previous_date_index: int = -1
        
        # Initialize cumulative values
        self.cum_vol = 0.0
        self.cum_hlc_vol = 0.0

    def next(self) -> None:
        try:
            # Extract the current date from the datetime field
            current_datetime = self.data.datetime.datetime()
            current_date = current_datetime.date()

            # Check if the date has changed
            if self.current_date != current_date:
                # Reset cumulative values on new day
                self.current_date = current_date
                self.cum_vol = 0.0
                self.cum_hlc_vol = 0.0

            # Get current values
            current_volume = self.data.volume[0]
            current_hlc = self.hlc[0]

            # Update cumulative values
            if not math.isnan(current_volume) and not math.isnan(current_hlc):
                self.cum_vol += current_volume
                self.cum_hlc_vol += current_hlc * current_volume

            # Calculate VWAP with safety check for zero volume
            if self.cum_vol > 0:
                self.lines.vwap_intraday[0] = self.cum_hlc_vol / self.cum_vol
            else:
                # If no volume, use the typical price as fallback
                self.lines.vwap_intraday[0] = current_hlc

        except Exception as e:
            print(f"Error in VWAP calculation: {str(e)}")
            # Use typical price as fallback in case of any error
            self.lines.vwap_intraday[0] = self.hlc[0]

class KeltnerChannels(bt.Indicator):
    """Keltner Channels indicator for Backtrader"""
    
    lines = ('kc_upper', 'kc_middle', 'kc_lower',)
    params = (
        ('period', 20),
        ('multiplier', 2.0),
        ('atr_period', 10),
    )

    def __init__(self):
        # Calculate the middle band using EMA
        self.lines.kc_middle = bt.indicators.EMA(self.data.close, period=self.p.period)
        
        # Calculate ATR for volatility
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # Calculate upper and lower bands
        self.lines.kc_upper = self.lines.kc_middle + self.p.multiplier * self.atr
        self.lines.kc_lower = self.lines.kc_middle - self.p.multiplier * self.atr

class StochRSI(bt.Indicator):
    """Stochastic RSI indicator for Backtrader"""
    
    lines = ('stochrsi_k', 'stochrsi_d', 'rsi',)
    params = (
        ('rsi_period', 14),
        ('stoch_period', 14),
        ('smooth_k', 3),
        ('smooth_d', 3),
    )
    
    plotlines = dict(
        stochrsi_k=dict(color='blue'),
        stochrsi_d=dict(color='red'),
        rsi=dict(_plotskip=True),  # Don't plot RSI line
    )

    def __init__(self):
        # Calculate RSI first
        self.lines.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.p.rsi_period,
            movav=bt.indicators.SMA
        )
        
        # Calculate rolling min/max of RSI
        rsi_min = bt.indicators.Lowest(self.lines.rsi, period=self.p.stoch_period)
        rsi_max = bt.indicators.Highest(self.lines.rsi, period=self.p.stoch_period)
        
        # Calculate raw StochRSI without smoothing: (RSI - RSI_min) / (RSI_max - RSI_min)
        # Handle division by zero using bt.If
        denominator = rsi_max - rsi_min
        stochrsi_raw = bt.If(
            denominator != 0,
            (self.lines.rsi - rsi_min) / denominator * 100,
            0.0  # Assign 0.0 when denominator is zero
        )
        
        # Apply smoothing to K and D
        self.lines.stochrsi_k = bt.indicators.SMA(stochrsi_raw, period=self.p.smooth_k)
        self.lines.stochrsi_d = bt.indicators.SMA(self.lines.stochrsi_k, period=self.p.smooth_d)



class StrategyTemplate(bt.Strategy):
    """Base template for creating trading strategies"""
    
    params = (
        ("keltner_period", 20),
        ("keltner_multiplier", 2.0),
        ("stochrsi_period", 14),
        ("stochrsi_k", 3),
        ("stochrsi_d", 3),
        ("stop_loss_pct", 0.005),
        ("take_profit_ratio", 0.01),
    )

    def log(self, txt, dt=None):
        """Logging function for strategy events"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} | {txt}")

    def __init__(self):
        """Initialize strategy components"""
        # Initialize trade tracking
        self.trade_exits = []
        self.active_trades = []  # To track ongoing trades for visualization
        
        self.vwap_intraday = VwapIntradayIndicator()
        self.keltner = KeltnerChannels(
            period=self.p.keltner_period,
            multiplier=self.p.keltner_multiplier
        )
        self.stochrsi = StochRSI(
            rsi_period=self.p.stochrsi_period,
            stoch_period=self.p.stochrsi_period,
            smooth_k=self.p.stochrsi_k,
            smooth_d=self.p.stochrsi_d
        )

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
        # Skip if not enough data for indicators
        if (self.vwap_intraday[0] is None or
            self.keltner.kc_lower[0] is None or
            self.keltner.kc_upper[0] is None or
            self.stochrsi.stochrsi_k[0] is None or
            self.stochrsi.stochrsi_d[0] is None):
            return

        # Get current position
        position = self.position.size
        
        # Get current price and indicator values
        current_price = self.data.close[0]
        keltner_lower = self.keltner.kc_lower[0]
        keltner_upper = self.keltner.kc_upper[0]
        keltner_middle = self.keltner.kc_middle[0]
        vwap = self.vwap_intraday[0]
        stochrsi_k = self.stochrsi.stochrsi_k[0]
        stochrsi_d = self.stochrsi.stochrsi_d[0]
        rsi = self.stochrsi.rsi[0]

        # Debug logging (uncomment to see values)
        # self.log(f"StochRSI K: {stochrsi_k:.2f}, D: {stochrsi_d:.2f}, RSI: {rsi:.2f}")

        # Entry conditions (only if no position is open)
        if not position:
            # Long signal when StochRSI K crosses above 20 from oversold
            long_signal = (
                current_price < keltner_lower and
                current_price < vwap and
                self.stochrsi.stochrsi_k[-1] < 20 and  # Previous K value
                stochrsi_k > 20  # Current K value
            )

            # Short signal when StochRSI K crosses below 80 from overbought
            short_signal = (
                current_price > keltner_upper and
                current_price > vwap and
                self.stochrsi.stochrsi_k[-1] > 80 and  # Previous K value
                stochrsi_k < 80  # Current K value
            )

            # Calculate position size if we need to enter a trade
            position_size = self.calculate_position_size(current_price)

            # Exit conditions for existing positions
            if position:
                # For long positions
                if position > 0:
                    # Take profit at VWAP or middle Keltner band
                    if current_price >= min(vwap, keltner_middle) or \
                       (current_price - self.position.price) / self.position.price >= self.p.take_profit_ratio:
                        self.close()
                        self.log("Long Position Closed (Take Profit)")
                    # Stop loss
                    elif (current_price - self.position.price) / self.position.price <= -self.p.stop_loss_pct:
                        self.close()
                        self.log("Long Position Closed (Stop Loss)")
                
                # For short positions
                elif position < 0:
                    # Take profit at VWAP or middle Keltner band
                    if current_price <= max(vwap, keltner_middle) or \
                       (self.position.price - current_price) / self.position.price >= self.p.take_profit_ratio:
                        self.close()
                        self.log("Short Position Closed (Take Profit)")
                    # Stop loss
                    elif (self.position.price - current_price) / self.position.price <= -self.p.stop_loss_pct:
                        self.close()
                        self.log("Short Position Closed (Stop Loss)")

            # Execute trades
            if long_signal:
                position_size = self.calculate_position_size(current_price)
                stopprice = current_price - (current_price * self.p.stop_loss_pct)
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=keltner_middle,
                    stopprice=stopprice
                )
                # self.log("Long Position Entered")

            elif short_signal:
                position_size = self.calculate_position_size(current_price)
                stopprice = current_price + (current_price * self.p.stop_loss_pct)
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=keltner_middle,
                    stopprice=stopprice
                )
                # self.log("Short Position Entered")

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

    # Pass strategy parameters via kwargs
    cerebro.addstrategy(StrategyTemplate, **kwargs)
    
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,               # your commission rate
        commtype=bt.CommInfoBase.COMM_PERC,
        leverage=50,                     # set leverage
        margin=1.0/50                    # margin requirement (for 50x leverage)
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

    # Add expectancy calculation to formatted_results
    win_rate = formatted_results['Win Rate [%]'] / 100
    loss_rate = 1 - win_rate
    avg_win = safe_get(trades_analysis, 'won', 'pnl', 'average', default=0)
    avg_loss = abs(safe_get(trades_analysis, 'lost', 'pnl', 'average', default=0))
    
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    
    formatted_results['Expectancy'] = expectancy

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

def evaluate(individual, data):
    """Evaluate individual's fitness during optimization"""
    try:
        params = {
            "vwap_period": individual[0],
            "sma_period": individual[1],
            "stop_loss_pct": individual[2] / 1000,  # Convert to percentage (e.g., 5 -> 0.005)
            "take_profit_ratio": individual[3] / 1000  # Convert to percentage (e.g., 10 -> 0.01)
        }

        results = run_backtest(data, verbose=False, **params)
        
        # Calculate fitness based on multiple metrics
        ret = results.get("Return [%]", 0)
        sqn = results.get("SQN", 0)
        sharpe = results.get("Sharpe Ratio", 0)
        trades = results.get("# Trades", 0)
        win_rate = results.get("Win Rate [%]", 0)
        
        # Penalize strategies with too few trades
        if trades < 10:
            return (-np.inf,)
            
        # Combine metrics into a single fitness score
        fitness = (ret * 0.4) + (sqn * 0.2) + (sharpe * 0.2) + (win_rate * 0.2)
        
        return (fitness,)
    except Exception as e:
        print(f"Error evaluating individual: {str(e)}")
        return (-np.inf,)

def optimize_strategy(data, pop_size=50, generations=30):
    """Optimize strategy parameters using genetic algorithm"""
    
    # Create fitness and individual types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register genetic operators with parameter ranges
    toolbox.register("vwap_period", random.randint, 5, 30)  # VWAP period
    toolbox.register("sma_period", random.randint, 10, 50)  # Volume SMA period
    toolbox.register("stop_loss_pct", random.randint, 3, 20)    # Stop loss in 0.1% (30 -> 3%)
    toolbox.register("take_profit_ratio", random.randint, 6, 40)  # Take profit in 0.1% (60 -> 6%)

    # Create individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.vwap_period, toolbox.sma_period, 
                      toolbox.stop_loss_pct, toolbox.take_profit_ratio))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom mutation operator that ensures integer values
    def custom_mutate(individual, mu, sigma, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                # Add Gaussian noise and round to nearest integer
                individual[i] = int(round(individual[i] + random.gauss(mu, sigma)))
                # Ensure values stay within reasonable bounds
                if i == 0:  # vwap_period
                    individual[i] = max(5, min(30, individual[i]))
                elif i == 1:  # sma_period
                    individual[i] = max(10, min(50, individual[i]))
                elif i == 2:  # stop_loss_pct
                    individual[i] = max(3, min(20, individual[i]))
                elif i == 3:  # take_profit_ratio
                    individual[i] = max(6, min(40, individual[i]))
        return individual,

    # Register genetic operators
    evaluate_partial = partial(evaluate, data=data)
    toolbox.register("evaluate", evaluate_partial)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    # Create hall of fame
    hof = tools.HallOfFame(1)

    # Run optimization
    with multiprocessing.Pool() as pool:
        toolbox.register("map", pool.map)
        pop = toolbox.population(n=pop_size)
        final_pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.7,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

    # Get best parameters
    best_individual = hof[0]
    best_params = {
        "vwap_period": best_individual[0],
        "sma_period": best_individual[1],
        "stop_loss_pct": best_individual[2] / 1000,
        "take_profit_ratio": best_individual[3] / 1000
    }

    return best_params, logbook

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
            data_df = data_df[data_df["Close"] != 0.0]  # remove rows with zero close
            data_df = data_df[data_df["Open"] != 0.0]
            data_df = data_df[data_df["High"] != 0.0]
            data_df = data_df[data_df["Low"]  != 0.0]


            data_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
            data_df.sort_values("datetime", inplace=True)

            
            # Run backtest
            results = run_backtest(data_df, verbose=False)
            
            # Add symbol and timeframe to results
            results['symbol'] = symbol
            results['timeframe'] = timeframe
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {data_path}: {str(e)}")
            full_traceback = traceback.format_exc()
            print(f"Full traceback: {full_traceback}")
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
        print(f"Expectancy: {result['Expectancy']:.4f}")
        print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
        print(f"Profit Factor: {result['Profit Factor']:.2f}")