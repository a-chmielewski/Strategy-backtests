import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import random
from deap import base, creator, tools, algorithms
import multiprocessing
from functools import partial
import math
import traceback
from pathlib import Path
import json

random.seed(42)

class VWAP(bt.Indicator):
    """
    Volume Weighted Average Price (VWAP) Indicator
    Formula: VWAP = Σ(Price * Volume) / Σ(Volume)
    """
    lines = ('vwap',)
    params = (('period', 20),)  # Adjusted default period to match strategy

    plotinfo = dict(subplot=False)
    plotlines = dict(vwap=dict(alpha=0.50, linestyle='-.', linewidth=2.0))

    def __init__(self):
        typical_price = (self.data.high + self.data.low + self.data.close) / 3
        price_vol = typical_price * self.data.volume
        cumulative_price_vol = bt.indicators.SumN(price_vol, period=self.p.period)
        cumulative_vol = bt.indicators.SumN(self.data.volume, period=self.p.period)
        self.lines.vwap = cumulative_price_vol / cumulative_vol

class VWAP_BreakoutStrategy(bt.Strategy):
    """VWAP Breakout Strategy with Relative Volume Filter"""
    
    params = (
        ("vwap_period", 20),
        ("sma_period", 20),
        ("stop_loss_pct", 0.01),
        ("take_profit_pct", 0.01),
        ("relative_volume_multiplier", 1.5),
        ("position_value_max", 100.0),
        ("risk_percentage", 0.01),
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

    def __init__(self):
        """Initialize strategy components"""
        # Custom VWAP indicator
        self.vwap = VWAP(self.data, period=self.p.vwap_period)
        
        # Volume SMA for relative volume
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.sma_period)
        
        # Track trades for visualization
        self.trade_exits = []
        self.active_trades = []
        
        # Variables to track strategy state
        self.buy_signal_candles = []  # Store candles that closed above VWAP
        self.sell_signal_candles = []  # Store candles that closed below VWAP

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

    def next(self):
        """Define trading logic"""
        if len(self.data) < 2:  # Need at least 2 bars
            return

        current_close = self.data.close[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_volume = self.data.volume[0]
        
        relative_volume = self.data.volume[0] / self.volume_sma[0] if self.volume_sma[0] != 0 else 0

        # Only consider trades when relative volume exceeds the multiplier
        if relative_volume < self.p.relative_volume_multiplier:
            return

        # Check if we have an open position
        if not self.position:
            # Update signal candles lists without resetting
            if current_close > self.vwap[0]:
                self.buy_signal_candles.append({
                    'high': current_high,
                    'low': current_low,
                    'close': current_close
                })
                if len(self.buy_signal_candles) > 5:
                    self.buy_signal_candles.pop(0)
            elif current_close < self.vwap[0]:
                self.sell_signal_candles.append({
                    'high': current_high,
                    'low': current_low,
                    'close': current_close
                })
                if len(self.sell_signal_candles) > 5:
                    self.sell_signal_candles.pop(0)
            
            # Check for buy setup
            if len(self.buy_signal_candles) >= 2:
                # Get highest high and lowest low of the last two buy signal candles
                last_two = self.buy_signal_candles[-2:]
                highest_high = max(candle['high'] for candle in last_two)
                
                # Breakout condition
                if current_high > highest_high:
                    position_size = self.calculate_position_size(current_close)
                    if position_size <= 0:
                        return
                        
                    # Calculate static stop loss and take profit levels
                    stop_price = current_close * (1 - self.p.stop_loss_pct)
                    take_profit = current_close * (1 + self.p.take_profit_pct)
                    
                    self.buy_bracket(
                        size=position_size,
                        exectype=bt.Order.Market,
                        stopprice=stop_price,
                        limitprice=take_profit,
                    )
            
            # Check for sell setup
            if len(self.sell_signal_candles) >= 2:
                # Get lowest low and highest high of the last two sell signal candles
                last_two = self.sell_signal_candles[-2:]
                lowest_low = min(candle['low'] for candle in last_two)
                
                # Breakdown condition
                if current_low < lowest_low:
                    position_size = self.calculate_position_size(current_close)
                    if position_size <= 0:
                        return
                        
                    # Calculate static stop loss and take profit levels
                    stop_price = current_close * (1 + self.p.stop_loss_pct)
                    take_profit = current_close * (1 - self.p.take_profit_pct)
                    
                    self.sell_bracket(
                        size=position_size,
                        exectype=bt.Order.Market,
                        stopprice=stop_price,
                        limitprice=take_profit,
                    )

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
        self.trades = []  # Store trades as a list
        
    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.trades.append({
                'datetime': trade.data.datetime.datetime(0),
                'type': 'long' if trade.size > 0 else 'short',
                'size': trade.size,
                'price': trade.price,
                'value': abs(trade.price * trade.size),
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'commission': trade.commission,
                'entry_price': trade.price,
                'exit_price': trade.price,
                'bars_held': len(self.strategy)  # Current bar count
            })

    def get_analysis(self):
        """Return the list of trades"""
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
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'drawdowns': self.drawdowns
        }


class StrategyDataCollector:
    """Collects and saves strategy performance data"""
    
    def __init__(self):
        self.base_path = Path("strategy_database") / "VWAP_Breakout"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_backtest_results(self, strategy_name, results, strategy_params, data_info, trades_data, initial_capital, leverage):
        """Save strategy results to JSON file"""
        
        # Process trades data
        processed_trades = []
        for trade in trades_data:
            if isinstance(trade['datetime'], datetime):
                trade = trade.copy()  # Create a copy to avoid modifying original
                trade['datetime'] = trade['datetime'].strftime("%Y-%m-%d %H:%M:%S")
            processed_trades.append(trade)
        
        strategy_data = {
            "timestamp": self.current_timestamp,
            "strategy_name": strategy_name,
            "strategy_type": "VWAP Breakout",
            "strategy_description": "VWAP breakout strategy with volume filters",
            "data_info": data_info,
            "parameters": {
                "vwap_period": strategy_params.get("vwap_period", 20),
                "sma_period": strategy_params.get("sma_period", 20),
                "stop_loss_pct": strategy_params.get("stop_loss_pct", 0.01),
                "take_profit_pct": strategy_params.get("take_profit_pct", 0.01),
                "relative_volume_multiplier": strategy_params.get("relative_volume_multiplier", 1.5)
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
        filename = f"VWAP_Breakout_{data_info['symbol']}_{data_info['timeframe']}_{self.current_timestamp}.json"
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

def run_backtest(data, plot=False, verbose=True, **kwargs):
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
        "vwap_period": kwargs.get("vwap_period", 20),
        "sma_period": kwargs.get("sma_period", 20),
        "stop_loss_pct": kwargs.get("stop_loss_pct", 0.01),
        "take_profit_pct": kwargs.get("take_profit_pct", 0.01),
        "relative_volume_multiplier": kwargs.get("relative_volume_multiplier", 1.5)
    }

    # Add strategy with parameters
    cerebro.addstrategy(VWAP_BreakoutStrategy, **strategy_params)
    
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
        drawdown_analysis = strat.analyzers.detailed_drawdown.get_analysis()
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

    # Create data info dictionary
    data_info = {
        "symbol": kwargs.get("symbol", "Unknown"),
        "timeframe": kwargs.get("timeframe", "Unknown"),
        "start_date": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "end_date": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "data_source": kwargs.get("data_source", "Unknown"),
        "total_bars": len(data)
    }
    
    # Get trade data
    trades_data = strat.analyzers.trade_recorder.trades  # Changed from get_analysis()
    
    # Save comprehensive results
    collector = StrategyDataCollector()
    saved_path = collector.save_backtest_results(
        strategy_name="VWAP_Breakout",
        results=formatted_results,
        strategy_params=strategy_params,
        data_info=data_info,
        trades_data=trades_data,  # Pass the trades list directly
        initial_capital=initial_cash,
        leverage=leverage
    )
    
    if verbose:
        print(f"\nStrategy data saved to: {saved_path}")

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
            
            # Run backtest with strategy parameters
            results = run_backtest(
                data_df,
                verbose=False,
                symbol=symbol,
                timeframe=timeframe,
                data_source="Bybit",
                vwap_period=20,
                sma_period=20,
                stop_loss_pct=0.01,
                take_profit_pct=0.01,
                relative_volume_multiplier=1.5
            )
            
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
        print(f"Expectancy: {result['Expectancy']:.4f}")
        print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
        print(f"Profit Factor: {result['Profit Factor']:.2f}")