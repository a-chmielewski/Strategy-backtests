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

class emasarp_Strategy(bt.Strategy):
    params = (
        ("ema", 200),
        ("sar_step", 0.02), 
        ("sar_max", 0.2),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("stop_loss", 0.01),
        ("take_profit", 0.01),
    )

    def __init__(self):
        """Initialize strategy components"""
        # Initialize indicators
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema)
        self.sar = bt.indicators.ParabolicSAR(self.data, self.data.close, af=self.p.sar_step, afmax=self.p.sar_max)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow, period_signal=self.p.macd_signal)

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
        if len(self.data) < 2:  # Need at least 2 bars
            return

        closing_price = self.data.close[0]

        if not self.position:  # If no position is open
            position_size = self.calculate_position_size(closing_price)

            # Skip if size is too small
            if position_size <= 0:
                return

            # Buy conditions
            ema_uptrend = closing_price > self.ema[0]
            sar_bullish = closing_price > self.sar[0]
            macd_bullish = self.macd.macd[0] > self.macd.signal[0]

            # Sell conditions
            ema_downtrend = closing_price < self.ema[0]
            sar_bearish = closing_price < self.sar[0]
            macd_bearish = self.macd.macd[0] < self.macd.signal[0]

            if ema_uptrend and sar_bullish and macd_bullish:
                # Calculate absolute stop and take profit prices
                stop_price = closing_price * (1 - self.p.stop_loss)  # e.g., 0.5% below
                take_profit = closing_price * (1 + self.p.take_profit)  # e.g., 1% above

                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    stopprice=stop_price,
                    limitprice=take_profit,
                )

            elif ema_downtrend and sar_bearish and macd_bearish:
                # Calculate absolute stop and take profit prices
                stop_price = closing_price * (1 + self.p.stop_loss)  # e.g., 0.5% above
                take_profit = closing_price * (1 - self.p.take_profit)  # e.g., 1% below

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
    
    def __init__(self, base_path="strategy_database/EMA_SAR_MACD"):
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
            "strategy_type": "Trend Following",
            "strategy_description": "EMA with SAR and MACD Strategy",
            "data_info": data_info,
            "parameters": {
                "ema": strategy_params.get("ema", 200),
                "sar_step": strategy_params.get("sar_step", 0.02),
                "sar_max": strategy_params.get("sar_max", 0.2),
                "macd_fast": strategy_params.get("macd_fast", 12),
                "macd_slow": strategy_params.get("macd_slow", 26),
                "macd_signal": strategy_params.get("macd_signal", 9),
                "stop_loss": strategy_params.get("stop_loss", 0.01),
                "take_profit": strategy_params.get("take_profit", 0.01)
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
        filename = f"EMA_SAR_MACD_{data_info['symbol']}_{data_info['timeframe']}_{self.current_timestamp}.json"
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

    # Extract strategy parameters from kwargs or use defaults
    strategy_params = {
        "ema": kwargs.get("ema", 200),
        "sar_step": kwargs.get("sar_step", 0.02),
        "sar_max": kwargs.get("sar_max", 0.2),
        "macd_fast": kwargs.get("macd_fast", 12),
        "macd_slow": kwargs.get("macd_slow", 26),
        "macd_signal": kwargs.get("macd_signal", 9),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.01)
    }

    # Add strategy with parameters
    cerebro.addstrategy(emasarp_Strategy, **strategy_params)
    
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

    # Get trade data and calculate expectancy
    try:
        trades_data = strat.analyzers.trade_recorder.get_analysis()
        
        # Calculate expectancy directly from trades data
        def calculate_expectancy(trades):
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

        expectancy = calculate_expectancy(trades_data)
    except Exception as e:
        print(f"Error calculating expectancy: {e}")
        expectancy = 0.0

    # Format results with expectancy included
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
        "SQN": sqn,
        "Expectancy": expectancy
    }

    # Create data info dictionary with symbol and timeframe from kwargs
    data_info = {
        "symbol": kwargs.get("symbol", "Unknown"),
        "timeframe": kwargs.get("timeframe", "Unknown"),
        "start_date": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "end_date": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "data_source": kwargs.get("data_source", "Unknown"),
        "total_bars": len(data)
    }
    
    # Get trade data
    trades_data = strat.analyzers.trade_recorder.get_analysis()
    
    # Save comprehensive results
    collector = StrategyDataCollector()
    saved_path = collector.save_backtest_results(
        strategy_name="EMA_SAR_MACD",
        results=formatted_results,
        strategy_params=strategy_params,
        data_info=data_info,
        trades_data=trades_data,
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
            "ema": individual[0],
            "sar_step": individual[1] / 1000,  # Convert to small decimal
            "sar_max": individual[2] / 100,    # Convert to decimal
            "macd_fast": individual[3],
            "macd_slow": individual[4],
            "macd_signal": individual[5],
            "stop_loss": individual[6] / 1000,  # Convert to percentage
            "take_profit": individual[7] / 1000  # Convert to percentage
        }

        # Pass the params to run_backtest
        results = run_backtest(data, verbose=False, optimize=True, **params)
        
        # Calculate fitness score combining multiple metrics
        sharpe_ratio = float(results["Sharpe Ratio"])
        returns = float(results["Return [%]"])
        num_trades = float(results["# Trades"])
        
        
        # Penalize for excessive drawdown
        max_drawdown = abs(float(results["Max. Drawdown [%]"]))
        drawdown_penalty = max(0, max_drawdown - 20) / 100  # Penalize drawdowns over 20%
        
        # Combine metrics with weights
        fitness = (
            0.4 * sharpe_ratio +  # Weight for Sharpe ratio
            0.3 * (returns / 100) +  # Weight for returns (normalized)
            0.3 * num_trades -  # Weight for number of trades (normalized)
            drawdown_penalty  # Penalty for large drawdowns
        )
        
        return (fitness,)
    except Exception as e:
        print(f"Error evaluating individual: {str(e)}")
        return (-np.inf,)

def optimize_strategy(data, pop_size=50, generations=30):
    """Optimize strategy parameters using genetic algorithm"""
    
    # Clear any existing creators to avoid conflicts
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual
    
    # Create fitness and individual types
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Register parameter ranges for optimization
    toolbox.register("ema", random.randint, 50, 300)        # EMA period
    toolbox.register("sar_step", random.randint, 10, 50)    # SAR step (0.01-0.05)
    toolbox.register("sar_max", random.randint, 10, 30)     # SAR max (0.1-0.3)
    toolbox.register("macd_fast", random.randint, 8, 20)    # MACD fast period
    toolbox.register("macd_slow", random.randint, 21, 40)   # MACD slow period
    toolbox.register("macd_signal", random.randint, 5, 15)  # MACD signal period
    toolbox.register("stop_loss", random.randint, 15, 50)   # Stop loss in 0.1% (1.5-5%)
    toolbox.register("take_profit", random.randint, 20, 100) # Take profit in 0.1% (2-10%)

    # Create individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.ema, toolbox.sar_step, toolbox.sar_max,
                      toolbox.macd_fast, toolbox.macd_slow, toolbox.macd_signal,
                      toolbox.stop_loss, toolbox.take_profit))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators
    evaluate_partial = partial(evaluate, data=data)
    toolbox.register("evaluate", evaluate_partial)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, 
                     low=[50,10,10,8,21,5,15,20],
                     up=[300,50,30,20,40,15,50,100], 
                     indpb=0.2)
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
        "ema": best_individual[0],
        "sar_step": best_individual[1] / 1000,
        "sar_max": best_individual[2] / 100,
        "macd_fast": best_individual[3],
        "macd_slow": best_individual[4],
        "macd_signal": best_individual[5],
        "stop_loss": best_individual[6] / 1000,
        "take_profit": best_individual[7] / 1000
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
            
            # Run backtest
            results = run_backtest(
                data_df,
                verbose=False,
                symbol=symbol,
                timeframe=timeframe,
                data_source="Bybit",
                ema=200,
                sar_step=0.02,
                sar_max=0.2,
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
                stop_loss=0.01,
                take_profit=0.01
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