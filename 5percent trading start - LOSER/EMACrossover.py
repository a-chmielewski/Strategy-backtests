import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import random
from deap import base, creator, tools, algorithms
import multiprocessing
from functools import partial
import math
from collections import deque
import json
import traceback
from pathlib import Path

random.seed(42)

class EMACrossStrategy(bt.Strategy):
    """Base template for creating trading strategies"""
    
    params = (
        ("ema_short", 9),
        ("ema_long", 80),
        ("stochastic_period", 12),
        ("stochastic_pfast", 5),
        ("stochastic_pslow", 3),
        ("stop_loss", 0.01),
        ("take_profit", 0.01),
    )

    def __init__(self):
        """Initialize strategy components"""
        self.buy_signals = []
        self.sell_signals = []
        self.trade_exits = []
        self.active_trades = []  # To track ongoing trades for visualization
        
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.stochastic = bt.indicators.Stochastic(
            self.data,
            period=self.p.stochastic_period,
            period_dfast=self.p.stochastic_pfast,
            period_dslow=self.p.stochastic_pslow,
            movav=bt.indicators.MovAv.Simple
        )

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
        # Check if we're in the market
        if not self.position:
            # Long Entry Conditions
            golden_cross = self.ema_short > self.ema_long
            price_near_ema = (self.data.close[0] <= self.ema_short[0] * 1.01 and 
                             self.data.close[0] >= self.ema_short[0] * 0.99) or \
                            (self.data.close[0] <= self.ema_long[0] * 1.01 and 
                             self.data.close[0] >= self.ema_long[0] * 0.99)
            stoch_above_20 = self.stochastic.lines.percK[0] > 20

            # Short Entry Conditions
            death_cross = self.ema_short < self.ema_long
            stoch_below_80 = self.stochastic.lines.percK[0] < 80

            # Calculate position size
            position_size = self.calculate_position_size(self.data.close[0])
            current_price = self.data.close[0]

            # Enter Long Position
            if golden_cross and price_near_ema and stoch_above_20:
                stop_loss = current_price * (1 - self.p.stop_loss)
                take_profit = current_price * (1 + self.p.take_profit)
                self.buy_signals.append(self.data.datetime.datetime(0))
                
                # Create bracket order with market entry
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    price=current_price,
                    stopprice=stop_loss,
                    limitprice=take_profit,
                    stopexec=bt.Order.Stop,
                    limitexec=bt.Order.Limit,
                )

            # Enter Short Position
            elif death_cross and price_near_ema and stoch_below_80:
                stop_loss = current_price * (1 + self.p.stop_loss)
                take_profit = current_price * (1 - self.p.take_profit)
                self.sell_signals.append(self.data.datetime.datetime(0))
                
                # Create bracket order with market entry
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    price=current_price,
                    stopprice=stop_loss,
                    limitprice=take_profit,
                    stopexec=bt.Order.Stop,
                    limitexec=bt.Order.Limit,
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
    """
    A custom analyzer to record trade details upon entry (when trade just opened)
    and exit (when the trade closes). This ensures that we store and retrieve
    trade details by the same reference, avoiding the `order.ref` vs `trade.ref`
    mismatch.
    """

    def __init__(self):
        super(TradeRecorder, self).__init__()
        self.active_trades = {}  # Holds data for open trades by trade.ref
        self.trades = []         # Holds final results for closed trades

    def notify_trade(self, trade):
        """
        Called by Backtrader when a trade is updated. We capture the trade
        details upon opening (`trade.justopened`) and store them in `self.active_trades`.
        When the trade closes (`trade.status == trade.Closed`), we finalize
        the trade record and append it to `self.trades`.
        """

        # 1) Trade Just Opened
        if trade.isopen and trade.justopened:
            # Compute approximate "value" = entry_price * size
            # Use abs(...) to handle both long (positive) and short (negative) trades
            trade_value = abs(trade.price * trade.size)
            
            self.active_trades[trade.ref] = {
                'entry_time': len(self.strategy),  # integer index of current bar
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

                # We can use trade.price for the final execution price
                exit_price = trade.price
                
                # Store final trade record
                self.trades.append({
                    'datetime': self.strategy.datetime.datetime(),  # exit bar's datetime
                    'type': 'long' if trade.size > 0 else 'short',
                    'size': entry_data['size'],
                    'price': exit_price,           # Synonymous with 'exit_price'
                    'value': entry_data['value'],  # From entry time
                    'pnl': float(trade.pnl),
                    'pnlcomm': float(trade.pnlcomm),
                    'commission': float(trade.commission),
                    'entry_price': entry_data['entry_price'],
                    'exit_price': exit_price,
                    'bars_held': bars_held
                })

    def get_analysis(self):
        """
        Required Backtrader method that returns the final stored results.
        In this case, we return the list of closed trades.
        """
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

    # Extract strategy parameters from kwargs or use defaults
    strategy_params = {
        "ema_short": kwargs.get("ema_short", 9),
        "ema_long": kwargs.get("ema_long", 80),
        "stochastic_period": kwargs.get("stochastic_period", 12),
        "stochastic_pfast": kwargs.get("stochastic_pfast", 5),
        "stochastic_pslow": kwargs.get("stochastic_pslow", 3),
        "stop_loss": kwargs.get("stop_loss", 0.01),
        "take_profit": kwargs.get("take_profit", 0.01),
    }

    # Add strategy with parameters
    cerebro.addstrategy(EMACrossStrategy, **strategy_params)

    initial_cash = 100.0
    leverage = 10  # Move leverage to a variable
    
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        margin=1.0 / leverage,  # Use leverage variable
        commtype=bt.CommInfoBase.COMM_PERC
    )
    cerebro.broker.set_slippage_perc(0.0001)
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(DetailedDrawdownAnalyzer, _name="detailed_drawdown")
    cerebro.addanalyzer(TradeRecorder, _name='trade_recorder')

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

    # Get detailed trade data
    trades_data = strat.analyzers.trade_recorder.get_analysis()
    
    # Get drawdown analysis
    dd_analysis = strat.analyzers.detailed_drawdown.get_analysis()
    
    # Update formatted_results with correct drawdown values
    formatted_results.update({
        "Max. Drawdown [%]": dd_analysis.get('max_drawdown', 0),
        "Avg. Drawdown [%]": dd_analysis.get('avg_drawdown', 0),
    })
    
    # Create data info dictionary with strategy parameters
    data_info = {
        "symbol": kwargs.get("symbol", "Unknown"),
        "timeframe": kwargs.get("timeframe", "Unknown"),
        "start_date": data.datetime.iloc[0].strftime("%Y-%m-%d"),
        "end_date": data.datetime.iloc[-1].strftime("%Y-%m-%d"),
        "data_source": kwargs.get("data_source", "Unknown"),
        "total_bars": len(data)
    }
    
    # Save comprehensive results
    collector = StrategyDataCollector()
    saved_path = collector.save_backtest_results(
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
        # Convert individual parameters to strategy parameters
        params = {
            "ema_short": int(individual[0]),
            "ema_long": int(individual[1]),
            "stochastic_period": int(individual[2]),
            "stochastic_pfast": int(individual[3]),
            "stochastic_pslow": int(individual[4]),
            "stop_loss": round(individual[5], 3),
            "take_profit": round(individual[6], 3),
        }
        
        # Run backtest with these parameters
        results = run_backtest(data, verbose=False, **params)
        
        # Multi-objective fitness calculation
        sqn = results.get('Equity Final [$]', 0)
        sharpe = results.get('Sharpe Ratio', 0)
        returns = results.get('Return [%]', 0)
        trades = results.get('# Trades', 0)
        
        # Penalize strategies with too few trades
        min_trades = 50  # Minimum desired trades
        trade_penalty = max(0, 1 - (trades / min_trades))
        
        # Combine metrics into single fitness score
        fitness = (
            0.4 * sqn +                    # 40% weight on SQN
            0.3 * sharpe +                 # 30% weight on Sharpe
            0.2 * (returns / 100) +        # 20% weight on Returns
            0.1 * (1 - trade_penalty)      # 10% weight on trade frequency
        )
        
        return (fitness,)  # DEAP requires a tuple
        
    except Exception as e:
        print(f"Error evaluating individual: {str(e)}")
        return (-100.0,)  # Return very low fitness on error

def optimize_strategy(data, pop_size=50, generations=30, early_stop_generations=10, improvement_threshold=1e-4):
    """
    Optimize the EMACrossStrategy parameters using genetic algorithm
    """
    try:
        setup_deap()
        toolbox = base.Toolbox()

        # Register genetic operators with ranges suitable for 1-min crypto trading
        toolbox.register("ema_short", random.randint, 5, 50)        # 5-50 minutes
        toolbox.register("ema_long", random.randint, 20, 200)       # 20-200 minutes
        toolbox.register("stoch_period", random.randint, 3, 14)     # 3-14 minutes
        toolbox.register("stoch_pfast", random.randint, 2, 5)       # 2-5 periods
        toolbox.register("stoch_pslow", random.randint, 2, 5)       # 2-5 periods
        toolbox.register("stop_loss", random.uniform, 0.005, 0.02)  # 0.5-2%
        toolbox.register("take_profit", random.uniform, 0.01, 0.04) # 1-4%

        # Create individual
        def create_individual():
            return creator.Individual([
                toolbox.ema_short(),
                toolbox.ema_long(),
                toolbox.stoch_period(),
                toolbox.stoch_pfast(),
                toolbox.stoch_pslow(),
                toolbox.stop_loss(),
                toolbox.take_profit(),
            ])

        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register evaluation function
        toolbox.register("evaluate", evaluate, data=data)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Statistics setup
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else 0)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of Fame
        hof = tools.HallOfFame(5)

        # Initialize population
        pop = toolbox.population(n=pop_size)

        # Custom constraint decorator
        def checkBounds():
            def decorator(func):
                def wrapper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        # EMA constraints
                        if child[0] >= child[1]:
                            child[0] = random.randint(5, 50)
                            child[1] = random.randint(child[0] + 10, 200)
                        
                        # Stochastic constraints
                        child[2] = max(3, min(14, int(child[2])))
                        child[3] = max(2, min(5, int(child[3])))
                        child[4] = max(2, min(5, int(child[4])))
                        
                        # Stop loss and take profit constraints
                        child[5] = max(0.005, min(0.02, float(child[5])))
                        child[6] = max(0.01, min(0.04, float(child[6])))
                    return offspring
                return wrapper
            return decorator

        # Decorate operators
        toolbox.decorate("mate", checkBounds())
        toolbox.decorate("mutate", checkBounds())

        # Run the algorithm
        final_pop, logbook = algorithms.eaSimple(
            pop, 
            toolbox,
            cxpb=0.7,      # Crossover probability
            mutpb=0.2,     # Mutation probability
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Get best parameters
        best_individual = tools.selBest(final_pop, k=1)[0]
        best_params = {
            "ema_short": int(best_individual[0]),
            "ema_long": int(best_individual[1]),
            "stochastic_period": int(best_individual[2]),
            "stochastic_pfast": int(best_individual[3]),
            "stochastic_pslow": int(best_individual[4]),
            "stop_loss": float(best_individual[5]),
            "take_profit": float(best_individual[6]),
        }

        # Save Hall of Fame results
        try:
            hof_list = []
            for ind in hof:
                hof_list.append({
                    "parameters": {
                        "ema_short": int(ind[0]),
                        "ema_long": int(ind[1]),
                        "stochastic_period": int(ind[2]),
                        "stochastic_pfast": int(ind[3]),
                        "stochastic_pslow": int(ind[4]),
                        "stop_loss": float(ind[5]),
                        "take_profit": float(ind[6]),
                    },
                    "fitness": float(ind.fitness.values[0])
                })
            with open("emacross_hof_results.json", "w") as f:
                json.dump(hof_list, f, indent=4)
            print("\nBest parameters saved to 'emacross_hof_results.json'")
        except Exception as e:
            print(f"Error saving Hall of Fame results: {str(e)}")

        return best_params, logbook

    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print(f"full traceback: {traceback.format_exc()}")
        return None, None

def setup_deap():
    """Setup DEAP creator classes only if they don't exist"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def walk_forward_optimization(data, window_size_ratio=0.5, step_size_ratio=0.1):
    """Perform walk-forward optimization with relative window and step sizes"""
    results = []
    
    setup_deap()
    # Ensure data is properly indexed
    data = data.copy()
    data = data.reset_index(drop=True)
    
    window_size = int(len(data) * window_size_ratio)
    step_size = int(len(data) * step_size_ratio)
    
    for start in range(0, len(data) - window_size, step_size):
        end = start + window_size
        test_end = min(end + step_size, len(data))
        
        train_data = data.iloc[start:end].copy().reset_index(drop=True)
        test_data = data.iloc[end:test_end].copy().reset_index(drop=True)
        
        if len(train_data) < 1000 or len(test_data) < 100:
            continue

        print(f"Training period: {train_data['datetime'].iloc[0]} to {train_data['datetime'].iloc[-1]}")
        print(f"Testing period: {test_data['datetime'].iloc[0]} to {test_data['datetime'].iloc[-1]}")

        try:
            # Update to handle single return value
            best_params = optimize_strategy(
                train_data,
                pop_size=30,
                generations=20
            )

            print("Best parameters found:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

            # Rest of the walk-forward optimization code...
            test_results = run_backtest(
                test_data,
                plot=False,
                verbose=True,
                optimize=False,
                **best_params
            )
            
            test_results.update({
                'train_start': train_data['datetime'].iloc[0],
                'train_end': train_data['datetime'].iloc[-1],
                'test_start': test_data['datetime'].iloc[0],
                'test_end': test_data['datetime'].iloc[-1],
                'parameters': best_params
            })
            
            results.append(test_results)

        except Exception as e:
            print(f"Error in walk-forward period: {str(e)}")
            continue

    return results

class StrategyDataCollector:
    """Collects and stores comprehensive strategy backtest data"""
    
    def __init__(self, base_path="strategy_database"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_backtest_results(self, results, strategy_params, data_info, trades_data, 
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
        
        # Create strategy data dictionary
        strategy_data = {
            "timestamp": self.current_timestamp,
            "strategy_name": "EMACrossover",
            "strategy_type": "Trend Following",
            "strategy_description": "EMA Crossover with Stochastic Filter and Risk Management",
            "data_info": data_info,
            "parameters": strategy_params,
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
        filename = f"EMACrossover_{data_info['symbol']}_{data_info['timeframe']}_{self.current_timestamp}.json"
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
                ema_short=9,  # Add default strategy parameters
                ema_long=80,
                stochastic_period=12,
                stochastic_pfast=5,
                stochastic_pslow=3,
                stop_loss=0.01,
                take_profit=0.01
            )
            
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
        print(f"Final Equity: {result['Equity Final [$]']}")
        print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
        print(f"Profit Factor: {result['Profit Factor']:.2f}")