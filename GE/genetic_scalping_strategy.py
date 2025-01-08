import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import random
from deap import base, creator, tools, algorithms, gp
import operator
import math
import multiprocessing
from functools import partial
import traceback

# Set random seed
random.seed(42)

# Define a safe division operator
def safe_div(x, y):
    # First evaluate the inputs if they're functions
    if callable(x):
        x = x()
    if callable(y):
        y = y()
    return x / y if abs(y) > 1e-12 else 0

def rand_val_func():
    return random.uniform(-1, 1)

current_data = {}

def get_rsi_val():
    return float(current_data['rsi'])

def get_sma_val():
    return float(current_data['sma'])

def get_bb_upper_val():
    return float(current_data['bb_upper'])

def get_bb_lower_val():
    return float(current_data['bb_lower'])

def get_macd_val():
    return float(current_data['macd'])

def get_macd_signal_val():
    return float(current_data['macd_signal'])

def get_close_val():
    return float(current_data['close'])

def get_high_val():
    return float(current_data['high'])

def get_low_val():
    return float(current_data['low'])

def get_volume_val():
    return float(current_data['volume'])

class GeneticScalpingStrategy(bt.Strategy):
    params = (
        ('strategy_func', None),  
    )
    
    def __init__(self):
        # Use built-in indicators from Backtrader or add more
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.sma = bt.indicators.SMA(self.data.close, period=20)
        self.bb = bt.indicators.BollingerBands(self.data.close, period=20, devfactor=2)
        # Correct lines for Bollinger Bands
        self.bb_upper = self.bb.lines.top
        self.bb_lower = self.bb.lines.bot
        self.macd = bt.indicators.MACD(self.data.close)

        # Initialize current_data to avoid KeyErrors on first calls
        global current_data
        current_data = {
            'rsi': 0.0,
            'sma': 0.0,
            'bb_upper': 0.0,
            'bb_lower': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'close': 0.0,
            'high': 0.0,
            'low': 0.0,
            'volume': 0.0
        }
        
    def calculate_position_size(self, current_price):
        current_equity = self.broker.getvalue()
        position_value = min(current_equity, 100.0)
        leverage = 50
        position_size = (position_value * leverage) / current_price if current_price != 0 else 0
        return position_size

    def next(self):
        global current_data
        # Update current_data global with correct indicator keys
        current_data = {
            'rsi': self.rsi[0],
            'sma': self.sma[0],
            'bb_upper': self.bb_upper[0],
            'bb_lower': self.bb_lower[0],
            'macd': self.macd.lines.macd[0],
            'macd_signal': self.macd.lines.signal[0],
            'close': self.data.close[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'volume': self.data.volume[0]
        }

        # Evaluate strategy_func with no arguments
        if self.p.strategy_func is not None:
            try:
                signal = self.p.strategy_func()
            except ZeroDivisionError:
                signal = 0.0

            if signal > 0 and not self.position:
                size = self.calculate_position_size(self.data.close[0])
                if size > 0:
                    self.buy(size=size)
            elif signal < 0 and self.position:
                self.close()

def setup_primitives():
    pset = gp.PrimitiveSet("MAIN", 0)

    # Use regular functions instead of lambdas
    pset.addTerminal(get_rsi_val, name='rsi')
    pset.addTerminal(get_sma_val, name='sma')
    pset.addTerminal(get_bb_upper_val, name='bb_upper')
    pset.addTerminal(get_bb_lower_val, name='bb_lower')
    pset.addTerminal(get_macd_val, name='macd')
    pset.addTerminal(get_macd_signal_val, name='macd_signal')
    pset.addTerminal(get_close_val, name='close')
    pset.addTerminal(get_high_val, name='high')
    pset.addTerminal(get_low_val, name='low')
    pset.addTerminal(get_volume_val, name='volume')

    # Arithmetic
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(safe_div, 2)

    # Comparisons
    pset.addPrimitive(operator.gt, 2)
    pset.addPrimitive(operator.lt, 2)

    # Logical
    pset.addPrimitive(operator.and_, 2)
    pset.addPrimitive(operator.or_, 2)

    # Ephemeral constant
    pset.addEphemeralConstant("rand_val", rand_val_func)

    return pset

def evaluate_strategy(individual, data, pset):
    try:
        # First compile the strategy function
        strategy_func = gp.compile(individual, pset)
        
        # Create and setup cerebro
        cerebro = bt.Cerebro()
        data_feed = bt.feeds.PandasData(
            dataname=data,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=-1
        )
        cerebro.adddata(data_feed)
        
        # Add strategy with the compiled function
        cerebro.addstrategy(GeneticScalpingStrategy, strategy_func=strategy_func)
        
        # Setup broker
        cerebro.broker.setcash(100.0)
        cerebro.broker.setcommission(commission=0.0002, leverage=50)
        cerebro.broker.set_slippage_perc(0.0001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run the strategy
        results = cerebro.run()
        strategy = results[0]
        
        # Get analysis results
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        dd = strategy.analyzers.drawdown.get_analysis()
        max_dd = dd['max']['drawdown'] if 'max' in dd else 0
        trades = strategy.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        
        if total_trades == 0:
            return (-1000,)  # no trades = large penalty
        
        win_rate = won_trades / total_trades if total_trades > 0 else 0.0
        
        # Fitness function
        fitness = (sharpe * 2.0) + (win_rate * 1.0) - (max_dd * 0.5)
        return (fitness,)
        
    except Exception as e:
        print(f"Error in evaluate_strategy: {e}")
        return (-1000,)  # Return a penalty value on error

def main():
    data = pd.read_csv('data/bybit-BTCUSDT-1m-20240607-to-20241204.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Create a test strategy instance to initialize indicators
    test_cerebro = bt.Cerebro()
    test_cerebro.adddata(bt.feeds.PandasData(
        dataname=data,
        datetime='datetime',
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=None
    ))
    test_cerebro.addstrategy(GeneticScalpingStrategy)
    test_cerebro.run()  # This will initialize the indicators
    
    # Now setup primitives after indicators are initialized
    pset = setup_primitives()
    
    # Rest of the genetic algorithm setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_strategy, data=data, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Use simple map instead of multiprocessing
    toolbox.register("map", map)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=30, stats=stats, halloffame=hof, verbose=True)
    
    # Best strategy
    print("\nBest strategy found:")
    print(hof[0])
    
    # Evaluate best strategy again
    best_strategy_func = gp.compile(hof[0], pset)
    cerebro = bt.Cerebro()
    data_feed = bt.feeds.PandasData(
        dataname=data,

        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume'
    )
    cerebro.adddata(data_feed)
    cerebro.addstrategy(GeneticScalpingStrategy, strategy_func=best_strategy_func)
    cerebro.broker.setcash(100.0)
    cerebro.broker.setcommission(commission=0.0002, leverage=50)
    cerebro.broker.set_slippage_perc(0.0001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print("\nRunning final backtest with best strategy...")
    results = cerebro.run()
    strategy = results[0]
    sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    dd = strategy.analyzers.drawdown.get_analysis()
    max_dd = dd['max']['drawdown'] if 'max' in dd else 0
    trades = strategy.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"\nFinal Results:")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")

if __name__ == "__main__":
    main()
