import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from deap import base, creator, tools, algorithms
import random
import multiprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

class MaCrossoverStrategy(Strategy):
    '''
    Optimialization results:
    Best parameters found:
    fast_ma_period = 5
    slow_ma_period = 15
    rsi_period = 13
    atr_period = 15
    vol_ma_period = 22
    cci_period = 11
    atr_multiplier = 2.0
    min_atr_threshold = 40
    profit_target = 0.0002
    max_spread = 9
    risk_per_trade = 0.02
    rsi_oversold = 20
    rsi_overbought = 77
    max_trade_duration = 9
    '''

    # Define parameters that can be optimized
    fast_ma_period = 5      # Fast EMA period
    slow_ma_period = 20     # Slow EMA period
    rsi_period = 14        # RSI period
    atr_period = 14        # ATR period
    vol_ma_period = 20     # Volume MA period
    cci_period = 14        # CCI period
    profit_target = 0.02   # 2% profit target
    atr_multiplier = 1.5   # Multiplier for ATR-based stop loss
    min_atr_threshold = 50.0  # Minimum ATR value to trade (in price points)
    max_spread = 10.0       # Maximum allowed spread in points
    risk_per_trade = 0.01 # 1% of equity per trade
    
    # Indicator thresholds
    rsi_oversold = 30
    rsi_overbought = 70
    rsi_middle = 50
    cci_high = 100
    cci_low = -100
    
    # Add new parameters
    max_trade_duration = 5  # Maximum trade duration in minutes
    
    def init(self):
        # Calculate EMAs using mid price for signals
        self.fast_ma = self.I(talib.EMA, self.data.mid_price, timeperiod=self.fast_ma_period)
        self.slow_ma = self.I(talib.EMA, self.data.mid_price, timeperiod=self.slow_ma_period)
        
        # Calculate RSI
        self.rsi = self.I(talib.RSI, self.data.mid_price, timeperiod=self.rsi_period)
        
        # Calculate ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period)
        
        # Calculate Volume MA
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        
        # Add CCI indicator
        self.cci = self.I(talib.CCI, self.data.High, self.data.Low, self.data.Close,
                         timeperiod=self.cci_period)
        
        # Additional indicators for analysis
        self.spread = self.data.spread
        self.volatility = self.data.volatility

        # Initialize variables to store SL and TP levels
        self.sl = None
        self.tp = None
        
        # Add trade time tracking
        self.entry_time = None

    def should_exit_trade(self):
        """Check if we should exit the current trade based on our exit criteria"""
        if not self.position or not self.entry_time:
            return False
        
        current_time = self.data.index[-1]
        trade_duration = (current_time - self.entry_time).total_seconds() / 60
        
        # 1. Time-based exit
        if trade_duration >= self.max_trade_duration:
            return True
        
        # 2. Signal reversal exit
        ma_signal_reversed = (self.position.is_long and crossover(self.slow_ma, self.fast_ma)) or \
                             (self.position.is_short and crossover(self.fast_ma, self.slow_ma))
        
        rsi_rising = self.rsi[-1] > self.rsi[-2]
        rsi_falling = self.rsi[-1] < self.rsi[-2]
        
        rsi_signal_reversed = (self.position.is_long and rsi_falling) or \
                              (self.position.is_short and rsi_rising)
        
        cci_bull = self.cci[-1] > self.cci_high
        cci_bear = self.cci[-1] < self.cci_low
        
        cci_signal_reversed = (self.position.is_long and cci_bear) or \
                              (self.position.is_short and cci_bull)
        
        
        if ma_signal_reversed and rsi_signal_reversed and cci_signal_reversed:
            return True
        
        # 3. Profit target check (using actual trade P&L)
        # if self.position.pl_pct >= self.profit_target:
        #     return True
        
        return False

    def next(self):
        # First, check if we should exit existing position
        if np.isnan([self.fast_ma[-1], self.slow_ma[-1], self.rsi[-1], self.cci[-1]]).any():
            return
        
        if self.position:
            # Check for exit conditions
            if self.should_exit_trade():
                self.position.close()
                self.entry_time = None
                self.sl = None
                self.tp = None
                return
            
            # If in position, check for SL/TP
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            
            if self.position.is_long:
                if (self.tp is not None and current_high >= self.tp) or \
                   (self.sl is not None and current_low <= self.sl):
                    self.position.close()
                    self.entry_time = None
                    self.sl = None
                    self.tp = None
            elif self.position.is_short:
                if (self.tp is not None and current_low <= self.tp) or \
                   (self.sl is not None and current_high >= self.sl):
                    self.position.close()
                    self.entry_time = None
                    self.sl = None
                    self.tp = None
            return  # Exit after handling SL/TP
        
        # Skip if spread is too wide or ATR is too low
        if (self.spread[-1] > self.max_spread or 
            self.atr[-1] < self.min_atr_threshold):
            return
        
        # Skip if volume is below its MA
        if self.data.Volume[-1] <= self.volume_ma[-1]:
            return
        
        # Calculate position size based on risk
        risk_amount = self.equity * self.risk_per_trade

        crossed_above = crossover(self.fast_ma, self.slow_ma)
        crossed_below = crossover(self.slow_ma, self.fast_ma)
        rsi_rising = self.rsi[-1] > self.rsi[-2]
        rsi_falling = self.rsi[-1] < self.rsi[-2]
        cci_bull = self.cci[-1] > self.cci_high
        cci_bear = self.cci[-1] < self.cci_low
        
        # Long position setup
        if (crossed_above and rsi_rising and cci_bull and not self.position and
            self.rsi_middle < self.rsi[-1] < self.rsi_overbought):
            
            # Calculate long position parameters
            entry_price = self.data.ask[-1]  # Use Close price for entry
            
            # Calculate ATR-based stop loss
            atr_stop = self.atr[-1] * self.atr_multiplier
            stop_loss_price = entry_price - atr_stop
            take_profit_price = entry_price + (atr_stop * self.profit_target)  # 2:1 reward-to-risk ratio
            
            # Ensure SL and TP are logical
            if stop_loss_price >= entry_price:
                return  # Invalid SL
            
            # Calculate position size based on ATR stop loss
            risk_per_coin = entry_price - stop_loss_price
            btc_position = risk_amount / risk_per_coin
            
            # Round to nearest whole number and ensure minimum of 1 BTC
            size = 1 # max(1, int(btc_position))
            
            # Place buy order
            self.buy(size=size)
            
            # Set entry time
            self.entry_time = self.data.index[-1]
            
            # Set SL and TP
            self.sl = stop_loss_price
            self.tp = take_profit_price
        
        # Short position setup
        elif (crossed_below and rsi_falling and cci_bear and not self.position and
            self.rsi_oversold < self.rsi[-1] < self.rsi_middle):
            
            # Calculate short position parameters
            entry_price = self.data.bid[-1]  # Use Close price for entry
            
            # Calculate ATR-based stop loss
            atr_stop = self.atr[-1] * self.atr_multiplier
            stop_loss_price = entry_price + atr_stop
            take_profit_price = entry_price - (atr_stop * self.profit_target)  # 2:1 reward-to-risk ratio
            
            # Ensure SL and TP are logical
            if stop_loss_price <= entry_price:
                return  # Invalid SL
            
            # Calculate position size based on ATR stop loss
            risk_per_coin = stop_loss_price - entry_price
            btc_position = risk_amount / risk_per_coin
            
            # Round to nearest whole number and ensure minimum of 1 BTC
            size = 1 # max(1, int(btc_position))
            
            # Place sell order
            self.sell(size=size)
            
            # Set entry time
            self.entry_time = self.data.index[-1]
            
            # Set SL and TP
            self.sl = stop_loss_price
            self.tp = take_profit_price

def run_backtest(data, **kwargs):
    """
    Run the backtest with the given data and parameters.
    """
    # Initialize Backtest
    bt = Backtest(
        data,
        MaCrossoverStrategy,
        cash=1000000.0,
        commission=0.0002,
        exclusive_orders=False,
        trade_on_close=False,
        margin=1.0 / 50
    )

    # Set default parameters if not provided
    if not kwargs:
        kwargs = {
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'rsi_period': 14,
            'atr_period': 14,
            'vol_ma_period': 20,
            'cci_period': 14,
            'atr_multiplier': 1.5,
            'min_atr_threshold': 50.0,
            'profit_target': 0.02,
            'max_spread': 10.0,
            'risk_per_trade': 0.01,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'max_trade_duration': 5,  # 5 minutes maximum trade duration
        }
    
    # Run Backtest with provided parameters
    results = bt.run(**kwargs)

    # Create a simpler filename for the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"MA_Crossover_Backtest_{timestamp}.html"
    
    # Plot the results with the simplified filename
    bt.plot(filename=plot_filename, resample=False)

    return results

def evaluate_strategy(individual, data):
    """
    Evaluate the strategy with given parameters.
    Returns a composite fitness score based on multiple metrics.
    """
    # Suppress RuntimeWarnings during backtest evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        params = {
            'fast_ma_period': individual[0],
            'slow_ma_period': individual[1],
            'rsi_period': individual[2],
            'atr_period': individual[3],
            'vol_ma_period': individual[4],
            'cci_period': individual[5],
            'atr_multiplier': individual[6],
            'min_atr_threshold': individual[7],
            'profit_target': individual[8],
            'max_spread': individual[9],
            'risk_per_trade': individual[10],
            'rsi_oversold': individual[11],
            'rsi_overbought': individual[12],
            'max_trade_duration': individual[13]
        }
        
        # Run backtest with these parameters
        try:
            bt = Backtest(
                data,
                MaCrossoverStrategy,
                cash=1000000.0,
                commission=0.0002,
                exclusive_orders=False,
                trade_on_close=False,
                margin=1.0 / 50
            )   
            stats = bt.run(**params)
        except Exception as e:
            # Penalize invalid parameter combinations
            return (-np.inf,)
        
        # Extract relevant metrics
        total_trades = stats['# Trades']
        win_rate = stats['Win Rate [%]'] / 100
        profit_factor = stats['Profit Factor']
        return_pct = stats['Return [%]'] / 100
        max_drawdown = stats['Max. Drawdown [%]'] / 100
        
        # Handle edge cases
        if total_trades < 10:  # Require minimum number of trades
            return (-np.inf,)
        
        if max_drawdown == 0:  # Avoid division by zero
            max_drawdown = 0.0001
        
        if profit_factor == 0:  # Avoid division by zero
            profit_factor = 0.0001
        
        # Calculate composite score
        # Weighted combination of different metrics
        score = (
            win_rate * 0.3 +                    # 30% weight on win rate
            (return_pct / max_drawdown) * 0.4 + # 40% weight on return/drawdown ratio
            (profit_factor / 10) * 0.3         # 30% weight on profit factor (scaled)
        )
        
        # Penalize extreme parameter combinations
        if individual[0] >= individual[1]:  # fast MA period >= slow MA period
            score *= 0.1
        
        if individual[11] >= individual[12]:  # RSI oversold >= overbought
            score *= 0.1
        
        return (max(score, 0.0001),)  # Ensure non-negative score

def optimize_strategy(data, population_size=100, generations=30):
    """
    Optimize strategy parameters using genetic algorithm.
    """
    # Clear any existing DEAP types
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    # Create fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # Initialize toolbox
    toolbox = base.Toolbox()
    
    # Define parameter ranges (min, max) to avoid overfitting
    param_ranges = [
        (3, 10),       # fast_ma_period (integer)
        (15, 30),      # slow_ma_period (integer)
        (10, 20),      # rsi_period (integer)
        (10, 20),      # atr_period (integer)
        (15, 30),      # vol_ma_period (integer)
        (10, 20),      # cci_period (integer)
        (1.0, 2.0),    # atr_multiplier (float)
        (30, 70),      # min_atr_threshold (integer)
        (0.0002, 0.003),  # profit_target (float)
        (5, 15),       # max_spread (integer)
        (0.005, 0.02), # risk_per_trade (float)
        (20, 40),      # rsi_oversold (integer)
        (60, 80),      # rsi_overbought (integer)
        (3, 10)        # max_trade_duration (integer)
    ]
    
    # Register parameter generators with proper types
    for i, (min_val, max_val) in enumerate(param_ranges):
        if i in [0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 13]:  # Integer parameters
            toolbox.register(f"attr_{i}", random.randint, min_val, max_val)
        else:  # Float parameters
            toolbox.register(f"attr_{i}", random.uniform, min_val, max_val)
    
    # Mutation function for mixed types
    def mixed_type_mutation(individual, mu, sigma, indpb):
        for i, val in enumerate(individual):
            if random.random() < indpb:
                if i in [0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 13]:  # Integer parameters
                    individual[i] = int(random.gauss(val, sigma))
                    # Ensure values stay within bounds
                    individual[i] = max(param_ranges[i][0], min(param_ranges[i][1], individual[i]))
                else:  # Float parameters
                    individual[i] = random.gauss(val, sigma)
                    individual[i] = max(param_ranges[i][0], min(param_ranges[i][1], individual[i]))
        return individual,
    
    # Create individual and population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [getattr(toolbox, f"attr_{i}") for i in range(len(param_ranges))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators
    toolbox.register("evaluate", evaluate_strategy, data=data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mixed_type_mutation, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Set up parallel processing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run optimization
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3,
                                         ngen=generations, stats=stats, halloffame=hof,
                                         verbose=True)
    finally:
        pool.close()
    
    # Get best parameters
    best_params = {
        'fast_ma_period': int(hof[0][0]),
        'slow_ma_period': int(hof[0][1]),
        'rsi_period': int(hof[0][2]),
        'atr_period': int(hof[0][3]),
        'vol_ma_period': int(hof[0][4]),
        'cci_period': int(hof[0][5]),
        'atr_multiplier': float(hof[0][6]),
        'min_atr_threshold': int(hof[0][7]),
        'profit_target': float(hof[0][8]),
        'max_spread': int(hof[0][9]),
        'risk_per_trade': float(hof[0][10]),
        'rsi_oversold': int(hof[0][11]),
        'rsi_overbought': int(hof[0][12]),
        'max_trade_duration': int(hof[0][13])
    }
    
    return best_params, logbook

if __name__ == "__main__":

    data = pd.read_csv(
        r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20241007-to-20241106.csv'
    )

        
    # Data preprocessing as before
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'bid', 'ask', 'mid_price', 'spread', 'volatility']
    
    assert all(col in data.columns for col in required_columns), \
        f"Missing required columns. Required: {required_columns}"

    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    if not data.index.is_monotonic_increasing:
        data.sort_index(inplace=True)
    data['mid_price'] = (data['bid'] + data['ask']) / 2
    data.dropna(subset=['bid', 'ask', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    data.fillna(method='ffill', inplace=True)



    # Run optimization
    # print("Starting parameter optimization...")
    # best_params, logbook = optimize_strategy(data)
    # print("\nBest parameters found:")
    # for param, value in best_params.items():
    #     print(f"{param}: {value}")
    
    # Run backtest with optimized parameters
    print("\nRunning backtest with optimized parameters...")
    results = run_backtest(data) #, **best_params)
    print("\nBacktest results:")
    print(results)
