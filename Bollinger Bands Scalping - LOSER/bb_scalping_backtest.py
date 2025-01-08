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
from functools import partial

warnings.filterwarnings("ignore", category=FutureWarning)

class BollingerBandsStrategy(Strategy):
    """
    Fixed parameters - core strategy parameters that shouldn't be optimized
    """
    # Risk management
    min_size = 1.0  # Minimum position size in BTC
    risk_pct = 0.01  # Risk per trade (1% of equity)
    
    # Technical thresholds
    bb_touch_threshold = 0.001  # How close price needs to be to bands
    volume_threshold = 1.5  # Minimum volume increase vs average
    
    """
    Optimizable parameters - parameters we want to find optimal values for
    """
    # Bollinger Bands
    n = 20  # Period for BB (try range 10-30)
    ndev = 2.0  # Standard deviation multiplier (try range 1.5-3.0)
    
    # RSI
    rsi_period = 14  # Period for RSI (try range 10-20)
    
    # Stochastic
    stoch_k = 14  # %K period (try range 10-20)
    stoch_d = 3   # %D period (try range 2-5)
    stoch_smooth = 3  # %K smoothing (try range 2-5)
    stoch_upper = 80  # Overbought threshold (try range 75-85)
    stoch_lower = 20  # Oversold threshold (try range 15-25)
    
    # Volume MA
    volume_ma_period = 20  # Period for volume moving average (try range 10-30)
    
    # Trade Management
    sl_pct = 0.01  # Stop loss percentage (try range 0.005-0.02)
    tp_pct = 0.02  # Take profit percentage (try range 0.01-0.04)

    # MACD parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    # **Indicator Selection Flags** (Add these as class variables)
    use_rsi = True
    use_stoch = True
    use_volume = True
    use_macd = True
    use_candlestick = True

    def init(self):
        # Retrieve indicator selection flags
        self.use_rsi = self.use_rsi
        self.use_stoch = self.use_stoch
        self.use_volume = self.use_volume
        self.use_macd = self.use_macd
        self.use_candlestick = self.use_candlestick
        
        # Ensure n is integer and ndev is rounded to 3 decimal places
        n_int = int(self.n)  # Convert n to integer
        rounded_ndev = round(self.ndev, 3)
        
        # Calculate Bollinger Bands using pandas_ta
        bb = pd.DataFrame(self.data.df).ta.bbands(
            length=n_int,  # Use integer n
            std=rounded_ndev,
            close='Close'
        )
        
        # Store Bollinger Bands components as strategy indicators
        # Use formatted string with integer n and rounded ndev for consistent column names
        bb_middle_col = f'BBM_{n_int}_{rounded_ndev}'
        bb_upper_col = f'BBU_{n_int}_{rounded_ndev}'
        bb_lower_col = f'BBL_{n_int}_{rounded_ndev}'
        
        self.bb_middle = self.I(lambda x: bb[bb_middle_col], 'BB_middle')
        self.bb_upper = self.I(lambda x: bb[bb_upper_col], 'BB_upper')
        self.bb_lower = self.I(lambda x: bb[bb_lower_col], 'BB_lower')

        # Calculate RSI if selected
        if self.use_rsi:
            rsi = pd.DataFrame(self.data.df).ta.rsi(
                length=int(self.rsi_period),  # Ensure integer
                column='Close'
            )
            self.rsi = self.I(lambda x: rsi, 'RSI')
        else:
            self.rsi = np.full(len(self.data.Close), np.nan)

        # Calculate Stochastic Oscillator with integer parameters if selected
        if self.use_stoch:
            stoch = pd.DataFrame(self.data.df).ta.stoch(
                k=int(self.stoch_k),  # Ensure integer
                d=int(self.stoch_d),  # Ensure integer
                smooth_k=int(self.stoch_smooth)  # Ensure integer
            )
            
            # Ensure indicators are the same length as data by padding with NaN
            data_length = len(self.data.Close)
            
            # Get stochastic column names with integer parameters
            stoch_k_col = f'STOCHk_{int(self.stoch_k)}_{int(self.stoch_d)}_{int(self.stoch_smooth)}'
            stoch_d_col = f'STOCHd_{int(self.stoch_k)}_{int(self.stoch_d)}_{int(self.stoch_smooth)}'
            
            # Pad stochastic values
            stoch_k = stoch[stoch_k_col].values
            stoch_d = stoch[stoch_d_col].values
            
            # Create arrays of NaN with the same length as data
            k_line = np.full(data_length, np.nan)
            d_line = np.full(data_length, np.nan)
            
            # Fill in the calculated values
            k_line[-len(stoch_k):] = stoch_k
            d_line[-len(stoch_d):] = stoch_d
            
            # Store K and D lines as indicators
            self.stoch_k_line = self.I(lambda x: k_line, '%K')
            self.stoch_d_line = self.I(lambda x: d_line, '%D')
        else:
            self.stoch_k_line = np.full(len(self.data.Close), np.nan)
            self.stoch_d_line = np.full(len(self.data.Close), np.nan)

        # Calculate Volume Moving Average if selected
        if self.use_volume:
            volume_data = pd.DataFrame(self.data.df)['Volume']
            
            if not self.volume_ma_period.is_integer():
                self.volume_ma_period = int(self.volume_ma_period)
            elif self.volume_ma_period < 0:
                self.volume_ma_period = 1

            volume_ma = volume_data.rolling(window=self.volume_ma_period).mean()
            
            # Ensure volume MA is same length as data by padding with NaN
            volume_ma_padded = np.full(len(self.data.Close), np.nan)
            volume_ma_padded[-len(volume_ma):] = volume_ma
            
            # Store Volume MA as indicator
            self.volume_ma = self.I(lambda x: volume_ma_padded, 'Volume MA')
            
            # Calculate Volume/MA ratio for visualization
            volume_ratio = volume_data / volume_ma
            volume_ratio_padded = np.full(len(self.data.Close), np.nan)
            volume_ratio_padded[-len(volume_ratio):] = volume_ratio
            
            self.volume_ratio = self.I(lambda x: volume_ratio_padded, 'Volume Ratio')
        else:
            self.volume_ma = np.full(len(self.data.Close), np.nan)
            self.volume_ratio = np.full(len(self.data.Close), np.nan)

        # Calculate MACD if selected
        if self.use_macd:
            macd = pd.DataFrame(self.data.df).ta.macd(
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal
            )
            
            # Pad MACD values
            macd_line = macd['MACD_12_26_9'].values
            signal_line = macd['MACDs_12_26_9'].values
            histogram = macd['MACDh_12_26_9'].values
            
            # Create arrays of NaN with the same length as data
            macd_padded = np.full(len(self.data.Close), np.nan)
            signal_padded = np.full(len(self.data.Close), np.nan)
            hist_padded = np.full(len(self.data.Close), np.nan)
            
            # Fill in the calculated values
            macd_padded[-len(macd_line):] = macd_line
            signal_padded[-len(signal_line):] = signal_line
            hist_padded[-len(histogram):] = histogram
            
            # Store MACD components as indicators
            self.macd_line = self.I(lambda x: macd_padded, 'MACD')
            self.signal_line = self.I(lambda x: signal_padded, 'Signal')
            self.macd_hist = self.I(lambda x: hist_padded, 'Histogram')
        else:
            self.macd_line = np.full(len(self.data.Close), np.nan)
            self.signal_line = np.full(len(self.data.Close), np.nan)
            self.macd_hist = np.full(len(self.data.Close), np.nan)

        # Calculate candlestick patterns for reversal confirmation if selected
        if self.use_candlestick:
            self.hammer = self.I(self._is_hammer, 'Hammer')
            self.shooting_star = self.I(self._is_shooting_star, 'ShootingStar')
        else:
            self.hammer = np.full(len(self.data.Close), False)
            self.shooting_star = np.full(len(self.data.Close), False)

        # Initialize trade signal counters
        self.long_signals = 0
        self.short_signals = 0

    def _is_hammer(self, *args) -> np.ndarray:
        """Identify hammer candlestick pattern (bullish reversal)"""
        O, H, L, C = self.data.Open, self.data.High, self.data.Low, self.data.Close
        body = abs(O - C)
        lower_wick = np.minimum(O, C) - L
        upper_wick = H - np.maximum(O, C)
        
        # Hammer criteria
        return (lower_wick > 2 * body) & (upper_wick < body) & (body > 0)

    def _is_shooting_star(self, *args) -> np.ndarray:
        """Identify shooting star pattern (bearish reversal)"""
        O, H, L, C = self.data.Open, self.data.High, self.data.Low, self.data.Close
        body = abs(O - C)
        lower_wick = np.minimum(O, C) - L
        upper_wick = H - np.maximum(O, C)
        
        # Shooting star criteria
        return (upper_wick > 2 * body) & (lower_wick < body) & (body > 0)

    def _check_rsi_long_condition(self) -> bool:
        """Check if RSI is rising (bullish momentum)"""
        if len(self.rsi) < 2:
            return False
        return self.rsi[-1] > self.rsi[-2]  # RSI is rising

    def _check_rsi_short_condition(self) -> bool:
        """Check if RSI is falling (bearish momentum)"""
        if len(self.rsi) < 2:
            return False
        return self.rsi[-1] < self.rsi[-2]  # RSI is falling

    def _check_stoch_long_condition(self) -> bool:
        """Check if %K crosses above %D below the oversold level"""
        if len(self.stoch_k_line) < 2:
            return False
        
        # Check if we're below oversold level
        below_threshold = self.stoch_k_line[-1] < self.stoch_lower
        
        # Check for bullish crossover
        crossover = (self.stoch_k_line[-2] <= self.stoch_d_line[-2] and 
                    self.stoch_k_line[-1] > self.stoch_d_line[-1])
        
        return below_threshold and crossover

    def _check_stoch_short_condition(self) -> bool:
        """Check if %K crosses below %D above the overbought level"""
        if len(self.stoch_k_line) < 2:
            return False
        
        # Check if we're above overbought level
        above_threshold = self.stoch_k_line[-1] > self.stoch_upper
        
        # Check for bearish crossover
        crossover = (self.stoch_k_line[-2] >= self.stoch_d_line[-2] and 
                    self.stoch_k_line[-1] < self.stoch_d_line[-1])
        
        return above_threshold and crossover

    def _check_volume_confirmation(self) -> bool:
        """
        Check if current volume is significantly higher than recent average
        Returns True if volume is above threshold compared to its moving average
        """
        if len(self.volume_ma) < 1:
            return False
        
        current_volume = self.data.Volume[-1]
        current_volume_ma = self.volume_ma[-1]
        
        return current_volume > (current_volume_ma * self.volume_threshold)

    def _check_macd_long_condition(self) -> bool:
        """
        Check for bullish MACD signal
        Returns True when histogram crosses above zero (bullish crossover)
        """
        if len(self.macd_hist) < 2:
            return False
        
        # Check for histogram crossing above zero
        return (self.macd_hist[-2] <= 0) & (self.macd_hist[-1] > 0)

    def _check_macd_short_condition(self) -> bool:
        """
        Check for bearish MACD signal
        Returns True when histogram crosses below zero (bearish crossover)
        """
        if len(self.macd_hist) < 2:
            return False
        
        # Check for histogram crossing below zero
        return (self.macd_hist[-2] >= 0) & (self.macd_hist[-1] < 0)

    def next(self):
        # For market orders, execution will happen at the next bar's open
        current_price = self.data.Close[-1]
        
        # Check if price is touching/crossing bands
        touching_lower = (current_price <= self.bb_lower[-1] * (1 + self.bb_touch_threshold))
        touching_upper = (current_price >= self.bb_upper[-1] * (1 - self.bb_touch_threshold))

        # Long entry conditions
        long_condition = (
            touching_lower  # Price touches lower band
            and (not self.use_candlestick or self.hammer[-1])  # Hammer candlestick pattern
            and (not self.use_rsi or self._check_rsi_long_condition())  # RSI condition
            and (not self.use_stoch or self._check_stoch_long_condition())  # Stochastic condition
            and (not self.use_volume or self._check_volume_confirmation())  # Volume confirmation
            and (not self.use_macd or self._check_macd_long_condition())  # MACD confirmation
            and not self.position  # No existing position
        )

        # Short entry conditions
        short_condition = (
            touching_upper  # Price touches upper band
            and (not self.use_candlestick or self.shooting_star[-1])  # Shooting star pattern
            and (not self.use_rsi or self._check_rsi_short_condition())  # RSI condition
            and (not self.use_stoch or self._check_stoch_short_condition())  # Stochastic condition
            and (not self.use_volume or self._check_volume_confirmation())  # Volume confirmation
            and (not self.use_macd or self._check_macd_short_condition())  # MACD confirmation
            and not self.position  # No existing position
        )

        # Execute trades with stop loss and take profit
        if long_condition:
            self.long_signals += 1
            # For long positions:
            sl_price = current_price * (1 - self.sl_pct)
            tp_price = max(current_price * (1 + self.tp_pct), self.bb_middle[-1])
            
            # Directly place the buy order without additional conditions
            size = self._calculate_position_size(current_price - sl_price) * 50
            self.buy(size=size, sl=sl_price, tp=tp_price)
            
        elif short_condition:
            self.short_signals += 1
            # For short positions:
            sl_price = current_price * (1 + self.sl_pct)
            tp_price = min(current_price * (1 - self.tp_pct), self.bb_middle[-1])
            
            # Directly place the sell order without additional conditions
            size = self._calculate_position_size(sl_price - current_price) * 50
            self.sell(size=size, sl=sl_price, tp=tp_price)

    def _calculate_position_size(self, stop_distance):
        """
        Calculate position size as 1% of available equity.
        If 1% would result in less than 1 unit, return 1 unit instead.
        
        Args:
            stop_distance (float): Distance between entry price and stop loss (not used in this version)
        
        Returns:
            float: Either 0.01 (1%) or 1.0 (one unit), whichever is larger
        """

        if stop_distance <= 0:
            return 1.0
        # Calculate what 1% of equity would be in units
        price = self.data.Close[-1]
        equity_in_units = self.equity / price
        one_percent_in_units = equity_in_units * self.risk_pct
        
        # If 1% would be less than 1 unit, return 1 unit
        # Otherwise return 0.01 (1% as a fraction)
        return 1.0 if one_percent_in_units < 1.0 else self.risk_pct

    def stop(self):
        print(f"Indicator Usage - RSI: {self.use_rsi}, Stochastic: {self.use_stoch}, "
              f"Volume: {self.use_volume}, MACD: {self.use_macd}, Candlestick: {self.use_candlestick}")
        print(f"Total long signals: {self.long_signals}")
        print(f"Total short signals: {self.short_signals}")

def run_backtest(data, plot=False, **kwargs):
    """
    Run the backtest with the given data and parameters.
    """
    bt = Backtest(
        data,
        BollingerBandsStrategy,
        cash=1000000.0,
        commission=0.0002,
        exclusive_orders=False,
        trade_on_close=False,
        margin=1.0 / 50  # 50x leverage
    )
    
    results = bt.run(**kwargs)

    if plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"BB_Scalping_Backtest_{timestamp}.html"
        bt.plot(filename=plot_filename, resample=False)

    return results

def evaluate(individual, data):
    """Evaluate an individual (set of parameters and indicator selection)."""
    params = {
        'n': int(individual[0]),
        'ndev': round(individual[1], 3),
        'rsi_period': int(individual[2]),
        'stoch_k': int(individual[3]),
        'stoch_d': int(individual[4]),
        'stoch_smooth': int(individual[5]),
        'stoch_upper': int(individual[6]),
        'stoch_lower': int(individual[7]),
        'volume_ma_period': int(individual[8]),
        'sl_pct': float(individual[9]),
        'tp_pct': float(individual[10]),
        'use_rsi': bool(individual[11]),
        'use_stoch': bool(individual[12]),
        'use_volume': bool(individual[13]),
        'use_macd': bool(individual[14]),
        'use_candlestick': bool(individual[15])
    }
    
    # Validate parameters
    if not (10 <= params['n'] <= 30 and
            1.5 <= params['ndev'] <= 3.0 and
            10 <= params['rsi_period'] <= 20 and
            10 <= params['stoch_k'] <= 20 and
            2 <= params['stoch_d'] <= 5 and
            2 <= params['stoch_smooth'] <= 5 and
            75 <= params['stoch_upper'] <= 85 and
            15 <= params['stoch_lower'] <= 25 and
            10 <= params['volume_ma_period'] <= 30 and
            0.005 <= params['sl_pct'] <= 0.02 and
            0.01 <= params['tp_pct'] <= 0.04 and
            isinstance(params['use_rsi'], bool) and
            isinstance(params['use_stoch'], bool) and
            isinstance(params['use_volume'], bool) and
            isinstance(params['use_macd'], bool) and
            isinstance(params['use_candlestick'], bool)):
        return (-np.inf,)  # Penalize invalid parameters
    
    # Run backtest without plotting during optimization
    try:
        results = run_backtest(data, plot=False, **params)
        
        # Extract metrics
        sharpe = results['Sharpe Ratio']
        return_pct = results['Return [%]']
        num_trades = results['_trades'].shape[0]
        
        # Handle cases with no trades
        if num_trades == 0:
            # print(f"No trades for params: {params}")
            return (-np.inf,)
        
        # Normalize metrics (Assuming typical ranges; adjust as needed)
        # For example purposes, let's assume:
        # Sharpe Ratio: 0 to 3
        # Return Percentage: 0% to 100%
        # Number of Trades: 0 to 1000
        
        norm_sharpe = sharpe / 3.0  # Normalize Sharpe to [0,1]
        norm_return = return_pct / 100.0  # Normalize Return to [0,1]
        norm_trades = min(num_trades / 1000.0, 1.0)  # Normalize Trades to [0,1]
        
        # Assign weights
        weight_sharpe = 0.2
        weight_return = 0.4
        weight_trades = 0.4
        
        # Composite fitness score
        fitness = (norm_sharpe * weight_sharpe) + (norm_return * weight_return) + (norm_trades * weight_trades)
        
        # print(f"Params: {params}, Sharpe: {sharpe:.2f}, Return: {return_pct:.2f}%, Trades: {num_trades}, Fitness: {fitness:.4f}")
        return (fitness,)
    except Exception as e:
        # print(f"Backtest failed for parameters: {params} with error: {e}")
        return (-np.inf,)

from functools import partial

def optimize_strategy(data, population_size=50, generations=30):
    """
    Optimize strategy parameters and indicator selection using DEAP genetic algorithm
    """
    # Create types for optimization
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except AttributeError:
        pass  # FitnessMax already exists
    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except AttributeError:
        pass

    # Initialize toolbox
    toolbox = base.Toolbox()

    # Define genes (parameters to optimize)
    toolbox.register("n", random.randint, 10, 30)
    toolbox.register("ndev", lambda: round(random.uniform(1.5, 3.0), 3))
    toolbox.register("rsi_period", random.randint, 10, 20)
    toolbox.register("stoch_k", random.randint, 10, 20)
    toolbox.register("stoch_d", random.randint, 2, 5)
    toolbox.register("stoch_smooth", random.randint, 2, 5)
    toolbox.register("stoch_upper", random.randint, 75, 85)
    toolbox.register("stoch_lower", random.randint, 15, 25)
    toolbox.register("volume_ma_period", random.randint, 10, 30)
    toolbox.register("sl_pct", lambda: round(random.uniform(0.005, 0.02), 5))
    toolbox.register("tp_pct", lambda: round(random.uniform(0.01, 0.04), 5))
    
    # Indicator selection genes
    toolbox.register("use_rsi", random.randint, 0, 1)
    toolbox.register("use_stoch", random.randint, 0, 1)
    toolbox.register("use_volume", random.randint, 0, 1)
    toolbox.register("use_macd", random.randint, 0, 1)
    toolbox.register("use_candlestick", random.randint, 0, 1)

    # Create individual and population with indicator selection
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n, toolbox.ndev, toolbox.rsi_period,
                      toolbox.stoch_k, toolbox.stoch_d, toolbox.stoch_smooth,
                      toolbox.stoch_upper, toolbox.stoch_lower,
                      toolbox.volume_ma_period, toolbox.sl_pct, toolbox.tp_pct,
                      toolbox.use_rsi, toolbox.use_stoch, toolbox.use_volume,
                      toolbox.use_macd, toolbox.use_candlestick),
                     n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register 'evaluate' with 'data' using partial
    toolbox.register("evaluate", partial(evaluate, data=data))

    # Register genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    
    # Custom mutation function to handle both parameters and indicator selection
    def mutate_and_clip(individual, mu=0, sigma=0.1, indpb=0.2):
        # Perform Gaussian mutation on continuous and integer parameters
        # For binary genes, perform bit flip with probability indpb
        for i in range(len(individual)):
            if i < 11:  # Parameters to mutate
                if random.random() < indpb:
                    if isinstance(individual[i], int):
                        mutated_value = individual[i] + int(random.gauss(mu, sigma))
                        # Clip to bounds
                        if i == 0:
                            low, high = 10, 30
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 1:
                            low, high = 1.5, 3.0
                            individual[i] = round(np.clip(individual[i] + random.gauss(mu, sigma), low, high), 3)
                        elif i == 2:
                            low, high = 10, 20
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 3:
                            low, high = 10, 20
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 4:
                            low, high = 2, 5
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 5:
                            low, high = 2, 5
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 6:
                            low, high = 75, 85
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 7:
                            low, high = 15, 25
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 8:
                            low, high = 10, 30
                            individual[i] = int(np.clip(mutated_value, low, high))
                        elif i == 9:
                            low, high = 0.005, 0.02
                            individual[i] = round(np.clip(individual[i] + random.gauss(mu, sigma), low, high), 5)
                        elif i == 10:
                            low, high = 0.01, 0.04
                            individual[i] = round(np.clip(individual[i] + random.gauss(mu, sigma), low, high), 5)
            else:  # Binary genes for indicators
                if random.random() < indpb:
                    # Flip the bit
                    individual[i] = 1 - individual[i]
        return (individual,)

    toolbox.register("mutate", mutate_and_clip, mu=0, sigma=0.1, indpb=0.2)

    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Initialize multiprocessing pool
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Run optimization
    final_pop, logbook = algorithms.eaSimple(pop, toolbox,
                                           cxpb=0.7,  # Crossover probability
                                           mutpb=0.3,  # Mutation probability
                                           ngen=generations,
                                           stats=stats,
                                           verbose=True)
    
    pool.close()
    pool.join()
    
    # Get best individual
    best_individual = tools.selBest(final_pop, k=1)[0]
    best_params = {
        'n': int(best_individual[0]),
        'ndev': round(best_individual[1], 3),
        'rsi_period': int(best_individual[2]),
        'stoch_k': int(best_individual[3]),
        'stoch_d': int(best_individual[4]),
        'stoch_smooth': int(best_individual[5]),
        'stoch_upper': int(best_individual[6]),
        'stoch_lower': int(best_individual[7]),
        'volume_ma_period': int(best_individual[8]),
        'sl_pct': float(best_individual[9]),
        'tp_pct': float(best_individual[10]),
        'use_rsi': bool(best_individual[11]),
        'use_stoch': bool(best_individual[12]),
        'use_volume': bool(best_individual[13]),
        'use_macd': bool(best_individual[14]),
        'use_candlestick': bool(best_individual[15])
    }
    
    return best_params, logbook

if __name__ == "__main__":

    data = pd.read_csv(
        r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20241007-to-20241106.csv'
    )
        
    # Data preprocessing as before
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'bid_price', 'ask_price', 'bid_vol_l1', 'ask_vol_l1', 
                        'buy_volume', 'sell_volume']
    
    assert all(col in data.columns for col in required_columns), \
        f"Missing required columns. Required: {required_columns}"

    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    # **Step 1: Run Backtest with Fixed Parameters and Indicators**
    fixed_params = {
        'n': 10,
        'ndev': 1.738,
        'rsi_period': 11,
        'stoch_k': 14,
        'stoch_d': 4,
        'stoch_smooth': 4,
        'stoch_upper': 81,
        'stoch_lower': 16,
        'volume_ma_period': 16,
        'sl_pct': 0.01257,
        'tp_pct': 0.01572,
        'use_rsi': False,
        'use_stoch': False,
        'use_volume': False,
        'use_macd': False,
        'use_candlestick': True
    }

    print("\nRunning backtest with fixed parameters and indicators...")
    results_fixed = run_backtest(data, plot=True, **fixed_params)
    print("\nBacktest results with fixed parameters:")
    print(results_fixed)

    # **Step 2: Run Optimization**
    print("\nStarting parameter and indicator optimization...")
    best_params, logbook = optimize_strategy(data)

    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # **Step 3: Run Final Backtest with Optimized Parameters and Indicators**
    print("\nRunning backtest with optimized parameters and indicators...")
    results_optimized = run_backtest(data, plot=True, **best_params)
    print("\nBacktest results with optimized parameters:")
    print(results_optimized)
