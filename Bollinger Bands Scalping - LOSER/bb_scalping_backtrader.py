import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from deap import base, creator, tools, algorithms
import random
import multiprocessing
from functools import partial
import warnings
import logging
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    filename='backtrader.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# Add this at the global scope, before any function definitions
global_data = None  # Will store the data for evaluation

def evaluate_individual(individual) -> Tuple[float]:
    """Global evaluation function for optimization"""
    params = {
        'n': int(individual[0]),
        'ndev': round(individual[1], 3),
        'sl_pct': float(individual[2]),
        'tp_pct': float(individual[3]),
        'rsi_period': 14,
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_smooth': 3,
        'stoch_upper': 80,
        'stoch_lower': 20,
        'volume_ma_period': 20,
        'use_rsi': False,     
        'use_stoch': False,
        'use_macd': True,
        'use_candlestick': True,
        'use_volume': False,
        'bb_touch_threshold': 0.002
    }
    
    try:
        results = run_backtest(global_data, plot=False, verbose=False, **params)
        
        return_pct = results['Return [%]']
        sharpe = results['Sharpe Ratio']
        trades = results['# Trades']
        
        if trades < 10 or return_pct <= 0:
            return (-np.inf,)
            
        fitness = (
            0.4 * sharpe +           
            0.4 * (return_pct / 100) +    
            0.2 * min(trades / 100, 1.0)
        )
        
        return (fitness,)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return (-np.inf,)

def mutate_individual(individual, indpb=0.2):
    """Global mutation function"""
    param_ranges = {
        'n': (10, 30),
        'ndev': (1.5, 3.0),
        'sl_pct': (0.005, 0.02),
        'tp_pct': (0.01, 0.04),
    }
    
    for i, (param, (low, high)) in enumerate(param_ranges.items()):
        if random.random() < indpb:
            if isinstance(individual[i], int):
                individual[i] = random.randint(low, high)
            else:
                individual[i] = round(random.uniform(low, high), 5)
    return individual,

def optimize_strategy(data: pd.DataFrame, 
                     population_size: int = 30,
                     generations: int = 20,
                     tournament_size: int = 3,
                     cx_prob: float = 0.7,
                     mut_prob: float = 0.2) -> Tuple[Dict, tools.Logbook]:
    """Optimized genetic algorithm implementation"""
    global global_data
    global_data = data  # Set the global data
    
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()
    
    # Define parameter ranges
    param_ranges = {
        'n': (10, 30),
        'ndev': (1.5, 3.0),
        'sl_pct': (0.005, 0.02),
        'tp_pct': (0.01, 0.04),
    }
    
    # Register parameter generators
    toolbox.register("n", random.randint, param_ranges['n'][0], param_ranges['n'][1])
    toolbox.register("ndev", lambda: round(random.uniform(param_ranges['ndev'][0], param_ranges['ndev'][1]), 3))
    toolbox.register("sl_pct", lambda: round(random.uniform(param_ranges['sl_pct'][0], param_ranges['sl_pct'][1]), 5))
    toolbox.register("tp_pct", lambda: round(random.uniform(param_ranges['tp_pct'][0], param_ranges['tp_pct'][1]), 5))
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.n, toolbox.ndev, toolbox.sl_pct, toolbox.tp_pct),
                     n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_individual, indpb=mut_prob)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
    
    # Initialize population
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    
    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)
    
    # Run evolution without multiprocessing
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # Convert best individual to strategy parameters
    best_individual = hof[0]
    best_params = {
        'n': int(best_individual[0]),
        'ndev': round(best_individual[1], 3),
        'sl_pct': float(best_individual[2]),
        'tp_pct': float(best_individual[3]),
        'rsi_period': 14,
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_smooth': 3,
        'stoch_upper': 80,
        'stoch_lower': 20,
        'volume_ma_period': 20,
        'use_rsi': False,
        'use_stoch': False,
        'use_macd': True,
        'use_volume': True,
        'use_candlestick': True,
        'bb_touch_threshold': 0.002
    }
    
    return best_params, logbook

class BollingerBandsStrategy(bt.Strategy):
    """
    Strategy implementing Bollinger Bands with additional indicators and candlestick patterns.
    """
    params = (
        # Risk management
        ('min_size', 1.0),  # Minimum position size in BTC
        ('risk_pct', 0.01),  # Risk per trade (1% of equity)
        ('bb_touch_threshold', 0.001),  # How close price needs to be to bands
        ('volume_threshold', 1.5),  # Minimum volume increase vs average

        # Bollinger Bands
        ('n', 20),  # Period for BB (try range 10-30)
        ('ndev', 2.0),  # Standard deviation multiplier (try range 1.5-3.0)

        # RSI
        ('rsi_period', 14),  # Period for RSI (try range 10-20)

        # Stochastic
        ('stoch_k', 14),  # %K period (try range 10-20)
        ('stoch_d', 3),   # %D period (try range 2-5)
        ('stoch_smooth', 3),  # %K smoothing (try range 2-5)
        ('stoch_upper', 80),  # Overbought threshold (try range 75-85)
        ('stoch_lower', 20),  # Oversold threshold (try range 15-25)

        # Volume MA
        ('volume_ma_period', 20),  # Period for volume moving average

        # Trade Management
        ('sl_pct', 0.01),  # Stop loss percentage
        ('tp_pct', 0.02),  # Take profit percentage

        # MACD parameters
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),

        # Indicator Selection Flags
        ('use_rsi', True),
        ('use_stoch', True),
        ('use_volume', True),
        ('use_macd', True),
        ('use_candlestick', True),

        # Add RSI thresholds
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        
        # Add minimum pattern size requirements for candlesticks
        ('min_pattern_size', 0.001),  # Minimum size of candlestick body as % of price
        ('min_wick_size', 0.0015),    # Minimum size of wick as % of price
    )

    def __init__(self):
        logger.debug(f"Initializing Strategy with params: {self.params}")
        # Pre-calculate and cache indicators
        self.indicators = {}
        
        # Initialize Bollinger Bands with caching
        self.indicators['bb'] = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.p.n,
            devfactor=self.p.ndev
        )
        self.bb_top = self.indicators['bb'].lines.top
        self.bb_bot = self.indicators['bb'].lines.bot
        self.bb_mid = self.indicators['bb'].lines.mid
        
        # Initialize RSI if used
        if self.p.use_rsi:
            self.indicators['rsi'] = bt.indicators.RSI(
                self.data.close,
                period=self.p.rsi_period
            )
        
        # Initialize Stochastic with caching
        if self.p.use_stoch:
            stoch = bt.indicators.Stochastic(
                self.data,
                period=self.p.stoch_k,
                period_dfast=self.p.stoch_d,
                period_dslow=self.p.stoch_smooth,
                movav=bt.indicators.SMA,
            )
            self.indicators['stoch_k'] = stoch.percK
            self.indicators['stoch_d'] = stoch.percD
            self.indicators['stoch_crossover'] = bt.indicators.CrossOver(
                self.indicators['stoch_k'], 
                self.indicators['stoch_d']
            )
        
        # Initialize Volume MA with caching
        if self.p.use_volume:
            self.indicators['volume_ma'] = bt.indicators.SMA(
                self.data.volume,
                period=self.p.volume_ma_period
            )
        
        # Initialize MACD with caching
        if self.p.use_macd:
            macd = bt.indicators.MACDHisto(
                self.data.close,
                period_me1=self.p.macd_fast,
                period_me2=self.p.macd_slow,
                period_signal=self.p.macd_signal
            )
            self.indicators['macd_histogram'] = macd.histo
            self.indicators['macd_crossover'] = bt.indicators.CrossOver(
                self.indicators['macd_histogram'], 
                0
            )
        
        # Pre-allocate arrays for performance
        self.trades_list = []
        self.long_signals = 0
        self.short_signals = 0
    
    def _is_hammer(self):
        """
        Enhanced hammer pattern detection with size requirements and proper proportions.
        A hammer should have:
        1. Small upper body
        2. Long lower wick (2-3x body size)
        3. Minimal upper wick
        4. Minimum size requirements to filter out noise
        """
        current_open = self.data.open[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_close = self.data.close[0]
        
        # Calculate body and wicks
        body = abs(current_open - current_close)
        body_pct = body / current_close  # Body size as percentage of price
        
        lower_wick = min(current_open, current_close) - current_low
        lower_wick_pct = lower_wick / current_close
        
        upper_wick = current_high - max(current_open, current_close)
        upper_wick_pct = upper_wick / current_close
        
        # Check if pattern meets size requirements
        return (
            body_pct >= self.p.min_pattern_size and          # Meaningful body size
            lower_wick_pct >= self.p.min_wick_size and      # Meaningful lower wick
            lower_wick > 2 * body and                       # Lower wick 2-3x body
            lower_wick <= 3 * body and
            upper_wick <= 0.5 * body and                    # Minimal upper wick
            current_close > current_open                     # Bullish close
        )
    
    def _is_shooting_star(self):
        """
        Enhanced shooting star pattern detection.
        A shooting star should have:
        1. Small lower body
        2. Long upper wick (2-3x body size)
        3. Minimal lower wick
        4. Minimum size requirements to filter out noise
        """
        current_open = self.data.open[0]
        current_high = self.data.high[0]
        current_low = self.data.low[0]
        current_close = self.data.close[0]
        
        # Calculate body and wicks
        body = abs(current_open - current_close)
        body_pct = body / current_close
        
        lower_wick = min(current_open, current_close) - current_low
        lower_wick_pct = lower_wick / current_close
        
        upper_wick = current_high - max(current_open, current_close)
        upper_wick_pct = upper_wick / current_close
        
        # Check if pattern meets size requirements
        return (
            body_pct >= self.p.min_pattern_size and          # Meaningful body size
            upper_wick_pct >= self.p.min_wick_size and      # Meaningful upper wick
            upper_wick > 2 * body and                       # Upper wick 2-3x body
            upper_wick <= 3 * body and
            lower_wick <= 0.5 * body and                    # Minimal lower wick
            current_close < current_open                     # Bearish close
        )
    
    def _check_volume_confirmation(self):
        """
        Enhanced volume confirmation that checks for increasing volume trend
        """
        if not self.p.use_volume:
            return True
            
        current_volume = self.data.volume[0]
        current_volume_ma = self.indicators['volume_ma'][0]
        
        # Check if volume is increasing
        volume_increasing = (
            current_volume > self.data.volume[-1] and
            self.data.volume[-1] > self.data.volume[-2]
        )
        
        # Check if volume is above average and increasing
        return (current_volume > current_volume_ma * self.p.volume_threshold and 
                volume_increasing)
    
    def _calculate_position_size(self, stop_distance):
        """Calculate position size based on risk management rules."""
        if stop_distance <= 0:
            return self.p.min_size

        current_equity = self.broker.getvalue()
        
        # Determine position size based on equity level
        if current_equity < 100:
            # Use 100% of remaining equity if below $100
            position_value = current_equity
        else:
            # Use fixed $100 if equity is $100 or above
            position_value = 100.0
        
        # Apply 50x leverage
        leveraged_position = position_value * 50
        
        # Calculate size in BTC based on current price and leveraged position value
        current_price = self.data.close[0]
        size = leveraged_position / current_price
        
        # Ensure size is not less than min_size
        return max(size, self.p.min_size)
    
    def _check_rsi_condition(self, is_long: bool) -> bool:
        """Optimized RSI condition check"""
        if not self.p.use_rsi:
            return True
            
        rsi = self.indicators['rsi'][0]
        rsi_prev = self.indicators['rsi'][-1]
        
        if is_long:
            return rsi < self.p.rsi_oversold and rsi > rsi_prev
        return rsi > self.p.rsi_overbought and rsi < rsi_prev
    
    def _check_stoch_condition(self, is_long: bool) -> bool:
        """Optimized stochastic condition check"""
        if not self.p.use_stoch:
            return True
            
        stoch_k = self.indicators['stoch_k'][0]
        stoch_k_prev = self.indicators['stoch_k'][-1]
        crossover = self.indicators['stoch_crossover'][0]
        
        if is_long:
            return (stoch_k < self.p.stoch_lower and
                   stoch_k > stoch_k_prev and
                   crossover > 0)
        return (stoch_k > self.p.stoch_upper and
                stoch_k < stoch_k_prev and
                crossover < 0)
    
    def _check_macd_condition(self, is_long: bool) -> bool:
        """Optimized MACD condition check"""
        if not self.p.use_macd:
            return True
            
        hist = self.indicators['macd_histogram'][0]
        hist_prev = self.indicators['macd_histogram'][-1]
        crossover = self.indicators['macd_crossover'][0]
        
        if is_long:
            return (hist > hist_prev and
                   (hist > 0 or crossover > 0))
        return (hist < hist_prev and
                (hist < 0 or crossover < 0))
    
    def next(self):
        """Optimized strategy logic"""
        if self.position:
            return
        
        # Cache current values
        current_close = self.data.close[0]
        current_bb_top = self.bb_top[0]
        current_bb_bot = self.bb_bot[0]
        current_bb_mid = self.bb_mid[0]
        
        # Optimized BB touch detection - Make this less restrictive
        price_distance_to_lower = (current_close - current_bb_bot) / current_bb_bot
        price_distance_to_upper = (current_bb_top - current_close) / current_bb_top
        
        # Increase the threshold to make it easier to enter trades
        touching_lower = price_distance_to_lower <= self.p.bb_touch_threshold * 2
        touching_upper = price_distance_to_upper <= self.p.bb_touch_threshold * 2
        
        # Check long conditions with fewer restrictions
        if touching_lower:
            long_condition = True  # Start with True
            
            if self.p.use_candlestick:
                long_condition = long_condition and self._is_hammer()
            
            if self.p.use_macd:
                long_condition = long_condition and self._check_macd_condition(True)
            
            if long_condition:
                self._execute_long_trade(current_close, current_bb_mid)
        
        # Check short conditions with fewer restrictions
        elif touching_upper:
            short_condition = True  # Start with True
            
            if self.p.use_candlestick:
                short_condition = short_condition and self._is_shooting_star()
            
            if self.p.use_macd:
                short_condition = short_condition and self._check_macd_condition(False)
            
            if short_condition:
                self._execute_short_trade(current_close, current_bb_mid)
    
    def _execute_long_trade(self, current_close: float, current_bb_mid: float):
        """Execute long trade with optimized calculations"""
        self.long_signals += 1
        stop_loss = current_close * (1 - self.p.sl_pct)
        take_profit = max(current_close * (1 + self.p.tp_pct), current_bb_mid)
        
        size = self._calculate_position_size(current_close - stop_loss)
        self.buy_bracket(
            size=size,
            exectype=bt.Order.Market,
            stopprice=stop_loss,
            limitprice=take_profit
        )
        logger.debug(f"Long Entry - Close: {current_close:.2f}, "
                    f"Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
    
    def _execute_short_trade(self, current_close: float, current_bb_mid: float):
        """Execute short trade with optimized calculations"""
        self.short_signals += 1
        stop_loss = current_close * (1 + self.p.sl_pct)
        take_profit = min(current_close * (1 - self.p.tp_pct), current_bb_mid)
        
        size = self._calculate_position_size(stop_loss - current_close)
        self.sell_bracket(
            size=size,
            exectype=bt.Order.Market,
            stopprice=stop_loss,
            limitprice=take_profit
        )
        logger.debug(f"Short Entry - Close: {current_close:.2f}, "
                    f"Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
    
    def notify_trade(self, trade):
        if trade.isclosed:
            trade_info = {
                'Size': trade.size,
                'Entry Price': trade.price,  # Backtrader updates trade.price on close
                'Exit Price': trade.price,  # Assume same price for simplicity
                'PnL': trade.pnl
            }
            self.trades_list.append(trade_info)
            logger.debug(f"Trade closed: {trade_info}")
    
    def stop(self):
        # print(f"Indicator Usage - RSI: {self.p.use_rsi}, Stochastic: {self.p.use_stoch}, "
        #       f"Volume: {self.p.use_volume}, MACD: {self.p.use_macd}, Candlestick: {self.p.use_candlestick}")
        # print(f"Total long signals: {self.long_signals}")
        # print(f"Total short signals: {self.short_signals}")
        
        # Convert trades_list to DataFrame
        self._trades = pd.DataFrame(self.trades_list)
        logger.debug(f"Indicator Usage - RSI: {self.p.use_rsi}, Stochastic: {self.p.use_stoch}, "
                    f"Volume: {self.p.use_volume}, MACD: {self.p.use_macd}, Candlestick: {self.p.use_candlestick}")
        logger.debug(f"Total long signals: {self.long_signals}")
        logger.debug(f"Total short signals: {self.short_signals}")
        logger.debug(f"Total trades: {len(self._trades)}")
        logger.debug(f"Trades: {self._trades}")
        
def run_backtest(data, plot=False, verbose=True, **kwargs):
    """
    Run the backtest with the given data and parameters.
    
    Args:
        data: DataFrame with OHLCV data
        plot: Boolean, whether to create plots
        verbose: Boolean, whether to print detailed statistics
        **kwargs: Strategy parameters
    """
    # Create a cerebro instance
    cerebro = bt.Cerebro()
    
    # Add data
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # None since datetime is the index
        open=0,         # Column position for Open
        high=1,         # Column position for High
        low=2,          # Column position for Low
        close=3,        # Column position for Close
        volume=4,       # Column position for Volume
        openinterest=-1,  # -1 indicates no open interest data
        fromdate=data.index[0],
        todate=data.index[-1]
    )
    cerebro.adddata(data_feed)
    
    # Add strategy
    cerebro.addstrategy(BollingerBandsStrategy, **kwargs)
    
    # Set broker parameters
    initial_cash = 100.0  # Starting with $100
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0002, margin=0.02)  # 0.02% commission, 50x leverage
    cerebro.broker.set_slippage_perc(0.0001)  # 0.01% slippage
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    
    # Add observers for plotting
    if plot:
        cerebro.addobserver(bt.observers.BuySell)  # Show buy/sell points
        cerebro.addobserver(bt.observers.Value)    # Show portfolio value
        cerebro.addobserver(bt.observers.DrawDown) # Show drawdown
    
    # Run backtest
    results = cerebro.run()
    strat = results[0]
    
    # Create and save plot if requested
    if plot:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"BB_Scalping_Backtest_{timestamp}.png"
            
            cerebro.plot(
                volume=True,
                style='candlestick',
                barup='green',
                bardown='red',
                volup='green',
                voldown='red',
                grid=True,
                figsize=(20, 10)
            )
            
            logger.debug(f"Plot displayed")
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
    
    # Extract metrics with safer handling
    trades_analysis = strat.analyzers.trades.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    returns_analysis = strat.analyzers.returns.get_analysis()
    
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
        sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
    except:
        sharpe_ratio = 0.0
    
    # Calculate SQN
    def calculate_sqn(trades):
        if len(trades) < 2:
            return 0.0
        pnl = [trade['PnL'] for trade in trades]
        avg_pnl = np.mean(pnl)
        std_pnl = np.std(pnl)
        if std_pnl == 0:
            return 0.0
        return (avg_pnl / std_pnl) * math.sqrt(len(trades))
    
    sqn = calculate_sqn(strat._trades.to_dict('records'))
    
    # Format results with safe getters and explicit None handling
    formatted_results = {
        'Start': data.index[0].strftime('%Y-%m-%d'),
        'End': data.index[-1].strftime('%Y-%m-%d'),
        'Duration': f"{(data.index[-1] - data.index[0]).days} days",
        'Exposure Time [%]': 100 * (safe_get(trades_analysis, 'total', 'total', default=0) / len(data)),
        'Equity Final [$]': final_value,
        'Equity Peak [$]': final_value + (safe_get(drawdown_analysis, 'max', 'drawdown', default=0) * final_value / 100),
        'Return [%]': total_return,
        'Buy & Hold Return [%]': ((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100),
        'Max. Drawdown [%]': safe_get(drawdown_analysis, 'max', 'drawdown', default=0),
        'Avg. Drawdown [%]': safe_get(drawdown_analysis, 'average', 'drawdown', default=0),
        'Max. Drawdown Duration': safe_get(drawdown_analysis, 'max', 'len', default=0),
        'Avg. Drawdown Duration': safe_get(drawdown_analysis, 'average', 'len', default=0),
        '# Trades': safe_get(trades_analysis, 'total', 'total', default=0),
        'Win Rate [%]': (safe_get(trades_analysis, 'won', 'total', default=0) / 
                        max(safe_get(trades_analysis, 'total', 'total', default=1), 1) * 100),
        'Best Trade [%]': safe_get(trades_analysis, 'won', 'pnl', 'max', default=0),
        'Worst Trade [%]': safe_get(trades_analysis, 'lost', 'pnl', 'min', default=0),
        'Avg. Trade [%]': safe_get(trades_analysis, 'pnl', 'net', 'average', default=0),
        'Max. Trade Duration': safe_get(trades_analysis, 'len', 'max', default=0),
        'Avg. Trade Duration': safe_get(trades_analysis, 'len', 'average', default=0),
        'Profit Factor': (safe_get(trades_analysis, 'won', 'pnl', 'total', default=0) / 
                         max(abs(safe_get(trades_analysis, 'lost', 'pnl', 'total', default=1)), 1)),
        'Sharpe Ratio': float(sharpe_ratio),  # Ensure it's a float
        'SQN': sqn #,
        # '_trades': strat._trades
    }
    
    # Print detailed statistics only if verbose is True
    if verbose:
        print('\n=== Strategy Performance Report ===')
        print(f"\nPeriod: {formatted_results['Start']} - {formatted_results['End']} ({formatted_results['Duration']})")
        print(f"Initial Capital: ${initial_cash:,.2f}")
        print(f"Final Capital: ${float(formatted_results['Equity Final [$]']):,.2f}")
        print(f"Total Return: {float(formatted_results['Return [%]']):,.2f}%")
        print(f"Buy & Hold Return: {float(formatted_results['Buy & Hold Return [%]']):,.2f}%")
        print(f"\nTotal Trades: {int(formatted_results['# Trades'])}")
        print(f"Win Rate: {float(formatted_results['Win Rate [%]']):,.2f}%")
        print(f"Best Trade: {float(formatted_results['Best Trade [%]']):,.2f}%")
        print(f"Worst Trade: {float(formatted_results['Worst Trade [%]']):,.2f}%")
        print(f"Avg. Trade: {float(formatted_results['Avg. Trade [%]']):,.2f}%")
        print(f"\nMax Drawdown: {float(formatted_results['Max. Drawdown [%]']):,.2f}%")
        print(f"Sharpe Ratio: {float(formatted_results['Sharpe Ratio']):,.2f}")
        print(f"Profit Factor: {float(formatted_results['Profit Factor']):,.2f}")
        print(f"SQN: {float(formatted_results['SQN']):,.2f}")
    
    return formatted_results

def cross_validate_strategy(data, params, n_splits=5):
    """
    Perform time series cross-validation of the strategy.
    
    Args:
        data: DataFrame with OHLCV data
        params: Dictionary of strategy parameters
        n_splits: Number of splits for cross-validation
    
    Returns:
        Dictionary with cross-validation metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []
    
    for _, test_idx in tscv.split(data):
        # Split data into test set only
        test_data = data.iloc[test_idx]
        
        # Run backtest on test set
        try:
            results = run_backtest(test_data, plot=False, verbose=False, **params)
            
            cv_results.append({
                'Return [%]': results['Return [%]'],
                'Sharpe Ratio': results['Sharpe Ratio'],
                'Max. Drawdown [%]': results['Max. Drawdown [%]'],
                'Win Rate [%]': results['Win Rate [%]'],
                '# Trades': results['# Trades']
            })
        except Exception as e:
            logger.error(f"Cross-validation failed for split: {str(e)}")
            continue
    
    # Calculate mean and std of metrics
    metrics = {}
    for key in cv_results[0].keys():
        values = [result[key] for result in cv_results]
        metrics[f'{key} Mean'] = np.mean(values)
        metrics[f'{key} Std'] = np.std(values)
    
    return metrics

def calculate_complexity_penalty(params: Dict) -> float:
    """
    Calculate a simpler complexity penalty based on number of active indicators
    and parameter values.
    """
    # Count active indicators
    active_indicators = sum([
        params['use_rsi'],
        params['use_stoch'],
        params['use_volume'],
        params['use_macd'],
        params['use_candlestick']
    ])
    
    # Base penalty on number of active indicators
    base_penalty = 0.1 * (active_indicators / 3)  # Normalized to max 3 indicators
    
    return base_penalty

def detect_market_regime(data, window=20):
    """Simple market regime detection based on volatility."""
    returns = data['close'].pct_change()
    volatility = returns.rolling(window).std()
    mean_vol = volatility.mean()
    
    regimes = []
    for vol in volatility:
        if np.isnan(vol):
            regimes.append('unknown')
        elif vol < mean_vol * 0.75:
            regimes.append('low_volatility')
        elif vol > mean_vol * 1.25:
            regimes.append('high_volatility')
        else:
            regimes.append('normal_volatility')
    
    return pd.Series(regimes, index=data.index)

def evaluate(individual, data, cv_splits=5):
    """Enhanced evaluation function with anti-overfitting measures."""
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
            0.01 <= params['tp_pct'] <= 0.04):
        return (-np.inf,)
    
    try:
        # Perform cross-validation
        cv_metrics = cross_validate_strategy(data, params, n_splits=cv_splits)
        
        # Extract mean metrics for fitness calculation
        mean_sharpe = cv_metrics['Sharpe Ratio Mean']
        mean_return = cv_metrics['Return [%] Mean']
        mean_trades = cv_metrics['# Trades Mean']
        
        # Consider stability (lower std is better)
        sharpe_stability = 1 / (1 + cv_metrics['Sharpe Ratio Std'])
        return_stability = 1 / (1 + cv_metrics['Return [%] Std'])
        
        if mean_trades == 0:
            return (-np.inf,)
        
        # Normalize metrics
        norm_sharpe = mean_sharpe / 3.0 if not np.isnan(mean_sharpe) else 0
        norm_return = mean_return / 100.0 if not np.isnan(mean_return) else 0
        norm_trades = min(mean_trades / 1000.0, 1.0)
        
        # Calculate base fitness
        base_fitness = (
            norm_sharpe * 0.2 * sharpe_stability +
            norm_return * 0.4 * return_stability +
            norm_trades * 0.4
        )
        
        # Calculate complexity penalty
        complexity_penalty = calculate_complexity_penalty(params)
        
        # Detect market regimes
        regimes = detect_market_regime(data)
        regime_performances = []
        
        # Test performance in different regimes
        for regime in ['low_volatility', 'normal_volatility', 'high_volatility']:
            regime_data = data[regimes == regime]
            if len(regime_data) > 100:  # Minimum data points required
                regime_results = run_backtest(regime_data, plot=False, verbose=False, **params)
                regime_performances.append(regime_results['Return [%]'])
        
        # Penalize high variance across regimes
        regime_stability = 1 / (1 + np.std(regime_performances)) if regime_performances else 0
        
        # Calculate final fitness with all penalties
        fitness = base_fitness * (1 - complexity_penalty) * regime_stability
        
        if np.isnan(fitness) or np.isinf(fitness):
            return (-np.inf,)
            
        return (fitness,)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return (-np.inf,)

def mutate_and_clip(individual, mu=0, sigma=0.1, indpb=0.2):
    """Custom mutation function to handle parameter bounds and indicator flags."""
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

def walk_forward_analysis(data: pd.DataFrame, 
                         params: dict, 
                         train_size: int = 30,
                         test_size: int = 7,
                         step_size: int = 7) -> Dict:
    """
    Perform walk-forward analysis on the strategy.
    
    Args:
        data: DataFrame with OHLCV data
        params: Strategy parameters
        train_size: Number of days for training period
        test_size: Number of days for test period
        step_size: Number of days to step forward
    
    Returns:
        Dictionary with walk-forward results
    """
    # Convert days to data points (assuming 1-minute data)
    points_per_day = 24 * 60  # minutes per day
    train_points = train_size * points_per_day
    test_points = test_size * points_per_day
    step_points = step_size * points_per_day
    
    results = []
    start_idx = 0
    
    while start_idx + train_points + test_points <= len(data):
        # Split data into train and test sets
        train_data = data.iloc[start_idx:start_idx + train_points]
        test_data = data.iloc[start_idx + train_points:start_idx + train_points + test_points]
        
        # Optimize parameters on training data
        train_results = run_backtest(train_data, plot=False, verbose=False, **params)
        
        # Test optimized parameters on test data
        test_results = run_backtest(test_data, plot=False, verbose=False, **params)
        
        results.append({
            'train_period': (train_data.index[0], train_data.index[-1]),
            'test_period': (test_data.index[0], test_data.index[-1]),
            'train_return': train_results['Return [%]'],
            'test_return': test_results['Return [%]'],
            'train_sharpe': train_results['Sharpe Ratio'],
            'test_sharpe': test_results['Sharpe Ratio'],
            'trades': test_results['# Trades']
        })
        
        # Move forward by step size
        start_idx += step_points
    
    return analyze_walk_forward_results(results)

def analyze_walk_forward_results(results: List[Dict]) -> Dict:
    """
    Analyze results from walk-forward testing.
    """
    train_returns = [r['train_return'] for r in results]
    test_returns = [r['test_return'] for r in results]
    train_sharpes = [r['train_sharpe'] for r in results]
    test_sharpes = [r['test_sharpe'] for r in results]
    
    # Calculate stability metrics
    return_stability = 1 - abs(np.mean(test_returns) - np.mean(train_returns)) / np.mean(train_returns)
    sharpe_stability = 1 - abs(np.mean(test_sharpes) - np.mean(train_sharpes)) / np.mean(train_sharpes)
    
    return {
        'avg_train_return': np.mean(train_returns),
        'avg_test_return': np.mean(test_returns),
        'avg_train_sharpe': np.mean(train_sharpes),
        'avg_test_sharpe': np.mean(test_sharpes),
        'return_stability': return_stability,
        'sharpe_stability': sharpe_stability,
        'consistency_score': len([r for r in test_returns if r > 0]) / len(test_returns)
    }

def validate_params(params: Dict) -> bool:
    """Validate parameter ranges and combinations"""
    # Basic range validation
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
            0.01 <= params['tp_pct'] <= 0.04):
        return False
    
    # Logical validation
    if params['stoch_lower'] >= params['stoch_upper']:
        return False
    
    # Prevent excessive indicator usage
    active_indicators = sum([
        params['use_rsi'],
        params['use_stoch'],
        params['use_volume'],
        params['use_macd'],
        params['use_candlestick']
    ])
    if active_indicators > 3:  # Limit to maximum 3 indicators
        return False
    
    return True

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv(
        'data/bybit-BTCUSDT-1m-20240915-to-20241114.csv'
    )
    
    # Select only the required columns and rename them to match Backtrader's expectations
    data = data[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Validate required columns
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Standardize column names to lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Convert datetime string to pandas datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Set datetime as index
    data.set_index('datetime', inplace=True)
    
    # Sort index to ensure chronological order
    data.sort_index(inplace=True)
    
    # Drop any duplicate indices
    data = data[~data.index.duplicated(keep='first')]
    
    # Forward fill any NaN values
    data.fillna(method='ffill', inplace=True)
    
    # Drop any remaining NaN values
    data.dropna(inplace=True)
    

    
    # Run backtest with fixed parameters
    fixed_params = {
        'n': 20,              # Standard BB period
        'ndev': 2.0,          # Standard deviation
        'rsi_period': 14,
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_smooth': 3,
        'stoch_upper': 80,
        'stoch_lower': 20,
        'volume_ma_period': 20,
        'sl_pct': 0.02,       # Wider stop loss
        'tp_pct': 0.04,       # Wider take profit
        'use_rsi': False,     # Simplify by using fewer indicators
        'use_stoch': False,
        'use_volume': False,
        'use_macd': True,
        'use_candlestick': True,
        'bb_touch_threshold': 0.002  # More lenient threshold
    }
    
    print("\nRunning backtest with fixed parameters...")
    results_fixed = run_backtest(data, plot=False, verbose=True, **fixed_params)
    print("\nBacktest results with fixed parameters:")
    print(results_fixed)
    
    # Perform walk-forward optimization
    print("\nStarting walk-forward optimization...")
    best_params, logbook = optimize_strategy(data)
    
    # Validate results on the entire dataset
    print("\nValidating best parameters...")
    wf_results = walk_forward_analysis(data, best_params)
    
    print("\nWalk-Forward Analysis Results:")
    print(f"Average Test Return: {wf_results['avg_test_return']:.2f}%")
    print(f"Return Stability: {wf_results['return_stability']:.2f}")
    print(f"Strategy Consistency: {wf_results['consistency_score']:.2f}")
    
    # Final backtest with optimized parameters
    print("\nRunning final backtest with optimized parameters...")
    results_optimized = run_backtest(data, plot=True, verbose=True, **best_params)
    print("\nBacktest results with optimized parameters:")
    print(results_optimized)