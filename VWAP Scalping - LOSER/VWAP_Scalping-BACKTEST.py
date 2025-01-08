import backtrader as bt
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, time
import random
from deap import base, creator, tools, algorithms
import multiprocessing
import matplotlib.pyplot as plt
import logging
import json

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------- Logging Configuration -------------------
logging.basicConfig(
    filename='strategy_errors.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger()

# ------------------- Custom PercentChange Indicator -------------------
class PercentChange(bt.Indicator):
    """
    Custom Percentage Change Indicator.
    Calculates the percentage change in closing prices over a specified period.
    """
    lines = ('pct_change',)
    params = (('period', 1), )

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        try:
            previous_close = self.data.close[-self.p.period]
            current_close = self.data.close[0]
            if previous_close != 0:
                self.lines.pct_change[0] = ((current_close - previous_close) / previous_close) * 100
            else:
                self.lines.pct_change[0] = 0.0
        except Exception as e:
            logger.error(f"Error in PercentChange.next(): {str(e)}")
            self.lines.pct_change[0] = 0.0

# ------------------- Custom VWAP Indicator -------------------
class VWAP(bt.Indicator):
    """
    Custom Volume Weighted Average Price (VWAP) Indicator.
    Calculates VWAP over a specified rolling period.
    """
    lines = ('vwap',)
    params = (('period', 20), )

    def __init__(self):
        self.addminperiod(self.params.period)
        self.typical_price = (self.data.high + self.data.low + self.data.close) / 3
        self.pv = self.typical_price * self.data.volume

        # Use Simple Moving Average to calculate rolling sums
        self.cum_pv = bt.indicators.SimpleMovingAverage(self.pv, period=self.params.period)
        self.cum_volume = bt.indicators.SimpleMovingAverage(self.data.volume, period=self.params.period)

    def next(self):
        try:
            if self.cum_volume[0] != 0:
                self.lines.vwap[0] = self.cum_pv[0] / self.cum_volume[0]
            else:
                self.lines.vwap[0] = float('nan')
        except Exception as e:
            logger.error(f"Error in VWAP.next(): {str(e)}")
            self.lines.vwap[0] = float('nan')

# ------------------- VWAPScalping Strategy -------------------
class VWAPScalping(bt.Strategy):
    params = (
        ('vwap_period', 20),
        ('rsi_period', 27),
        ('macd_fast', 9),
        ('macd_slow', 17),
        ('macd_signal', 8),
        ('bb_period', 25),
        ('bb_std_dev', 2.5),
        ('atr_period', 20),
        ('sar_step', 0.03),
        ('sar_max', 0.221),
        ('stop_loss_pct', 0.01),
        ('take_profit_pct', 0.03),
        ('rsi_fixed_overbought', 70),
        ('rsi_fixed_oversold', 30),
        ('rsi_atr_multiplier', 2),
        ('min_volume_multiplier', 0.2),
        ('volume_ma_period', 20),
        ('max_position_size', 100),
        ('trailing_stop_atr_multiplier', 2.0),
        ('risk_atr_multiplier', 1.5),
        ('min_trading_hour', 2),
        ('max_trading_hour', 22),
        ('high_volatility_percentile', 75.0),
        ('volatility_lookback', 100),
        # Indicator usage flags
        ('use_rsi', True),
        ('use_macd', True),
        ('use_bollinger_bands', True),
        ('use_atr', True),
        ('use_sar', True),
    )

    def __init__(self):
        try:
            # Initialize Indicators
            self.vwap = VWAP(self.data, period=self.p.vwap_period)

            if self.p.use_rsi:
                self.rsi = bt.indicators.RSI(
                    self.data.close, period=self.p.rsi_period)

            if self.p.use_macd:
                self.macd = bt.indicators.MACD(
                    self.data.close,
                    period_me1=self.p.macd_fast,
                    period_me2=self.p.macd_slow,
                    period_signal=self.p.macd_signal
                )
                self.macd_signal = self.macd.signal
                self.macd_hist = self.macd.macd - self.macd.signal

            if self.p.use_bollinger_bands:
                self.bb = bt.indicators.BollingerBands(
                    self.data.close, period=self.p.bb_period, devfactor=self.p.bb_std_dev)
                self.bb_upper = self.bb.top
                self.bb_middle = self.bb.mid
                self.bb_lower = self.bb.bot

            if self.p.use_atr:
                self.atr = bt.indicators.ATR(
                    self.data, period=self.p.atr_period)

            if self.p.use_sar:
                self.sar = bt.indicators.ParabolicSAR(
                    self.data, af=self.p.sar_step, afmax=self.p.sar_max)

            self.volume_ma = bt.indicators.SimpleMovingAverage(
                self.data.volume, period=self.p.volume_ma_period)

            # Volatility calculation
            self.returns = PercentChange(
                self.data, period=1)
            self.volatility = bt.indicators.StandardDeviation(
                self.returns, period=self.p.volatility_lookback) * np.sqrt(356 * 60)  # Assuming 356 trading days

            # Trailing stop variables
            self.trailing_stop = None

            # Position tracking variables
            self.entry_price = None
            self.position_atr = None

            # Trade recording
            self.trades = []
        except Exception as e:
            logger.error(f"Error in VWAPScalping.__init__: {str(e)}")

    def log(self, txt, dt=None):
        """ Logging function for this strategy - Logs only errors """
        dt = dt or self.datas[0].datetime.datetime(0)
        logger.error(f'{dt} - {txt}')

    def calculate_position_size(self, current_price):
        try:
            current_equity = self.broker.getvalue()
            
            # Determine position size based on equity level
            if current_equity < 100:
                # Use 100% of remaining equity if below $100
                position_value = current_equity
            else:
                # Use fixed $100 if equity is $100 or above
                position_value = 100.0
            
            position_size = position_value / current_price
            return int(position_size)
        except Exception as e:
            self.log(f"Error in calculate_position_size: {str(e)}")
            return 0

    def get_dynamic_risk_parameters(self, current_atr):
        if self.p.use_atr and hasattr(self, 'atr'):
            try:
                # Ensure enough bars
                if len(self) < self.p.volatility_lookback:
                    return self.p.stop_loss_pct, self.p.take_profit_pct

                # Get past volatility
                past_volatility = [self.volatility[-i] for i in range(1, self.p.volatility_lookback + 1)]
                mean_recent_volatility = np.nanmean(past_volatility)
                if mean_recent_volatility == 0:
                    volatility_factor = 1.0
                else:
                    volatility_factor = self.volatility[0] / mean_recent_volatility

                dynamic_stop_loss = max(
                    self.p.stop_loss_pct,
                    self.p.stop_loss_pct * volatility_factor
                )
                dynamic_take_profit = max(
                    self.p.take_profit_pct,
                    self.p.take_profit_pct * volatility_factor
                )

                min_stop_distance = current_atr * self.p.risk_atr_multiplier
                final_stop_loss = max(dynamic_stop_loss, min_stop_distance / self.data.close[0])
                final_take_profit = max(dynamic_take_profit, (min_stop_distance * 2) / self.data.close[0])

                return final_stop_loss, final_take_profit
            except Exception as e:
                self.log(f"Error in get_dynamic_risk_parameters: {str(e)}")
                return self.p.stop_loss_pct, self.p.take_profit_pct
        else:
            return self.p.stop_loss_pct, self.p.take_profit_pct

    def is_valid_trading_time(self):
        try:
            current_dt = self.datas[0].datetime.datetime(0)
            current_time = current_dt.time()

            # Check if it's a weekend
            if current_dt.weekday() >= 5:
                return False

            # Check trading hours
            if not (time(self.p.min_trading_hour, 0) <= current_time <= time(self.p.max_trading_hour, 0)):
                return False

            # Check market volatility
            if self.p.use_atr and hasattr(self, 'volatility'):
                if len(self) >= self.p.volatility_lookback:
                    past_volatility = self.volatility[-self.p.volatility_lookback]
                    if past_volatility is not None and self.volatility[0] > past_volatility * 2:
                        return False

            return True
        except Exception as e:
            self.log(f"Error in is_valid_trading_time: {str(e)}")
            return False

    def update_trailing_stop(self):
        try:
            if not self.position:
                return

            current_price = self.data.close[0]
            if self.p.use_atr and hasattr(self, 'atr'):
                current_atr = self.atr[0]
            else:
                current_atr = 0

            if self.position.size > 0:  # Long position
                new_stop = current_price - (current_atr * self.p.trailing_stop_atr_multiplier)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    # Update stop loss order
                    self.sell(
                        exectype=bt.Order.Stop,
                        price=self.trailing_stop,
                        size=self.position.size,
                        parent=self.position
                    )
            elif self.position.size < 0:  # Short position
                new_stop = current_price + (current_atr * self.p.trailing_stop_atr_multiplier)
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                    # Update stop loss order
                    self.buy(
                        exectype=bt.Order.Stop,
                        price=self.trailing_stop,
                        size=abs(self.position.size),
                        parent=self.position
                    )
        except Exception as e:
            self.log(f"Error in update_trailing_stop: {str(e)}")

    def next(self):
        try:
            if not self.is_valid_trading_time():
                return

            current_price = self.data.close[0]
            if self.p.use_atr and hasattr(self, 'atr'):
                current_atr = self.atr[0]
            else:
                current_atr = 0
            current_volume = self.data.volume[0]
            current_volume_ma = self.volume_ma[0]

            min_volume_threshold = current_volume_ma * self.p.min_volume_multiplier

            if current_volume < min_volume_threshold:
                return

            # Update trailing stop for existing position
            if self.position:
                self.update_trailing_stop()

            # Get dynamic risk parameters
            stop_loss_pct, take_profit_pct = self.get_dynamic_risk_parameters(current_atr)

            # Retrieve current indicator values
            current_vwap = self.vwap[0]
            current_rsi = self.rsi[0] if self.p.use_rsi and hasattr(self, 'rsi') else None
            current_macd = self.macd[0] if self.p.use_macd and hasattr(self, 'macd') else None
            current_macd_signal = self.macd_signal[0] if self.p.use_macd and hasattr(self, 'macd_signal') else None
            current_bb_upper = self.bb_upper[0] if self.p.use_bollinger_bands and hasattr(self, 'bb_upper') else None
            current_bb_lower = self.bb_lower[0] if self.p.use_bollinger_bands and hasattr(self, 'bb_lower') else None
            current_sar = self.sar[0] if self.p.use_sar and hasattr(self, 'sar') else None

            # Retrieve previous bar's indicator values for crossover detection
            if len(self) < 2:
                return
            previous_price = self.data.close[-1]
            previous_vwap = self.vwap[-1]
            previous_macd = self.macd[-1] if self.p.use_macd and hasattr(self, 'macd') else None
            previous_macd_signal = self.macd_signal[-1] if self.p.use_macd and hasattr(self, 'macd_signal') else None

            # Dynamic Overbought/Oversold Levels
            if self.p.use_rsi and current_rsi is not None:
                if self.p.use_atr and hasattr(self, 'atr'):
                    dynamic_overbought = min(self.p.rsi_fixed_overbought + (self.p.rsi_atr_multiplier * current_atr), 100)
                    dynamic_oversold = max(self.p.rsi_fixed_oversold - (self.p.rsi_atr_multiplier * current_atr), 0)
                else:
                    dynamic_overbought = self.p.rsi_fixed_overbought
                    dynamic_oversold = self.p.rsi_fixed_oversold
            else:
                dynamic_overbought = dynamic_oversold = None

            # Long Entry Conditions
            price_crosses_above_vwap = previous_price < previous_vwap and current_price > current_vwap
            rsi_condition = current_rsi < dynamic_overbought if dynamic_overbought is not None else True
            macd_crossover = (previous_macd < previous_macd_signal and current_macd > current_macd_signal) if self.p.use_macd and hasattr(self, 'macd') else True
            within_bollinger_bands = (current_price > current_bb_lower and current_price < current_bb_upper) if self.p.use_bollinger_bands and hasattr(self, 'bb_lower') else True
            sar_condition = current_sar < current_price if self.p.use_sar and hasattr(self, 'sar') else True

            # Short Entry Conditions
            price_crosses_below_vwap = previous_price > previous_vwap and current_price < current_vwap
            rsi_condition_short = current_rsi > dynamic_oversold if dynamic_oversold is not None else True
            macd_crossover_short = (previous_macd > previous_macd_signal and current_macd < current_macd_signal) if self.p.use_macd and hasattr(self, 'macd') else True
            within_bollinger_bands_short = (current_price > current_bb_lower and current_price < current_bb_upper) if self.p.use_bollinger_bands and hasattr(self, 'bb_lower') else True
            sar_condition_short = current_sar > current_price if self.p.use_sar and hasattr(self, 'sar') else True

            # Long Entry
            if (price_crosses_above_vwap and
                rsi_condition and
                macd_crossover and
                within_bollinger_bands and
                sar_condition and
                current_volume >= min_volume_threshold and
                not self.position):

                position_size = self.calculate_position_size(current_price)

                if position_size > 0:
                    self.buy(
                        size=position_size,
                        exectype=bt.Order.Market
                    )
                    self.entry_price = current_price
                    self.position_atr = current_atr
                    self.trailing_stop = current_price - (current_atr * self.p.trailing_stop_atr_multiplier)
                    # Log only critical errors or significant events if necessary

            # Short Entry
            elif (price_crosses_below_vwap and
                  rsi_condition_short and
                  macd_crossover_short and
                  within_bollinger_bands_short and
                  sar_condition_short and
                  current_volume >= min_volume_threshold and
                  not self.position):

                position_size = self.calculate_position_size(current_price)

                if position_size > 0:
                    self.sell(
                        size=position_size,
                        exectype=bt.Order.Market
                    )
                    self.entry_price = current_price
                    self.position_atr = current_atr
                    self.trailing_stop = current_price + (current_atr * self.p.trailing_stop_atr_multiplier)
                    # Log only critical errors or significant events if necessary
        except Exception as e:
            self.log(f"Error in next: {str(e)}")

# ------------------- Evaluation Function -------------------
def evaluate_individual(individual, data_df):
    # Unpack individual genes
    (vwap_period,
     rsi_period,
     macd_fast,
     macd_slow,
     macd_signal,
     bb_period,
     bb_std_dev,
     atr_period,
     sar_step,
     sar_max,
     stop_loss_pct,
     take_profit_pct,
     use_rsi,
     use_macd,
     use_bollinger_bands,
     use_atr,
     use_sar) = individual

    # Validate MACD parameters
    if macd_fast >= macd_slow:
        return 0.0, 0.0, 0.0, 0  # Return zeros instead of -np.inf

    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Create data feed
    try:
        data_bt = bt.feeds.PandasData(
            dataname=data_df,
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            datetime='datetime',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        cerebro.adddata(data_bt)
    except Exception as e:
        logger.error(f"Error adding data feed: {str(e)}")
        return 0.0, 0.0, 0.0, 0

    # Add strategy with parameters
    try:
        cerebro.addstrategy(
            VWAPScalping,
            vwap_period=vwap_period,
            rsi_period=rsi_period,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_period=bb_period,
            bb_std_dev=bb_std_dev,
            atr_period=atr_period,
            sar_step=sar_step,
            sar_max=sar_max,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_rsi=bool(use_rsi),
            use_macd=bool(use_macd),
            use_bollinger_bands=bool(use_bollinger_bands),
            use_atr=bool(use_atr),
            use_sar=bool(use_sar)
        )
    except Exception as e:
        logger.error(f"Error adding strategy: {str(e)}")
        return 0.0, 0.0, 0.0, 0

    # Set broker parameters
    try:
        cerebro.broker.setcash(100.0)
        cerebro.broker.setcommission(commission=0.002, margin=1.0 / 50)
    except Exception as e:
        logger.error(f"Error setting broker parameters: {str(e)}")
        return 0.0, 0.0, 0.0, 0

    # Add analyzers to Cerebro
    try:
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    except Exception as e:
        logger.error(f"Error adding analyzers: {str(e)}")
        return 0.0, 0.0, 0.0, 0

    # Run backtest
    try:
        results = cerebro.run(runonce=False, stdstats=False)
        if not results:
            return 0.0, 0.0, 0.0, 0

        strat = results[0]

    except Exception as e:
        logger.error(f"Error during backtest run: {str(e)}")
        return 0.0, 0.0, 0.0, 0

    # Extract performance metrics with better error handling
    try:
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe = sharpe_analysis.get('sharperatio', 0.0)
        # Convert None to 0.0
        sharpe = 0.0 if sharpe is None else float(sharpe)
    except Exception as e:
        logger.error(f"Error extracting Sharpe Ratio: {str(e)}")
        sharpe = 0.0

    try:
        returns_analysis = strat.analyzers.returns.get_analysis()
        returns = returns_analysis.get('rnorm100', 0.0)
        # Convert None to 0.0
        returns = 0.0 if returns is None else float(returns)
    except Exception as e:
        logger.error(f"Error extracting Returns: {str(e)}")
        returns = 0.0

    try:
        trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
        total_trades = trade_analyzer.total.closed if hasattr(trade_analyzer, 'total') and hasattr(trade_analyzer.total, 'closed') else 0
        win_trades = trade_analyzer.won.total if hasattr(trade_analyzer, 'won') else 0
        win_rate = (win_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
    except Exception as e:
        logger.error(f"Error extracting Trade Analyzer metrics: {str(e)}")
        total_trades = 0
        win_rate = 0.0

    # Ensure all values are valid numbers
    try:
        returns = float(returns) if returns is not None else 0.0
        sharpe = float(sharpe) if sharpe is not None else 0.0
        win_rate = float(win_rate) if win_rate is not None else 0.0
        total_trades = int(total_trades) if total_trades is not None else 0
    except Exception as e:
        logger.error(f"Error converting metrics to float/int: {str(e)}")
        returns = 0.0
        sharpe = 0.0
        win_rate = 0.0
        total_trades = 0

    return returns, sharpe, win_rate, total_trades

# ------------------- DEAP Setup -------------------
# Define the evaluation as multi-objective (maximize all)
try:
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))  # returns, sharpe, win_rate, num_trades
except Exception as e:
    logger.error(f"Error creating FitnessMulti: {str(e)}")

try:
    creator.create("Individual", list, fitness=creator.FitnessMulti)
except Exception as e:
    logger.error(f"Error creating Individual: {str(e)}")

toolbox = base.Toolbox()

# Attribute generators
toolbox.register("vwap_period", random.randint, 10, 50)
toolbox.register("rsi_period", random.randint, 10, 40)
toolbox.register("macd_fast", random.randint, 5, 20)
toolbox.register("macd_slow", random.randint, 21, 40)
toolbox.register("macd_signal", random.randint, 5, 15)
toolbox.register("bb_period", random.randint, 10, 40)
toolbox.register("bb_std_dev", random.uniform, 1.5, 3.5)
toolbox.register("atr_period", random.randint, 10, 30)
toolbox.register("sar_step", random.uniform, 0.02, 0.1)
toolbox.register("sar_max", random.uniform, 0.1, 0.3)
toolbox.register("stop_loss_pct", random.uniform, 0.005, 0.02)
toolbox.register("take_profit_pct", random.uniform, 0.015, 0.05)
toolbox.register("use_rsi", random.randint, 0, 1)
toolbox.register("use_macd", random.randint, 0, 1)
toolbox.register("use_bollinger_bands", random.randint, 0, 1)
toolbox.register("use_atr", random.randint, 0, 1)
toolbox.register("use_sar", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.vwap_period,
                  toolbox.rsi_period,
                  toolbox.macd_fast,
                  toolbox.macd_slow,
                  toolbox.macd_signal,
                  toolbox.bb_period,
                  toolbox.bb_std_dev,
                  toolbox.atr_period,
                  toolbox.sar_step,
                  toolbox.sar_max,
                  toolbox.stop_loss_pct,
                  toolbox.take_profit_pct,
                  toolbox.use_rsi,
                  toolbox.use_macd,
                  toolbox.use_bollinger_bands,
                  toolbox.use_atr,
                  toolbox.use_sar), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# ------------------- Optimization Function -------------------
def optimize_strategy(data_df, ngen=20, pop_size=50):
    # Define the evaluate function with partial
    toolbox.register("evaluate_opt", evaluate_individual, data_df=data_df)
    toolbox.register("evaluate", toolbox.evaluate_opt)

    # Create initial population
    population = toolbox.population(n=pop_size)

    # Define statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # Define Hall of Fame
    hof = tools.HallOfFame(10)

    # Setup multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Run the genetic algorithm
    try:
        algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size*2, cxpb=0.7, mutpb=0.2,
                                   ngen=ngen, stats=stats, halloffame=hof, verbose=True)
    except Exception as e:
        logger.error(f"Error during genetic algorithm execution: {str(e)}")

    pool.close()
    pool.join()

    # Persist results
    try:
        hof_list = []
        for ind in hof:
            hof_list.append({
                'parameters': ind,
                'fitness': ind.fitness.values
            })
        with open('hof_results.json', 'w') as f:
            json.dump(hof_list, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x, indent=4)
    except Exception as e:
        logger.error(f"Error saving Hall of Fame results: {str(e)}")

    return hof, population, stats

# ------------------- Main Execution -------------------
if __name__ == '__main__':
    try:
        # Load data once
        data_path = r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20240915-to-20241114.csv'
        data_df = pd.read_csv(data_path)
        data_df.columns = [col.lower() for col in data_df.columns]
        data_df['datetime'] = pd.to_datetime(data_df['datetime'])
        data_df.sort_values('datetime', inplace=True)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        exit(1)

    # Run optimization with more generations and larger population
    hof, population, stats = optimize_strategy(data_df=data_df, ngen=30, pop_size=100)

    # Print the best individuals
    print("\n=== Best Individuals ===")
    for i, individual in enumerate(hof):
        print(f"Individual {i+1}: {individual}, Fitness: {individual.fitness.values}")

    # Select the best individual
    if hof:
        best_individual = hof[0]
        print("\n=== Best Individual ===")
        print(f"{best_individual}, Fitness: {best_individual.fitness.values}")

        # Unpack the best individual
        (best_vwap_period,
         best_rsi_period,
         best_macd_fast,
         best_macd_slow,
         best_macd_signal,
         best_bb_period,
         best_bb_std_dev,
         best_atr_period,
         best_sar_step,
         best_sar_max,
         best_stop_loss_pct,
         best_take_profit_pct,
         best_use_rsi,
         best_use_macd,
         best_use_bollinger_bands,
         best_use_atr,
         best_use_sar) = best_individual

        # Initialize Cerebro for final backtest with best parameters
        cerebro = bt.Cerebro()

        # Create data feed
        try:
            data_bt = bt.feeds.PandasData(
                dataname=data_df,
                timeframe=bt.TimeFrame.Minutes,
                compression=1,
                datetime='datetime',
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest=None
            )
            cerebro.adddata(data_bt)
        except Exception as e:
            logger.error(f"Error adding data feed for final backtest: {str(e)}")
            exit(1)

        # Add analyzers to Cerebro
        try:
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        except Exception as e:
            logger.error(f"Error adding analyzers for final backtest: {str(e)}")
            exit(1)

        # Add strategy with best parameters
        try:
            cerebro.addstrategy(
                VWAPScalping,
                vwap_period=best_vwap_period,
                rsi_period=best_rsi_period,
                macd_fast=best_macd_fast,
                macd_slow=best_macd_slow,
                macd_signal=best_macd_signal,
                bb_period=best_bb_period,
                bb_std_dev=best_bb_std_dev,
                atr_period=best_atr_period,
                sar_step=best_sar_step,
                sar_max=best_sar_max,
                stop_loss_pct=best_stop_loss_pct,
                take_profit_pct=best_take_profit_pct,
                use_rsi=bool(best_use_rsi),
                use_macd=bool(best_use_macd),
                use_bollinger_bands=bool(best_use_bollinger_bands),
                use_atr=bool(best_use_atr),
                use_sar=bool(best_use_sar)
            )
        except Exception as e:
            logger.error(f"Error adding strategy for final backtest: {str(e)}")
            exit(1)

        # Set broker parameters
        try:
            cerebro.broker.setcash(100.0)
            cerebro.broker.setcommission(commission=0.002, margin=1.0 / 50)
        except Exception as e:
            logger.error(f"Error setting broker parameters for final backtest: {str(e)}")
            exit(1)

        # Print starting conditions
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Run backtest
        try:
            results = cerebro.run()
            strat = results[0]
        except Exception as e:
            logger.error(f"Error running final backtest: {str(e)}")
            strat = None

        if strat:
            # Print final conditions
            print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

            # Extract and print analyzer results
            try:
                sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
            except Exception as e:
                logger.error(f"Error extracting Sharpe Ratio from final backtest: {str(e)}")
                sharpe = 0.0

            try:
                returns = strat.analyzers.returns.get_analysis().get('rnorm100', 0.0)
            except Exception as e:
                logger.error(f"Error extracting Returns from final backtest: {str(e)}")
                returns = 0.0

            try:
                trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
                total_trades = trade_analyzer.total.closed if hasattr(trade_analyzer, 'total') and hasattr(trade_analyzer.total, 'closed') else 0
                win_trades = trade_analyzer.won.total if hasattr(trade_analyzer, 'won') else 0
                win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0.0
            except Exception as e:
                logger.error(f"Error extracting Trade Analyzer metrics from final backtest: {str(e)}")
                total_trades = 0
                win_rate = 0.0

            print(f'Sharpe Ratio: {sharpe}')
            print(f'Return %: {returns}')
            print(f'Win Rate: {win_rate}%')
            print(f'Number of Trades: {total_trades}')

            # Plot the result
            try:
                cerebro.plot(style='candlestick')
            except Exception as e:
                logger.error(f"Error plotting final backtest results: {str(e)}")
        else:
            logger.error("Final backtest did not run successfully.")
