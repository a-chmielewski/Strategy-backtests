import backtrader as bt
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
import math
import random
import json
from functools import partial
import traceback
from collections import deque

random.seed(42)
np.random.seed(42)


class VWAPScalping(bt.Strategy):
    params = (
        ("vwap_period", 10),
        ("rsi_period", 16),
        ("atr_period", 7),
        ("atr_multiplier", 1.5),
        ("macd_fast", 8),
        ("macd_slow", 29),
        ("macd_signal", 5),
        ("bb_period", 10),
        ("bb_std_dev", 1.8316295884897733),
        ("sar_step", 0.04436575053683621),
        ("sar_max", 0.34013889749454423),
        ("use_vwap", True),
        ("use_rsi", False),
        ("use_macd", True),
        ("use_bb", True),
        ("use_atr", False),
        ("use_sar", True),
        ("take_profit", 0.02),
        ("stop_loss", 0.01),
    )

    def __init__(self):
        if self.p.use_vwap:
            # Calculate typical price more efficiently
            self.typical_price = (
                self.data.high + self.data.low + self.data.close
            ) / 3.0

            # Use MovingAverageSimple (SMA) with caching for cumulative calculations
            cumulative_tp_vol = self.typical_price * self.data.volume

            # Use built-in indicators for moving sums with automatic caching
            self.cum_tp_vol = bt.indicators.SumN(
                cumulative_tp_vol, period=self.p.vwap_period
            )
            self.cum_vol = bt.indicators.SumN(
                self.data.volume, period=self.p.vwap_period
            )

            # Calculate VWAP using the cached values
            self.vwap = self.cum_tp_vol / self.cum_vol

            # Use CrossOver indicator for signal generation
            self.crossover = bt.indicators.CrossOver(self.data.close, self.vwap)

        if self.p.use_rsi:
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        if self.p.use_macd:
            self.macd = bt.indicators.MACD(
                self.data.close,
                period_me1=self.p.macd_fast,
                period_me2=self.p.macd_slow,
                period_signal=self.p.macd_signal,
            )

        if self.p.use_bb:
            self.bb = bt.indicators.BollingerBands(
                self.data.close, period=self.p.bb_period, devfactor=self.p.bb_std_dev
            )
            self.bb_upper = self.bb.top
            self.bb_middle = self.bb.mid
            self.bb_lower = self.bb.bot

        if self.p.use_atr:
            self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
            # Removed the line causing the error
            # Do not access self.atr[0] in __init__()

        if self.p.use_sar:
            self.sar = bt.indicators.ParabolicSAR(
                self.data, af=self.p.sar_step, afmax=self.p.sar_max
            )

    def calculate_position_size(self, current_price):
        try:
            current_equity = self.broker.getvalue()

            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0

            leverage = self.broker.getcommissioninfo(self.data).get_leverage()

            # Adjust position size according to leverage
            position_size = (position_value * leverage) / current_price

            return position_size
        except Exception as e:
            print(f"Error in calculate_position_size: {str(e)}")
            return 0


    def next(self):
        if not self.p.use_vwap or len(self.vwap) < 1:
            return

        # Use cached crossover indicator instead of manual calculation
        long_conditions = [self.crossover > 0]
        short_conditions = [self.crossover < 0]
        if self.p.use_rsi:
            long_conditions.append(self.rsi[0] < 40)
            short_conditions.append(self.rsi[0] > 60)
        if self.p.use_macd:
            long_conditions.append(self.macd.macd[0] > self.macd.signal[0])
            short_conditions.append(self.macd.macd[0] < self.macd.signal[0])
        if self.p.use_bb:
            long_conditions.append(
                self.data.close[0] > self.bb_lower[0]
                and self.data.close[0] < self.bb_upper[0]
            )
            short_conditions.append(
                self.data.close[0] < self.bb_lower[0]
                and self.data.close[0] > self.bb_upper[0]
            )
        if self.p.use_atr:
            # Ensure ATR has enough data before accessing
            if len(self.atr) > 1:
                long_conditions.append(self.atr[0] < self.atr[-1])
                short_conditions.append(self.atr[0] > self.atr[-1])
        if self.p.use_sar:
            long_conditions.append(self.data.close[0] > self.sar[0])
            short_conditions.append(self.data.close[0] < self.sar[0])

        current_price = self.data.close[0]

        # Long Entry
        if all(long_conditions) and self.position.size <= 0:
            position_size = self.calculate_position_size(current_price)
            if self.p.use_atr and len(self.atr) > 0:
                atr_value = self.atr[0]
                stop_loss = current_price - (atr_value * self.p.atr_multiplier)
                take_profit = current_price + (atr_value * self.p.atr_multiplier)
            else:
                stop_loss = current_price - (current_price * self.p.stop_loss)
                take_profit = current_price + (current_price * self.p.take_profit)

            if position_size > 0:
                self.buy_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss,
                )

        # Short Entry
        elif all(short_conditions) and self.position.size >= 0:
            position_size = self.calculate_position_size(current_price)
            if self.p.use_atr and len(self.atr) > 0:
                atr_value = self.atr[0]
                stop_loss = current_price + (atr_value * self.p.atr_multiplier)
                take_profit = current_price - (atr_value * self.p.atr_multiplier)
            else:
                stop_loss = current_price + (current_price * self.p.stop_loss)
                take_profit = current_price - (current_price * self.p.take_profit)

            if position_size > 0:
                self.sell_bracket(
                    size=position_size,
                    exectype=bt.Order.Market,
                    limitprice=take_profit,
                    stopprice=stop_loss,
                )

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

    cerebro.addstrategy(
        VWAPScalping,
        vwap_period=kwargs.get("vwap_period", 20),
        rsi_period=kwargs.get("rsi_period", 14),
        use_rsi=kwargs.get("use_rsi", False),
        use_macd=kwargs.get("use_macd", True),
        use_bb=kwargs.get("use_bb", False),
        bb_period=kwargs.get("bb_period", 20),
        bb_std_dev=kwargs.get("bb_std_dev", 2.0),
        macd_fast=kwargs.get("macd_fast", 9),
        macd_slow=kwargs.get("macd_slow", 17),
        macd_signal=kwargs.get("macd_signal", 8),
        use_atr=kwargs.get("use_atr", False),
        atr_period=kwargs.get("atr_period", 14),
        use_sar=kwargs.get("use_sar", False),
        sar_step=kwargs.get("sar_step", 0.02),
        sar_max=kwargs.get("sar_max", 0.2),
        take_profit=kwargs.get("take_profit", 0.02),
        stop_loss=kwargs.get("stop_loss", 0.01),
    )
    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        # margin=1.0 / 50,
        leverage=25,
        commtype=bt.CommInfoBase.COMM_PERC
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

    return formatted_results


def evaluate(individual, data):
    params = {
        "vwap_period": individual[0],
        "rsi_period": individual[1],
        "use_rsi": individual[2],
        "use_macd": individual[3],
        "use_bb": individual[4],
        "bb_period": individual[5],
        "bb_std_dev": individual[6],
        "macd_fast": individual[7],
        "macd_slow": individual[8],
        "macd_signal": individual[9],
        "use_atr": individual[10],
        "atr_period": individual[11],
        "use_sar": individual[12],
        "sar_step": individual[13],
        "sar_max": individual[14],
        "take_profit": 0.02,  # Fixed or consider optimizing
        "stop_loss": 0.01,    # Fixed or consider optimizing
    }
    if not individual or len(individual) < 3:  # We expect at least 3 parameters
        print(f"Invalid individual length: {len(individual) if individual else 0}")
        return (-np.inf,)
    
    # VWAP must always be used
    params["use_vwap"] = True

    # Validate parameter ranges
    if not (
        10 <= params["vwap_period"] <= 30 and
        7 <= params["rsi_period"] <= 21 and
        5 <= params["macd_fast"] <= 15 and
        20 <= params["macd_slow"] <= 40 and
        5 <= params["macd_signal"] <= 15 and
        10 <= params["bb_period"] <= 30 and
        1.5 <= params["bb_std_dev"] <= 2.5 and
        7 <= params["atr_period"] <= 21 and
        0.01 <= params["sar_step"] <= 0.05 and
        0.2 <= params["sar_max"] <= 0.5
    ):
        return (-np.inf,)
    if params["macd_fast"] >= params["macd_slow"]:
        return (-np.inf,)

    try:
        results = run_backtest(
            data, plot=False, verbose=False, optimize=False, **params
        )
        max_drawdown = results.get("Max. Drawdown [%]", 0.0)
        sharpe = results.get("Sharpe Ratio", 0.0)
        sqn = results.get("SQN", 0.0)
        return_pct = results.get("Return [%]", 0.0)

        norm_sharpe = sharpe / 3.0  # Adjust normalization based on expected range
        norm_sqn = sqn / 10.0        # Adjust normalization based on expected range
        norm_return = return_pct / 100.0
        norm_drawdown = max(1 - (max_drawdown / 100), 0)

        # Assign weights
        weight_sharpe = 0.3
        weight_sqn = 0.2
        weight_return = 0.3
        weight_drawdown = 0.2

        # Composite fitness score
        fitness = (
            (norm_sharpe * weight_sharpe) +
            (norm_sqn * weight_sqn) +
            (norm_return * weight_return) +
            (norm_drawdown * weight_drawdown)
        )

        return (fitness,)
    except Exception as e:
        print(f"Error evaluating individual: {str(e)}")
        print(f"full traceback: {traceback.format_exc()}")
        return (-np.inf,)


def adaptive_mutate(individual, generation, max_gen, mutation_rate_initial=0.3, mutation_rate_final=0.1):
    """
    Adaptive mutation that decreases mutation rate over generations and handles boolean parameter flips.
    """
    # Linearly decrease mutation rate from initial to final over generations
    mutation_rate = mutation_rate_initial - ((mutation_rate_initial - mutation_rate_final) * (generation / max_gen))
    mutation_rate = max(mutation_rate, mutation_rate_final)  # Ensure it doesn't go below final rate

    # Define which indices correspond to boolean parameters
    boolean_indices = [2, 3, 4, 10, 12]  # use_rsi, use_macd, use_bb, use_atr, use_sar

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            if i in boolean_indices:
                # Flip boolean parameter
                individual[i] = not individual[i]
            elif i in [0, 1, 5, 7, 8, 9, 11, 13, 14]:
                # Integer or float parameters: perform mutation within their ranges
                if i == 0:  # vwap_period
                    individual[i] = random.randint(10, 30)
                elif i == 1:  # rsi_period
                    individual[i] = random.randint(7, 21)
                elif i == 5:  # bb_period
                    individual[i] = random.randint(10, 30)
                elif i == 7:  # macd_fast
                    individual[i] = random.randint(5, 15)
                elif i == 8:  # macd_slow
                    individual[i] = random.randint(20, 40)
                elif i == 9:  # macd_signal
                    individual[i] = random.randint(5, 15)
                elif i == 11:  # atr_period
                    individual[i] = random.randint(7, 21)
                elif i == 13:  # sar_step
                    individual[i] = random.uniform(0.01, 0.05)
                elif i == 14:  # sar_max
                    individual[i] = random.uniform(0.2, 0.5)

    # Enforce logical constraints
    if individual[7] >= individual[8]:  # macd_fast >= macd_slow
        individual[8] = individual[7] + 1
        if individual[8] > 40:
            individual[8] = 40
            individual[7] = max(individual[8] - 1, 5)

    return (individual,)

def optimize_strategy(data, pop_size=50, generations=30, early_stop_generations=10, improvement_threshold=1e-4):
    """
    Optimize the VWAPScalping strategy parameters using the eaMuPlusLambda algorithm with:
    - Adaptive mutation rates
    - Specialized mutation for boolean parameters
    - Early stopping criteria based on fitness improvement
    """
    data_length = len(data)
    if data_length < 10000:
        pop_size = 30
        generations = 20
    elif data_length < 50000:
        pop_size = 50
        generations = 30
    else:
        pop_size = 70
        generations = 50

    setup_deap()

    toolbox = base.Toolbox()

    # Register genetic operators
    toolbox.register("vwap_period", random.randint, 10, 30)
    toolbox.register("rsi_period", random.randint, 7, 21)
    toolbox.register("use_rsi", random.choice, [True, False])
    toolbox.register("use_macd", random.choice, [True, False])
    toolbox.register("use_bb", random.choice, [True, False])
    toolbox.register("bb_period", random.randint, 10, 30)
    toolbox.register("bb_std_dev", random.uniform, 1.5, 2.5)
    toolbox.register("macd_fast", random.randint, 5, 15)
    toolbox.register("macd_slow", random.randint, 20, 40)
    toolbox.register("macd_signal", random.randint, 5, 15)
    toolbox.register("use_atr", random.choice, [True, False])
    toolbox.register("atr_period", random.randint, 7, 21)
    toolbox.register("use_sar", random.choice, [True, False])
    toolbox.register("sar_step", random.uniform, 0.01, 0.05)
    toolbox.register("sar_max", random.uniform, 0.2, 0.5)

    # Create individual with multiple parameters
    def create_individual():
        return creator.Individual([
            toolbox.vwap_period(),
            toolbox.rsi_period(),
            toolbox.use_rsi(),
            toolbox.use_macd(),
            toolbox.use_bb(),
            toolbox.bb_period(),
            toolbox.bb_std_dev(),
            toolbox.macd_fast(),
            toolbox.macd_slow(),
            toolbox.macd_signal(),
            toolbox.use_atr(),
            toolbox.atr_period(),
            toolbox.use_sar(),
            toolbox.sar_step(),
            toolbox.sar_max(),
        ])

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    evaluate_partial = partial(evaluate, data=data)
    toolbox.register("evaluate", evaluate_partial)

    # Register crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", adaptive_mutate, generation=0, max_gen=generations)
    # Register selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    # Create initial population
    pop = toolbox.population(n=pop_size)

    # Validate population
    if not pop or len(pop) == 0:
        raise ValueError("Failed to create initial population")

    # Remove invalid individuals
    pop = [ind for ind in pop if ind is not None and len(ind) > 0]
    if len(pop) == 0:
        raise ValueError("No valid individuals in initial population")

    # Statistics to track - modify to handle scalar values
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Note the [0] here
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    hof = tools.HallOfFame(1)
    # Early Stopping Setup
    best_fitness_history = deque(maxlen=early_stop_generations)
    no_improvement = 0
    best_overall_fitness = -np.inf
    
    try:
        population = pop
        logbook = tools.Logbook()
        
        # Update mutation operator for each generation
        for gen in range(generations):
            toolbox.unregister("mutate")
            toolbox.register("mutate", adaptive_mutate, generation=gen, max_gen=generations)
            
            # Run one generation
            population, gen_logbook = algorithms.eaMuPlusLambda(
                population,
                toolbox,
                mu=pop_size,
                lambda_=2*pop_size,
                cxpb=0.7,
                mutpb=0.2,
                ngen=1,
                stats=stats,
                halloffame=hof,
                verbose=True,
            )
            
            # Record statistics
            record = gen_logbook[0]
            current_best = record['max']

            # Early Stopping Check
            if current_best > best_overall_fitness + improvement_threshold:
                best_overall_fitness = current_best
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= early_stop_generations:
                print(f"No improvement in {early_stop_generations} generations. Stopping early.")
                break

            best_fitness_history.append(current_best)

    except Exception as e:
        print(f"Error during evolution: {str(e)}")
        print(f"full traceback: {traceback.format_exc()}")
    finally:
        pool.close()
        pool.join()

    # Process Hall of Fame
    try:
        hof_list = []
        for ind in hof:
            hof_list.append({
                "parameters": {
                    "vwap_period": int(ind[0]),
                    "rsi_period": int(ind[1]),
                    "use_rsi": bool(ind[2]),
                    "use_macd": bool(ind[3]),
                    "use_bb": bool(ind[4]),
                    "bb_period": int(ind[5]),
                    "bb_std_dev": float(ind[6]),
                    "macd_fast": int(ind[7]),
                    "macd_slow": int(ind[8]),
                    "macd_signal": int(ind[9]),
                    "use_atr": bool(ind[10]),
                    "atr_period": int(ind[11]),
                    "use_sar": bool(ind[12]),
                    "sar_step": float(ind[13]),
                    "sar_max": float(ind[14]),
                },
                "fitness": ind.fitness.values,
            })
        with open("hof_results.json", "w") as f:
            json.dump(hof_list, f, indent=4)
        print("\nBest parameters saved to 'hof_results.json'")
    except Exception as e:
        print(f"Error saving Hall of Fame results: {str(e)}")

    # Select the best individual
    best_individual = tools.selBest(pop, k=1)[0]
    best_params = {
        "vwap_period": int(best_individual[0]),
        "rsi_period": int(best_individual[1]),
        "use_vwap": True,  # Always True
        "use_rsi": bool(best_individual[2]),
        "use_macd": bool(best_individual[3]),
        "use_bb": bool(best_individual[4]),
        "bb_period": int(best_individual[5]),
        "bb_std_dev": float(best_individual[6]),
        "macd_fast": int(best_individual[7]),
        "macd_slow": int(best_individual[8]),
        "macd_signal": int(best_individual[9]),
        "use_atr": bool(best_individual[10]),
        "atr_period": int(best_individual[11]),
        "use_sar": bool(best_individual[12]),
        "sar_step": float(best_individual[13]),
        "sar_max": float(best_individual[14]),
        "take_profit": 0.02,  # Fixed or consider optimizing
        "stop_loss": 0.01,    # Fixed or consider optimizing
    }

    print("\nBest Parameters Found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    return best_params

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
            print(f"full traceback: {traceback.format_exc()}")
            continue

    return results

if __name__ == "__main__":
    data_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20240915-to-20241114.csv"
    setup_deap()
    # Load data with parallel processing
    print("Loading and preprocessing data...")
    data_df = pd.read_csv(data_path)
    data_df.dropna(inplace=True)
    columns = ["datetime", "Open", "High", "Low", "Close", "Volume"]
    data_df = data_df[columns]
    data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    # Ensure data is sorted by date/time if not already
    data_df = data_df.sort_values(by="datetime").reset_index(drop=True)

    # Determine the split index for 50%
    split_index = int(len(data_df) * 0.5)

    # Split the data into training and testing sets
    train_df = data_df.iloc[:split_index].reset_index(drop=True)
    test_df = data_df.iloc[split_index:].reset_index(drop=True)

    fixed_params = {
        "vwap_period": 16,
        "rsi_period": 12,
        "use_rsi": True,
        "use_macd": False,
        "use_bb": False,
        "bb_period": 16,
        "bb_std_dev": 2.2574098289906153,
        "macd_fast": 12,
        "macd_slow": 36,
        "macd_signal": 5,
        "use_atr": False,
        "atr_period": 18,
        "use_sar": True,
        "sar_step": 0.04861946623362669,
        "sar_max": 0.29479654081926365,
        "take_profit": 0.02,
        "stop_loss": 0.01,
    }


    print("Running backtest...")
    results_fixed = run_backtest(
        data_df, plot=False, verbose=True, optimize=False, **fixed_params
    )
    print("\nBacktest results with fixed parameters:")
    print(results_fixed)

    # # Run Optimization on the training data
    print("\nStarting parameter and indicator optimization on the training data...")
    best_params, logbook = optimize_strategy(train_df)

    print("\nBest parameters found from optimization:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # Run Backtest with Optimized Parameters on the testing data
    print("\nRunning backtest with optimized parameters on the testing data...")
    results_optimized = run_backtest(
        test_df, plot=True, verbose=True, optimize=False, **best_params
    )
    print("\nBacktest results with optimized parameters (Testing Data):")
    print(results_optimized)
    # Define window and step sizes for walk-forward optimization
    window_size = int(len(data_df) * 0.5)  # 50% of data for training
    step_size = int(len(data_df) * 0.1)    # 10% of data for testing

    ### Walk-Forward Optimization ####

    print("\nStarting walk-forward optimization...")
    wfo_results = walk_forward_optimization(data_df)

    # Analyze aggregated results
    if wfo_results:
        total_return = sum(res.get("Return [%]", 0.0) for res in wfo_results)
        total_trades = sum(res.get("# Trades", 0) for res in wfo_results)
        avg_sharpe = np.mean([res.get("Sharpe Ratio", 0.0) or 0.0 for res in wfo_results])
        avg_win_rate = np.mean([res.get("Win Rate [%]", 0.0) for res in wfo_results])

        print("\n=== Walk-Forward Optimization Results ===")
        print(f"Number of periods tested: {len(wfo_results)}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Average trades per period: {total_trades/len(wfo_results):.2f}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Win Rate: {avg_win_rate:.2f}%")