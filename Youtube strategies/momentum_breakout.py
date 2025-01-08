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
import json
from collections import deque

random.seed(42)

class MomentumCandlesticksStrategy(bt.Strategy):
    """
    Momentum Candlesticks Strategy that identifies supply/demand zones and trades based on long wicks
    """
    
    params = (
        ('zone_lookback', 20),          # Increased lookback for better zone detection
        ('zone_strength', 2),           # Keep lower strength requirement
        ('wick_threshold', 0.8),        # Keep strict wick requirement
        ('zone_proximity', 0.02),       # Increased proximity threshold to 2%
        ('conservative', False),        
        ('confirmation_bars', 2),      
        ('risk_reward', 2.0),         
        ('risk_percent', 2.0),        
        ('volume_factor', 1.8),        
        ('min_zone_distance', 0.01),    # Minimum 1% distance between zones
        ('max_zones', 10),             # Increased max zones
        ('swing_strength', 3),         # New: minimum bars for swing formation
    )

    def __init__(self):
        """Initialize strategy components"""
        super(MomentumCandlesticksStrategy, self).__init__()
        
        # Core data structures
        self.recent_highs = deque(maxlen=self.p.zone_lookback)
        self.recent_lows = deque(maxlen=self.p.zone_lookback)
        self.supply_zones = []
        self.demand_zones = []
        self.zone_touches = {}
        
        # Additional indicators
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.swing_strength)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.swing_strength)
        
        # Zone management
        self.last_zone_refresh = 0

    def is_long_upper_wick(self):
        """Check if current candle has a long upper wick"""
        candle = self.data0
        total_length = candle.high[0] - candle.low[0]
        if total_length == 0:
            return False
            
        upper_wick = candle.high[0] - max(candle.open[0], candle.close[0])
        wick_ratio = upper_wick / total_length
        
        return wick_ratio > self.p.wick_threshold

    def is_long_lower_wick(self):
        """Check if current candle has a long lower wick"""
        candle = self.data0
        total_length = candle.high[0] - candle.low[0]
        if total_length == 0:
            return False
            
        lower_wick = min(candle.open[0], candle.close[0]) - candle.low[0]
        wick_ratio = lower_wick / total_length
        
        return wick_ratio > self.p.wick_threshold
    
    def filter_zones(self, zones):
        """Filter zones to keep only the strongest ones within proximity
        
        Args:
            zones (list): List of tuples (price, strength)
        
        Returns:
            list: Filtered list of zones
        """
        if not zones:
            return []
        
        # Sort zones by price
        sorted_zones = sorted(zones, key=lambda x: x[0])
        filtered_zones = []
        
        for i, (price, strength) in enumerate(sorted_zones):
            # Check if this zone is too close to any stronger zone
            is_valid = True
            current_price = self.data0.close[0]
            
            for other_price, other_strength in filtered_zones:
                # Calculate price difference as percentage
                price_diff = abs(price - other_price) / current_price * 100
                
                # If zones are too close and other zone is stronger, skip this one
                if price_diff < self.p.zone_proximity and other_strength >= strength:
                    is_valid = False
                    break
            
            if is_valid:
                filtered_zones.append((price, strength))
        
        # Keep only the strongest zones up to zone_lookback
        filtered_zones.sort(key=lambda x: x[1], reverse=True)
        return filtered_zones[:self.p.zone_lookback]

    def detect_swing_high(self):
        """Detect swing high with improved logic"""
        # Check if current bar's high is the highest in the last n bars
        if self.data.high[-1] == self.highest[-1]:
            # Confirm it's higher than surrounding bars
            if (self.data.high[-1] > self.data.high[-2] and 
                self.data.high[-1] > self.data.high[0]):
                return True
        return False

    def detect_swing_low(self):
        """Detect swing low with improved logic"""
        # Check if current bar's low is the lowest in the last n bars
        if self.data.low[-1] == self.lowest[-1]:
            # Confirm it's lower than surrounding bars
            if (self.data.low[-1] < self.data.low[-2] and 
                self.data.low[-1] < self.data.low[0]):
                return True
        return False

    def update_zones(self):
        """Update supply and demand zones"""
        if len(self.data) < self.p.swing_strength + 1:
            return
            
        current_price = self.data0.close[0]
        
        # Detect swing points
        is_swing_high = self.detect_swing_high()
        is_swing_low = self.detect_swing_low()
        
        # Add new supply zone
        if is_swing_high and self.data0.volume[-1] > self.volume_ma[-1]:
            new_zone_price = self.data0.high[-1]
            # Check if price is significantly different from existing zones
            if not any(abs(new_zone_price - price) / price < self.p.min_zone_distance 
                      for price, _ in self.supply_zones):
                self.supply_zones.append((new_zone_price, 1))
        
        # Add new demand zone
        if is_swing_low and self.data0.volume[-1] > self.volume_ma[-1]:
            new_zone_price = self.data0.low[-1]
            # Check if price is significantly different from existing zones
            if not any(abs(new_zone_price - price) / price < self.p.min_zone_distance 
                      for price, _ in self.demand_zones):
                self.demand_zones.append((new_zone_price, 1))
        
        # Update zone strengths
        self._update_zone_strengths()
        
        # Clean up zones
        self._limit_zones()

    def _update_zone_strengths(self):
        """Update zone strengths"""
        current_price = self.data0.close[0]
        
        # Update supply zone strengths
        for price, _ in self.supply_zones[:]:
            # Calculate price difference as percentage
            price_diff = abs(current_price - price) / price
            
            # Update strength based on price action
            if price_diff < self.p.zone_proximity:
                # Base strength increase
                strength_increase = 0.1
                
                # Additional strength for volume
                if self.data0.volume[0] > self.volume_ma[0]:
                    strength_increase += 0.1
                
                # Additional strength for rejection
                if self.is_long_upper_wick() and self.data0.high[0] >= price:
                    strength_increase += 0.3
                
                # Update zone strength
                current_strength = self.zone_touches.get(price, 1)
                self.zone_touches[price] = current_strength + strength_increase
        
        # Update demand zone strengths
        for price, _ in self.demand_zones[:]:
            price_diff = abs(current_price - price) / price
            
            if price_diff < self.p.zone_proximity:
                strength_increase = 0.1
                
                if self.data0.volume[0] > self.volume_ma[0]:
                    strength_increase += 0.1
                
                if self.is_long_lower_wick() and self.data0.low[0] <= price:
                    strength_increase += 0.3
                
                current_strength = self.zone_touches.get(price, 1)
                self.zone_touches[price] = current_strength + strength_increase

    def _limit_zones(self):
        """Limit and sort zones"""
        current_price = self.data0.close[0]
        
        # Sort zones by strength and proximity to current price
        self.supply_zones.sort(key=lambda x: (
            -self.zone_touches.get(x[0], 1),  # Higher strength first
            abs(x[0] - current_price)         # Closer to price second
        ))
        
        self.demand_zones.sort(key=lambda x: (
            -self.zone_touches.get(x[0], 1),  # Higher strength first
            abs(x[0] - current_price)         # Closer to price second
        ))
        
        # Keep only the strongest zones
        self.supply_zones = self.supply_zones[:self.p.max_zones]
        self.demand_zones = self.demand_zones[:self.p.max_zones]

    def get_nearest_zones(self):
        """Get nearest valid supply and demand zones"""
        current_price = self.data0.close[0]
        
        # Find nearest supply zone above current price with sufficient strength
        supply_zone = None
        if self.supply_zones:
            valid_supply = [(p, s) for p, s in self.supply_zones 
                           if p > current_price and s >= self.p.zone_strength]
            if valid_supply:
                supply_zone = min(valid_supply, key=lambda x: abs(x[0] - current_price))
        
        # Find nearest demand zone below current price with sufficient strength
        demand_zone = None
        if self.demand_zones:
            valid_demand = [(p, s) for p, s in self.demand_zones 
                           if p < current_price and s >= self.p.zone_strength]
            if valid_demand:
                demand_zone = min(valid_demand, key=lambda x: abs(x[0] - current_price))
        
        return supply_zone, demand_zone

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
        """Define trading logic with improved entry conditions"""
        # Update zones
        self.update_zones()
        
        # Get nearest zones
        supply_zone, demand_zone = self.get_nearest_zones()
        
        # Skip if we don't have both zones identified
        if not supply_zone or not demand_zone:
            return
            
        # Handle existing position management
        if self.position:
            if self.position.size > 0:  # Long position
                if self.data0.high[0] >= self.take_profit:
                    self.close()
                elif self.data0.low[0] <= self.stop_loss:
                    self.close()
            else:  # Short position
                if self.data0.low[0] <= self.take_profit:
                    self.close()
                elif self.data0.high[0] >= self.stop_loss:
                    self.close()
            return

        current_price = self.data0.close[0]
        
        # Check for entries with improved proximity detection
        near_supply = abs(current_price - supply_zone[0]) / supply_zone[0] < self.p.zone_proximity
        near_demand = abs(current_price - demand_zone[0]) / demand_zone[0] < self.p.zone_proximity
        
        # Volume confirmation
        volume_confirmed = self.data0.volume[0] > self.volume_ma[0] * self.p.volume_factor
        
        # Near supply zone with long upper wick and volume confirmation
        if near_supply and self.is_long_upper_wick() and volume_confirmed:
            if not self.p.conservative:
                self.enter_short(supply_zone, demand_zone)
            else:
                self.pending_short = True
                self.confirmation_count = 0
        
        # Near demand zone with long lower wick and volume confirmation
        elif near_demand and self.is_long_lower_wick() and volume_confirmed:
            if not self.p.conservative:
                self.enter_long(demand_zone, supply_zone)
            else:
                self.pending_long = True
                self.confirmation_count = 0
        
        # Handle conservative entries
        if self.p.conservative:
            self.handle_conservative_entries(supply_zone, demand_zone)

    def enter_long(self, demand_zone, supply_zone):
        """Helper method for long entries"""
        entry_price = self.data0.close[0]
        stop_loss = demand_zone[0] * 0.999
        take_profit = supply_zone[0]
        
        if (entry_price - stop_loss) * self.p.risk_reward <= (take_profit - entry_price):
            size = self.calculate_position_size(entry_price)
            self.buy(size=size)
            self.entry_price = entry_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit

    def enter_short(self, supply_zone, demand_zone):
        """Helper method for short entries"""
        entry_price = self.data0.close[0]
        stop_loss = supply_zone[0] * 1.001
        take_profit = demand_zone[0]
        
        if (stop_loss - entry_price) * self.p.risk_reward <= (entry_price - take_profit):
            size = self.calculate_position_size(entry_price)
            self.sell(size=size)
            self.entry_price = entry_price
            self.stop_loss = stop_loss
            self.take_profit = take_profit

    def handle_conservative_entries(self, supply_zone, demand_zone):
        """Helper method for conservative entry confirmation"""
        if self.pending_long:
            if self.data0.close[0] > self.data0.open[0]:  # Bullish confirmation
                self.confirmation_count += 1
                if self.confirmation_count >= self.p.confirmation_bars:
                    self.enter_long(demand_zone, supply_zone)
                    self.pending_long = False
            else:
                self.pending_long = False
        
        if self.pending_short:
            if self.data0.close[0] < self.data0.open[0]:  # Bearish confirmation
                self.confirmation_count += 1
                if self.confirmation_count >= self.p.confirmation_bars:
                    self.enter_short(supply_zone, demand_zone)
                    self.pending_short = False
            else:
                self.pending_short = False

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

    # Add strategy
    cerebro.addstrategy(MomentumCandlesticksStrategy,
        zone_lookback=20,
        zone_strength=3,
        wick_threshold=0.7,
        conservative=True,
        confirmation_bars=2,
        risk_reward=3.0,
        risk_percent=2.0
    )

    initial_cash = 100.0
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=0.0002,
        margin=1.0 / 50,
        leverage=50,
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
            results = run_backtest(data_df, verbose=False)
            
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
        print(f"Final Equity: {result['Equity Final [$]']}")
        print(f"Sharpe Ratio: {result['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {result['Max. Drawdown [%]']:.2f}%")
        print(f"Profit Factor: {result['Profit Factor']:.2f}")