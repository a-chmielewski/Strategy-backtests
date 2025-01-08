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
    
    # Optimialization results:
    # Best parameters found:
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
    

    # Define parameters that can be optimized
    # fast_ma_period = 5      # Fast EMA period
    # slow_ma_period = 20     # Slow EMA period
    # rsi_period = 14        # RSI period
    # atr_period = 14        # ATR period
    # vol_ma_period = 20     # Volume MA period
    # cci_period = 14        # CCI period
    # profit_target = 0.02   # 2% profit target
    # atr_multiplier = 1.5   # Multiplier for ATR-based stop loss
    # min_atr_threshold = 50  # Minimum ATR value to trade (in price points)
    # max_spread = 10       # Maximum allowed spread in points
    # risk_per_trade = 0.01 # 1% of equity per trade
    
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
        
        # Store crossover signals
        self.crossed_above = self.I(lambda: self.fast_ma > self.slow_ma)
        self.crossed_below = self.I(lambda: self.fast_ma < self.slow_ma)
        
        # Store RSI momentum signals
        self.rsi_rising = self.I(lambda: self.rsi > self.rsi[-1])
        self.rsi_falling = self.I(lambda: self.rsi < self.rsi[-1])
        
        # Store CCI crossover signals
        self.cci_bull = self.I(lambda: self.cci > self.cci_high)
        self.cci_bear = self.I(lambda: self.cci < self.cci_low)
        
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
        if self.position.pl_pct >= self.profit_target:
            return True
        
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

        rsi_rising = self.rsi[-1] > self.rsi[-2]
        rsi_falling = self.rsi[-1] < self.rsi[-2]
        cci_bull = self.cci[-1] > self.cci_high
        cci_bear = self.cci[-1] < self.cci_low
        
        # Long position setup
        if (crossover(self.fast_ma, self.slow_ma) and not self.position and
            rsi_rising and  # RSI is rising
            self.rsi_middle < self.rsi[-1] < self.rsi_overbought and  # RSI between 50 and 70
            cci_bull):  # CCI above +100
            
            # Calculate long position parameters
            entry_price = self.data.ask[-1]  # Use Close price for entry
            
            # Calculate ATR-based stop loss
            atr_stop = self.atr[-1] * self.atr_multiplier
            stop_loss_price = entry_price - atr_stop
            take_profit_price = entry_price + (atr_stop * 2)  # 2:1 reward-to-risk ratio
            
            # Ensure SL and TP are logical
            if stop_loss_price >= entry_price:
                return  # Invalid SL
            
            # Calculate position size based on ATR stop loss
            risk_per_coin = entry_price - stop_loss_price
            btc_position = risk_amount / risk_per_coin
            
            # Round to nearest whole number and ensure minimum of 1 BTC
            size = max(1, int(btc_position))
            
            # Place buy order
            self.buy(size=size)
            
            # Set entry time
            self.entry_time = self.data.index[-1]
            
            # Set SL and TP
            self.sl = stop_loss_price
            self.tp = take_profit_price
        
        # Short position setup
        elif (crossover(self.slow_ma, self.fast_ma) and not self.position and
              rsi_falling and  # RSI is falling
              self.rsi_oversold < self.rsi[-1] < self.rsi_middle and  # RSI between 30 and 50
              cci_bear):  # CCI below -100
            
            # Calculate short position parameters
            entry_price = self.data.bid[-1]  # Use Close price for entry
            
            # Calculate ATR-based stop loss
            atr_stop = self.atr[-1] * self.atr_multiplier
            stop_loss_price = entry_price + atr_stop
            take_profit_price = entry_price - (atr_stop * 2)  # 2:1 reward-to-risk ratio
            
            # Ensure SL and TP are logical
            if stop_loss_price <= entry_price:
                return  # Invalid SL
            
            # Calculate position size based on ATR stop loss
            risk_per_coin = stop_loss_price - entry_price
            btc_position = risk_amount / risk_per_coin
            
            # Round to nearest whole number and ensure minimum of 1 BTC
            size = max(1, int(btc_position))
            
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
            'slow_ma_period': 15,
            'rsi_period': 13,
            'atr_period': 15,
            'vol_ma_period': 22,
            'cci_period': 11,
            'atr_multiplier': 2.0,
            'min_atr_threshold': 40,
            'profit_target': 0.0002,
            'max_spread': 9,
            'risk_per_trade': 0.02,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_middle': 50,
            'cci_high': 100,
            'cci_low': -100,
            'max_trade_duration': 9
        }

    
    # Run Backtest with provided parameters
    results = bt.run(**kwargs)

    # Create a simpler filename for the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"MA_Crossover_Backtest_{timestamp}.html"
    
    # Plot the results with the simplified filename
    bt.plot(filename=plot_filename, resample=False)

    return results

if __name__ == "__main__":
    # Load data as before
    data = pd.read_csv(
        r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-ETHUSDT-1m-20241007-to-20241106.csv'
    )
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'bid', 'ask', 'mid_price', 'spread', 'volatility']
    
    assert all(col in data.columns for col in required_columns), \
        f"Missing required columns. Required: {required_columns}"

    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    results = run_backtest(data)
    print(results)

# bybit fees: taker 0.0550%, maker 0.02%