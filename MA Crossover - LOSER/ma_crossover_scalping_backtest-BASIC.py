import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
import pandas_ta as ta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class MaCrossoverStrategy(Strategy):

    # Define parameters that can be optimized
    fast_ma_period = 5  # Fast MA period
    slow_ma_period = 20  # Slow MA period
    profit_target = 0.02   # 2% profit target
    stop_loss = 0.01      # 1% stop loss
    max_spread = 10        # Maximum allowed spread in points
    risk_per_trade = 0.01  # 1% of equity per trade
    
    def init(self):
        # Calculate fast and slow moving averages using mid price for signals
        self.fast_ma = self.I(talib.SMA, self.data.mid_price, timeperiod=self.fast_ma_period)
        self.slow_ma = self.I(talib.SMA, self.data.mid_price, timeperiod=self.slow_ma_period)
        
        # Store crossover signals
        self.crossed_above = self.I(lambda: self.fast_ma > self.slow_ma)
        self.crossed_below = self.I(lambda: self.fast_ma < self.slow_ma)
        
        # Additional indicators for analysis
        self.spread = self.data.spread
        self.volatility = self.data.volatility
    
    def next(self):
        # Skip if spread is too wide
        if self.spread[-1] > self.max_spread:
            return
        
        # Calculate position size based on risk
        risk_amount = self.equity * self.risk_per_trade
        
        # Long position setup
        if self.crossed_above[-1] and not self.position:
            # Calculate long position parameters
            entry_price = self.data.ask[-1]
            btc_position = risk_amount / entry_price
            size = max(1.0, btc_position)
            
            # For long positions:
            # Set limit price slightly above current ask
            limit_price = entry_price * 1.0002  # 0.02% above ask
            
            # Set SL/TP with more buffer from entry
            stop_loss_price = entry_price * (1 - self.stop_loss - 0.001)  # Additional 0.1% buffer
            take_profit_price = entry_price * (1 + self.profit_target + 0.001)  # Additional 0.1% buffer
            
            # Verify order levels
            if stop_loss_price < entry_price < take_profit_price:
                self.buy(size=size,
                        limit=limit_price,
                        sl=stop_loss_price,
                        tp=take_profit_price)
        
        # Short position setup
        elif self.crossed_below[-1] and not self.position:
            # Calculate short position parameters
            entry_price = self.data.bid[-1]
            btc_position = risk_amount / entry_price
            size = max(1.0, btc_position)
            
            # For short positions:
            # Set limit price slightly below current bid
            limit_price = entry_price * 0.9998  # 0.02% below bid
            
            # Set SL/TP with more buffer from entry
            stop_loss_price = entry_price * (1 + self.stop_loss + 0.001)  # Additional 0.1% buffer
            take_profit_price = entry_price * (1 - self.profit_target - 0.001)  # Additional 0.1% buffer
            
            # Verify order levels
            if take_profit_price < entry_price < stop_loss_price:
                self.sell(size=size,
                         limit=limit_price,
                         sl=stop_loss_price,
                         tp=take_profit_price)

def run_backtest(data, **kwargs):
    """
    Run the backtest with the given data and parameters.

    Parameters:
    - data: pandas DataFrame with required columns.
    - kwargs: strategy parameters.

    Returns:
    - results: Backtest results object.
    """

    # Initialize Backtest
    bt = Backtest(
        data,
        MaCrossoverStrategy,
        cash=1000000.0,
        commission=0.0001,  # Example commission
        exclusive_orders=False,  # Allow multiple concurrent orders
        trade_on_close=False,    # Important for proper bid/ask price execution
        margin=1.0 / 50
    )

    # Set default parameters if not provided
    if not kwargs:
        kwargs = {
            'fast_ma_period': 5,
            'slow_ma_period': 20,
            'profit_target': 0.02,
            'stop_loss': 0.01,
            'max_spread': 10,
            'risk_per_trade': 0.01  # 1% risk per trade
        }
    
    # Run Backtest with provided parameters
    results = bt.run(**kwargs)

    # Plot the results
    bt.plot(resample=False)

    return results

if __name__ == "__main__":
    # Load and preprocess your dataset
    data = pd.read_csv(
        r'F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-1m-20241005-to-20241104.csv'
    )

    # Ensure all required columns are present
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'bid', 'ask', 'mid_price', 'spread', 'volatility']
    
    assert all(col in data.columns for col in required_columns), \
        f"Missing required columns. Required: {required_columns}"

    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    results = run_backtest(data)
    print(results)
