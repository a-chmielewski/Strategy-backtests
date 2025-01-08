import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from momentum_breakout import MomentumCandlesticksStrategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

class TestMomentumStrategy(MomentumCandlesticksStrategy):
    """Extended strategy class for testing and visualization"""
    
    def __init__(self):
        super(TestMomentumStrategy, self).__init__()
        # Store historical data for plotting
        self.supply_zone_history = []
        self.demand_zone_history = []
        self.upper_wicks = []
        self.lower_wicks = []
        self.volume_signals = []
        
    def next(self):
        """Override next to store historical data"""
        super(TestMomentumStrategy, self).next()
        
        # Store current zones
        self.supply_zone_history.append(
            [(price, strength) for price, strength in self.supply_zones]
        )
        self.demand_zone_history.append(
            [(price, strength) for price, strength in self.demand_zones]
        )
        
        # Store wick information
        self.upper_wicks.append(1 if self.is_long_upper_wick() else 0)
        self.lower_wicks.append(1 if self.is_long_lower_wick() else 0)
        
        # Store volume signals
        volume_confirmed = (self.data0.volume[0] > self.volume_ma[0] * self.p.volume_factor)
        self.volume_signals.append(1 if volume_confirmed else 0)

def test_momentum_strategy(file_path: str):
    """Test Momentum Breakout strategy visualization"""
    # Load market data
    print(f"Loading market data from {file_path}...")
    df = load_market_data(file_path)
    print(f"Loaded {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    
    # Create Backtrader cerebro instance
    cerebro = bt.Cerebro()
    
    # Add data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=-1
    )
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(TestMomentumStrategy)
    
    # Run strategy
    print("Running strategy analysis...")
    results = cerebro.run()
    strategy = results[0]
    
    # Create visualization
    # plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Price with Supply/Demand Zones
    ax1 = plt.subplot(4, 1, 1)
    plt.plot(df.index, df['Close'], label='Price', color='blue', alpha=0.7)
    
    # Plot supply and demand zones
    last_supply_zones = strategy.supply_zone_history[-1]
    last_demand_zones = strategy.demand_zone_history[-1]
    
    for price, strength in last_supply_zones:
        # Normalize strength to be between 0 and 1
        normalized_alpha = min(0.3 * (strength / 100), 1.0)  # Assuming strength is percentage-based
        plt.axhline(y=price, color='red', linestyle='--', alpha=normalized_alpha)
    
    for price, strength in last_demand_zones:
        # Normalize strength to be between 0 and 1
        normalized_alpha = min(0.3 * (strength / 100), 1.0)  # Assuming strength is percentage-based
        plt.axhline(y=price, color='green', linestyle='--', alpha=normalized_alpha)
   
    plt.title('Price with Supply/Demand Zones')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Candlestick Wicks
    ax2 = plt.subplot(4, 1, 2)
    # Create an array of the same length as the data, filled with zeros
    upper_wicks = np.zeros(len(df))
    lower_wicks = np.zeros(len(df))
    # Fill in the signals starting from where they begin
    upper_wicks[-len(strategy.upper_wicks):] = strategy.upper_wicks
    lower_wicks[-len(strategy.lower_wicks):] = strategy.lower_wicks
    
    plt.plot(df.index, upper_wicks, label='Upper Wicks', color='red')
    plt.plot(df.index, lower_wicks, label='Lower Wicks', color='green')
    plt.title('Wick Signals (1 = Long Wick Detected)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Volume
    ax3 = plt.subplot(4, 1, 3)
    plt.plot(df.index, df['Volume'], label='Volume', color='blue', alpha=0.5)
    # Pad volume MA with zeros at the beginning
    volume_ma = np.zeros(len(df))
    volume_ma[-len(strategy.volume_ma):] = strategy.volume_ma
    plt.plot(df.index, volume_ma, label='Volume MA', color='orange')
    plt.title('Volume Analysis')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Volume Confirmation Signals
    ax4 = plt.subplot(4, 1, 4)
    # Pad volume signals with zeros at the beginning
    volume_signals = np.zeros(len(df))
    volume_signals[-len(strategy.volume_signals):] = strategy.volume_signals
    plt.plot(df.index, volume_signals, label='Volume Signals', color='purple')
    plt.title('Volume Confirmation Signals (1 = Volume Confirmed)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Print strategy parameters
    print("\nStrategy Parameters:")
    print(f"Zone Lookback: {strategy.p.zone_lookback}")
    print(f"Zone Strength Required: {strategy.p.zone_strength}")
    print(f"Wick Threshold: {strategy.p.wick_threshold}")
    print(f"Zone Proximity: {strategy.p.zone_proximity}")
    print(f"Volume Factor: {strategy.p.volume_factor}")
    
    # Print zone statistics
    print("\nZone Statistics:")
    print(f"Number of Supply Zones: {len(last_supply_zones)}")
    print(f"Number of Demand Zones: {len(last_demand_zones)}")
    
    if last_supply_zones:
        print("\nSupply Zones (Price, Strength):")
        for price, strength in last_supply_zones:
            print(f"  {price:.2f}, {strength}")
    
    if last_demand_zones:
        print("\nDemand Zones (Price, Strength):")
        for price, strength in last_demand_zones:
            print(f"  {price:.2f}, {strength}")
    
    # Calculate signal statistics
    upper_wick_signals = sum(strategy.upper_wicks)
    lower_wick_signals = sum(strategy.lower_wicks)
    volume_confirmations = sum(strategy.volume_signals)
    
    print("\nSignal Statistics:")
    print(f"Upper Wick Signals: {upper_wick_signals}")
    print(f"Lower Wick Signals: {lower_wick_signals}")
    print(f"Volume Confirmations: {volume_confirmations}")
    
    plt.show()

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_momentum_strategy(file_path) 