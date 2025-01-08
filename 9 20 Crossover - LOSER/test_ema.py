import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from EMA import EMAStrategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_ema_strategy(file_path: str):
    """Test EMA Crossover strategy indicators calculation and visualization"""
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
    
    # Add strategy with default parameters
    cerebro.addstrategy(EMAStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    ema_short = np.array(strategy.ema_short.array)
    ema_long = np.array(strategy.ema_long.array)
    crossover = np.array(strategy.crossover.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    ema_short = ema_short[-valid_length:]
    ema_long = ema_long[-valid_length:]
    crossover = crossover[-valid_length:]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[3, 1])
    
    # Plot price and EMAs
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7)
    ax1.plot(df.index, ema_short, label=f'EMA {strategy.params.ema_short}', alpha=0.7)
    ax1.plot(df.index, ema_long, label=f'EMA {strategy.params.ema_long}', alpha=0.7)
    
    # Highlight crossover points
    buy_signals = [i for i in range(len(crossover)) if crossover[i] > 0]
    sell_signals = [i for i in range(len(crossover)) if crossover[i] < 0]
    
    ax1.scatter(buy_signals, df['Close'].iloc[buy_signals], color='green', marker='^', 
                label='Buy Signal', alpha=0.7)
    ax1.scatter(sell_signals, df['Close'].iloc[sell_signals], color='red', marker='v',
                label='Sell Signal', alpha=0.7)
    
    ax1.set_title('Price and EMA Crossover Strategy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot crossover indicator
    ax2.plot(df.index, crossover, label='Crossover', color='blue', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_title('Crossover Indicator')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Short EMA Period: {strategy.params.ema_short}")
    print(f"Long EMA Period: {strategy.params.ema_long}")
    print(f"Stop Loss: {strategy.params.stop_loss}")
    print(f"Take Profit: {strategy.params.take_profit}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in Short EMA: {np.isnan(ema_short).any()}")
    print(f"NaN in Long EMA: {np.isnan(ema_long).any()}")
    print(f"NaN in Crossover: {np.isnan(crossover).any()}")
    
    # Print basic statistics
    print("\nCrossover Statistics:")
    print(f"Total Buy Signals: {len(buy_signals)}")
    print(f"Total Sell Signals: {len(sell_signals)}")
    print(f"Average Signals per Day: {(len(buy_signals) + len(sell_signals)) / (len(df) / 288):.2f}")  # Assuming 5m timeframe

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_ema_strategy(file_path) 