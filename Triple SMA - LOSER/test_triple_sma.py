import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from tripleSMA import TrippleSMA_Strategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Ensure data is sorted by datetime
    df = df.sort_values(by='datetime')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    return df

def test_triple_sma_strategy(file_path: str):
    """Test Triple SMA Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(TrippleSMA_Strategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    fast_sma = np.array(strategy.fast_sma.array)
    medium_sma = np.array(strategy.medium_sma.array)
    slow_sma = np.array(strategy.slow_sma.array)
    rsi = np.array(strategy.rsi.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    fast_sma = fast_sma[-valid_length:]
    medium_sma = medium_sma[-valid_length:]
    slow_sma = slow_sma[-valid_length:]
    rsi = rsi[-valid_length:]
    
    # Create visualization with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15), height_ratios=[2, 1])
    
    # Plot price and SMAs on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, fast_sma, label=f'Fast SMA ({strategy.params.fast_sma})', 
             alpha=0.7, color='blue')
    ax1.plot(df.index, medium_sma, label=f'Medium SMA ({strategy.params.medium_sma})', 
             alpha=0.7, color='orange')
    ax1.plot(df.index, slow_sma, label=f'Slow SMA ({strategy.params.slow_sma})', 
             alpha=0.7, color='red')
    
    # Add take profit and stop loss levels
    tp_level_long = df['Close'] * (1 + strategy.params.take_profit)
    sl_level_long = df['Close'] * (1 - strategy.params.stop_loss)
    ax1.plot(df.index, tp_level_long, ':', label=f'Take Profit Level (+{strategy.params.take_profit*100}%)', 
             alpha=0.3, color='green')
    ax1.plot(df.index, sl_level_long, ':', label=f'Stop Loss Level (-{strategy.params.stop_loss*100}%)', 
             alpha=0.3, color='red')
    
    ax1.set_title('Price with Triple SMAs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI on bottom subplot
    ax2.plot(df.index, rsi, label=f'RSI ({strategy.params.rsi})', color='purple', alpha=0.7)
    ax2.axhline(y=70, color='red', linestyle='--', label='Overbought (70)', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='--', label='Oversold (30)', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Fast SMA Period: {strategy.params.fast_sma}")
    print(f"Medium SMA Period: {strategy.params.medium_sma}")
    print(f"Slow SMA Period: {strategy.params.slow_sma}")
    print(f"RSI Period: {strategy.params.rsi}")
    print(f"Stop Loss: {strategy.params.stop_loss*100}%")
    print(f"Take Profit: {strategy.params.take_profit*100}%")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in Fast SMA: {np.isnan(fast_sma).any()}")
    print(f"NaN in Medium SMA: {np.isnan(medium_sma).any()}")
    print(f"NaN in Slow SMA: {np.isnan(slow_sma).any()}")
    print(f"NaN in RSI: {np.isnan(rsi).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"RSI Range: {np.nanmin(rsi):.2f} to {np.nanmax(rsi):.2f}")
    print(f"Average RSI: {np.nanmean(rsi):.2f}")
    print("\nSMA Relationships:")
    print(f"Fast vs Medium Crossovers: {np.sum(np.diff(fast_sma > medium_sma) != 0)}")
    print(f"Fast vs Slow Crossovers: {np.sum(np.diff(fast_sma > slow_sma) != 0)}")
    print(f"Average SMA Spread: {np.nanmean(fast_sma - slow_sma):.2f}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_triple_sma_strategy(file_path) 