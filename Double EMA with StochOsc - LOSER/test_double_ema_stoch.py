import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from double_EMA_StochOsc import DoubleEMA_StochOsc_Strategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_double_ema_stoch_strategy(file_path: str):
    """Test Double EMA with Stochastic Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(DoubleEMA_StochOsc_Strategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    ema_slow = np.array(strategy.ema_slow.array)
    ema_fast = np.array(strategy.ema_fast.array)
    stoch_k = np.array(strategy.stoch.percK.array)
    stoch_d = np.array(strategy.stoch.percD.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    ema_slow = ema_slow[-valid_length:]
    ema_fast = ema_fast[-valid_length:]
    stoch_k = stoch_k[-valid_length:]
    stoch_d = stoch_d[-valid_length:]
    
    # Create visualization with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[2, 1])
    
    # Plot price and EMAs on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, ema_slow, label=f'Slow EMA ({strategy.params.ema_slow})', 
             alpha=0.7, color='blue')
    ax1.plot(df.index, ema_fast, label=f'Fast EMA ({strategy.params.ema_fast})', 
             alpha=0.7, color='red')
    
    # Add swing points visualization
    swing_high = df['High'].rolling(window=3, center=True).max()
    swing_low = df['Low'].rolling(window=3, center=True).min()
    ax1.plot(df.index, swing_high, '--', label='Swing High', alpha=0.3, color='red')
    ax1.plot(df.index, swing_low, '--', label='Swing Low', alpha=0.3, color='green')
    
    ax1.set_title('Price with Double EMAs and Swing Points')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Stochastic on bottom subplot
    ax2.plot(df.index, stoch_k, label='%K', color='blue', alpha=0.7)
    ax2.plot(df.index, stoch_d, label='%D', color='red', alpha=0.7)
    ax2.axhline(y=strategy.params.stoch_overbought, color='red', linestyle='--', 
                label=f'Overbought ({strategy.params.stoch_overbought})', alpha=0.5)
    ax2.axhline(y=strategy.params.stoch_oversold, color='green', linestyle='--', 
                label=f'Oversold ({strategy.params.stoch_oversold})', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('Stochastic Oscillator')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Slow EMA Period: {strategy.params.ema_slow}")
    print(f"Fast EMA Period: {strategy.params.ema_fast}")
    print(f"Stochastic K Period: {strategy.params.stoch_k}")
    print(f"Stochastic D Period: {strategy.params.stoch_d}")
    print(f"Stochastic Slowing: {strategy.params.slowing}")
    print(f"Stochastic Overbought: {strategy.params.stoch_overbought}")
    print(f"Stochastic Oversold: {strategy.params.stoch_oversold}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in Slow EMA: {np.isnan(ema_slow).any()}")
    print(f"NaN in Fast EMA: {np.isnan(ema_fast).any()}")
    print(f"NaN in Stochastic %K: {np.isnan(stoch_k).any()}")
    print(f"NaN in Stochastic %D: {np.isnan(stoch_d).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"Stochastic %K Range: {np.nanmin(stoch_k):.2f} to {np.nanmax(stoch_k):.2f}")
    print(f"Stochastic %D Range: {np.nanmin(stoch_d):.2f} to {np.nanmax(stoch_d):.2f}")
    print(f"Average %K: {np.nanmean(stoch_k):.2f}")
    print(f"Average %D: {np.nanmean(stoch_d):.2f}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_double_ema_stoch_strategy(file_path) 