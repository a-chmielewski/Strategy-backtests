import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from time_filtered_trend import TimeFilteredTrendStrategy

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

def test_time_filtered_trend_strategy(file_path: str):
    """Test Time Filtered Trend Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(TimeFilteredTrendStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    ema_fast = np.array(strategy.ema_fast.array)
    ema_slow = np.array(strategy.ema_slow.array)
    rsi = np.array(strategy.rsi.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    ema_fast = ema_fast[-valid_length:]
    ema_slow = ema_slow[-valid_length:]
    rsi = rsi[-valid_length:]
    
    # Create time filter mask
    time_mask = np.zeros(len(df))
    for i, timestamp in enumerate(df['datetime']):
        hour = timestamp.hour
        weekday = timestamp.weekday()
        if weekday in strategy.trading_days and hour in strategy.trading_hours:
            time_mask[i] = 1
    
    # Create visualization with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[2, 1, 1])
    
    # Plot price and EMAs on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, ema_fast, label=f'Fast EMA ({strategy.params.ema_fast_period})', 
             alpha=0.7, color='blue')
    ax1.plot(df.index, ema_slow, label=f'Slow EMA ({strategy.params.ema_slow_period})', 
             alpha=0.7, color='red')
    
    ax1.set_title('Price with EMAs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI on middle subplot
    ax2.plot(df.index, rsi, label=f'RSI ({strategy.params.rsi_period})', color='purple', alpha=0.7)
    ax2.axhline(y=strategy.params.rsi_overbought, color='red', linestyle='--', 
                label=f'Overbought ({strategy.params.rsi_overbought})', alpha=0.5)
    ax2.axhline(y=strategy.params.rsi_oversold, color='green', linestyle='--', 
                label=f'Oversold ({strategy.params.rsi_oversold})', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Time Filter on bottom subplot
    ax3.fill_between(df.index, 0, time_mask, label='Trading Window', 
                    color='green', alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_title('Time Filter (Green = Trading Allowed)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Fast EMA Period: {strategy.params.ema_fast_period}")
    print(f"Slow EMA Period: {strategy.params.ema_slow_period}")
    print(f"RSI Period: {strategy.params.rsi_period}")
    print(f"RSI Overbought: {strategy.params.rsi_overbought}")
    print(f"RSI Oversold: {strategy.params.rsi_oversold}")
    print("\nTime Filter Settings:")
    print(f"Trading Hours (UTC): {strategy.trading_hours}")
    print(f"Trading Days: {strategy.trading_days} (0=Monday, 6=Sunday)")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in Fast EMA: {np.isnan(ema_fast).any()}")
    print(f"NaN in Slow EMA: {np.isnan(ema_slow).any()}")
    print(f"NaN in RSI: {np.isnan(rsi).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"RSI Range: {np.nanmin(rsi):.2f} to {np.nanmax(rsi):.2f}")
    print(f"Average RSI: {np.nanmean(rsi):.2f}")
    print(f"EMA Fast Range: {np.nanmin(ema_fast):.2f} to {np.nanmax(ema_fast):.2f}")
    print(f"EMA Slow Range: {np.nanmin(ema_slow):.2f} to {np.nanmax(ema_slow):.2f}")
    print(f"Trading Window Coverage: {np.mean(time_mask)*100:.1f}% of total time")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_time_filtered_trend_strategy(file_path) 