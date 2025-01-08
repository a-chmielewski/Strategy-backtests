import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from volatility_adaptive_scalping import VolatilityAdaptiveScalping

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

def test_volatility_adaptive_scalping(file_path: str):
    """Test Volatility Adaptive Scalping Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(VolatilityAdaptiveScalping)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    atr = np.array(strategy.atr.array)
    rsi = np.array(strategy.rsi.array)
    sma = np.array(strategy.sma.array)
    highest_high = np.array(strategy.highest_high.array)
    lowest_low = np.array(strategy.lowest_low.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    atr = atr[-valid_length:]
    rsi = rsi[-valid_length:]
    sma = sma[-valid_length:]
    highest_high = highest_high[-valid_length:]
    lowest_low = lowest_low[-valid_length:]
    
    # Create time filter mask
    time_mask = np.zeros(len(df))
    for i, timestamp in enumerate(df['datetime']):
        hour = timestamp.hour
        weekday = timestamp.weekday()
        if hour in [14, 15, 16] and weekday in [0, 1, 2]:
            time_mask[i] = 1
    
    # Create visualization with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 20), height_ratios=[2, 1, 1, 1])
    
    # Plot price, SMA, and breakout levels on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, sma, label=f'SMA ({strategy.params.sma_period})', alpha=0.7, color='blue')
    ax1.plot(df.index, highest_high, label=f'Highest High ({strategy.params.breakout_period})', 
             alpha=0.5, color='green', linestyle='--')
    ax1.plot(df.index, lowest_low, label=f'Lowest Low ({strategy.params.breakout_period})', 
             alpha=0.5, color='red', linestyle='--')
    
    # Add ATR-based bands
    atr_upper = df['Close'] + (atr * strategy.params.stop_loss_atr)
    atr_lower = df['Close'] - (atr * strategy.params.stop_loss_atr)
    ax1.plot(df.index, atr_upper, ':', label=f'ATR Upper ({strategy.params.stop_loss_atr}x)', 
             alpha=0.3, color='red')
    ax1.plot(df.index, atr_lower, ':', label=f'ATR Lower ({strategy.params.stop_loss_atr}x)', 
             alpha=0.3, color='green')
    
    ax1.set_title('Price with SMA and Breakout Levels')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI on second subplot
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
    
    # Plot ATR on third subplot
    ax3.plot(df.index, atr, label=f'ATR ({strategy.params.atr_period})', color='orange', alpha=0.7)
    ax3.axhline(y=strategy.params.min_atr, color='green', linestyle='--', 
                label=f'Min ATR ({strategy.params.min_atr})', alpha=0.5)
    ax3.axhline(y=strategy.params.max_atr, color='red', linestyle='--', 
                label=f'Max ATR ({strategy.params.max_atr})', alpha=0.5)
    ax3.set_title('Average True Range (ATR)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot Time Filter on bottom subplot
    ax4.fill_between(df.index, 0, time_mask, label='Trading Window', 
                    color='green', alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_title('Time Filter (Green = Trading Allowed)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"ATR Period: {strategy.params.atr_period}")
    print(f"RSI Period: {strategy.params.rsi_period}")
    print(f"SMA Period: {strategy.params.sma_period}")
    print(f"Breakout Period: {strategy.params.breakout_period}")
    print(f"ATR Range: {strategy.params.min_atr} to {strategy.params.max_atr}")
    print(f"RSI Levels: {strategy.params.rsi_oversold}/{strategy.params.rsi_overbought}")
    print(f"Stop Loss ATR Multiple: {strategy.params.stop_loss_atr}")
    print(f"Take Profit ATR Multiple: {strategy.params.take_profit_atr}")
    print(f"Trailing ATR Multiple: {strategy.params.trailing_atr}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in ATR: {np.isnan(atr).any()}")
    print(f"NaN in RSI: {np.isnan(rsi).any()}")
    print(f"NaN in SMA: {np.isnan(sma).any()}")
    print(f"NaN in Highest High: {np.isnan(highest_high).any()}")
    print(f"NaN in Lowest Low: {np.isnan(lowest_low).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"ATR Range: {np.nanmin(atr):.2f} to {np.nanmax(atr):.2f}")
    print(f"Average ATR: {np.nanmean(atr):.2f}")
    print(f"RSI Range: {np.nanmin(rsi):.2f} to {np.nanmax(rsi):.2f}")
    print(f"Average RSI: {np.nanmean(rsi):.2f}")
    print(f"Trading Window Coverage: {np.mean(time_mask)*100:.1f}% of total time")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_volatility_adaptive_scalping(file_path) 