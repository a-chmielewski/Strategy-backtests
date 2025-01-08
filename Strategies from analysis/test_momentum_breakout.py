import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from momentum_breakout import MomentumBreakoutStrategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Ensure data is sorted by datetime
    df = df.sort_values(by='datetime')
    
    # Drop rows with NaN in any of OHLC or Volume columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    return df

def test_momentum_breakout_strategy(file_path: str):
    """Test Momentum Breakout Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(MomentumBreakoutStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    rsi = np.array(strategy.rsi.array)
    atr = np.array(strategy.atr.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    rsi = rsi[-valid_length:]
    atr = atr[-valid_length:]
    
    # Create visualization with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[2, 1, 1])
    
    # Plot price and ATR bands on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    
    # Add ATR-based price bands
    atr_upper = df['Close'] + (atr * strategy.params.atr_stop_multiplier)
    atr_lower = df['Close'] - (atr * strategy.params.atr_stop_multiplier)
    ax1.plot(df.index, atr_upper, '--', label=f'ATR Upper ({strategy.params.atr_stop_multiplier}x)', 
             alpha=0.5, color='red')
    ax1.plot(df.index, atr_lower, '--', label=f'ATR Lower ({strategy.params.atr_stop_multiplier}x)', 
             alpha=0.5, color='green')
    
    # Add take profit levels
    tp_upper = df['Close'] + (atr * strategy.params.atr_target_multiplier)
    tp_lower = df['Close'] - (atr * strategy.params.atr_target_multiplier)
    ax1.plot(df.index, tp_upper, ':', label=f'Take Profit Upper ({strategy.params.atr_target_multiplier}x)', 
             alpha=0.3, color='blue')
    ax1.plot(df.index, tp_lower, ':', label=f'Take Profit Lower ({strategy.params.atr_target_multiplier}x)', 
             alpha=0.3, color='blue')
    
    ax1.set_title('Price with ATR Bands')
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
    
    # Plot ATR on bottom subplot
    ax3.plot(df.index, atr, label=f'ATR ({strategy.params.atr_period})', color='orange', alpha=0.7)
    ax3.set_title('Average True Range (ATR)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"RSI Period: {strategy.params.rsi_period}")
    print(f"RSI Overbought: {strategy.params.rsi_overbought}")
    print(f"RSI Oversold: {strategy.params.rsi_oversold}")
    print(f"ATR Period: {strategy.params.atr_period}")
    print(f"ATR Stop Multiplier: {strategy.params.atr_stop_multiplier}")
    print(f"ATR Target Multiplier: {strategy.params.atr_target_multiplier}")
    print(f"ATR Trail Multiplier: {strategy.params.atr_trail_multiplier}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in RSI: {np.isnan(rsi).any()}")
    print(f"NaN in ATR: {np.isnan(atr).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"RSI Range: {np.nanmin(rsi):.2f} to {np.nanmax(rsi):.2f}")
    print(f"Average RSI: {np.nanmean(rsi):.2f}")
    print(f"ATR Range: {np.nanmin(atr):.2f} to {np.nanmax(atr):.2f}")
    print(f"Average ATR: {np.nanmean(atr):.2f}")
    print(f"ATR Volatility (std): {np.nanstd(atr):.2f}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_momentum_breakout_strategy(file_path) 