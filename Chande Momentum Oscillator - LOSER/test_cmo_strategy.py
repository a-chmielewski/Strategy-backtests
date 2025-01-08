import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from ChandeMomentumOscillator import ChandeMomentumOscillatorStrategy, ChandeMomentumOscillator

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_cmo_strategy(file_path: str):
    """Test CMO Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(ChandeMomentumOscillatorStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    cmo = np.array(strategy.cmo.cmo.array)
    ma = np.array(strategy.ma.array)
    atr = np.array(strategy.atr.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    cmo = cmo[-valid_length:]
    ma = ma[-valid_length:]
    atr = atr[-valid_length:]
    
    # Create visualization with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[2, 1, 1])
    
    # Plot price and MA on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, ma, label=f'MA {strategy.params.ma_period}', alpha=0.7, color='blue')
    
    # Add price bands using ATR
    upper_band = df['Close'] + (atr * strategy.params.stop_loss_atr)
    lower_band = df['Close'] - (atr * strategy.params.stop_loss_atr)
    ax1.plot(df.index, upper_band, '--', label=f'Upper ATR Band ({strategy.params.stop_loss_atr}x)', 
             alpha=0.5, color='red')
    ax1.plot(df.index, lower_band, '--', label=f'Lower ATR Band ({strategy.params.stop_loss_atr}x)', 
             alpha=0.5, color='green')
    
    ax1.set_title('Price with Moving Average and ATR Bands')
    ax1.legend()
    ax1.grid(True)
    
    # Plot CMO on middle subplot
    ax2.plot(df.index, cmo, label='CMO', color='purple', alpha=0.7)
    ax2.axhline(y=strategy.params.overbought, color='red', linestyle='--', 
                label=f'Overbought ({strategy.params.overbought})', alpha=0.5)
    ax2.axhline(y=strategy.params.oversold, color='green', linestyle='--', 
                label=f'Oversold ({strategy.params.oversold})', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(-100, 100)
    ax2.set_title('Chande Momentum Oscillator')
    ax2.legend()
    ax2.grid(True)
    
    # Plot ATR on bottom subplot
    ax3.plot(df.index, atr, label=f'ATR ({strategy.params.atr_period})', color='orange', alpha=0.7)
    ax3.set_title('Average True Range')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"CMO Period: {strategy.params.period}")
    print(f"MA Period: {strategy.params.ma_period}")
    print(f"ATR Period: {strategy.params.atr_period}")
    print(f"Overbought Level: {strategy.params.overbought}")
    print(f"Oversold Level: {strategy.params.oversold}")
    print(f"Stop Loss ATR Multiple: {strategy.params.stop_loss_atr}")
    print(f"Take Profit ATR Multiple: {strategy.params.take_profit_atr}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in CMO: {np.isnan(cmo).any()}")
    print(f"NaN in MA: {np.isnan(ma).any()}")
    print(f"NaN in ATR: {np.isnan(atr).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"CMO Range: {np.nanmin(cmo):.2f} to {np.nanmax(cmo):.2f}")
    print(f"Average ATR: {np.nanmean(atr):.2f}")
    print(f"ATR Range: {np.nanmin(atr):.2f} to {np.nanmax(atr):.2f}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_cmo_strategy(file_path) 