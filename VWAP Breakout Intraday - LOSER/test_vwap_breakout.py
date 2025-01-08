import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from VWAP_Breakout import VWAP_BreakoutStrategy, VWAP

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

def test_vwap_breakout_strategy(file_path: str):
    """Test VWAP Breakout Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(VWAP_BreakoutStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    vwap = np.array(strategy.vwap.vwap.array)
    volume_sma = np.array(strategy.volume_sma.array)
    
    # Calculate relative volume
    volume = np.array(df['Volume'])
    relative_volume = np.where(volume_sma != 0, volume / volume_sma, 0)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    vwap = vwap[-valid_length:]
    volume_sma = volume_sma[-valid_length:]
    
    # Create visualization with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 20), height_ratios=[2, 1, 1, 1])
    
    # Plot price on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    
    # Add stop loss and take profit bands
    sl_band_up = df['Close'] * (1 + strategy.params.stop_loss_pct)
    sl_band_down = df['Close'] * (1 - strategy.params.stop_loss_pct)
    tp_band_up = df['Close'] + (df['Close'] * strategy.params.stop_loss_pct * strategy.params.take_profit_ratio)
    tp_band_down = df['Close'] - (df['Close'] * strategy.params.stop_loss_pct * strategy.params.take_profit_ratio)
    
    ax1.plot(df.index, sl_band_up, ':', label=f'Stop Loss Band (+{strategy.params.stop_loss_pct*100}%)', 
             alpha=0.3, color='red')
    ax1.plot(df.index, sl_band_down, ':', label=f'Stop Loss Band (-{strategy.params.stop_loss_pct*100}%)', 
             alpha=0.3, color='red')
    ax1.plot(df.index, tp_band_up, '--', label=f'Take Profit Band ({strategy.params.take_profit_ratio}x)', 
             alpha=0.3, color='green')
    ax1.plot(df.index, tp_band_down, '--', label=f'Take Profit Band (-{strategy.params.take_profit_ratio}x)', 
             alpha=0.3, color='green')
    
    ax1.set_title('Price with Stop Loss and Take Profit Bands')
    ax1.legend()
    ax1.grid(True)
    
    # Plot VWAP on second subplot
    ax2.plot(df.index, df['Close'], label='Price', alpha=0.4, color='gray')
    ax2.plot(df.index, vwap, label=f'VWAP ({strategy.params.vwap_period})', 
             alpha=0.8, color='blue', linewidth=2)
    
    # Add distance from VWAP as shaded area
    vwap_distance = df['Close'] - vwap
    ax2.fill_between(df.index, vwap, df['Close'], 
                     where=vwap_distance >= 0,
                     color='green', alpha=0.1, label='Above VWAP')
    ax2.fill_between(df.index, vwap, df['Close'],
                     where=vwap_distance < 0,
                     color='red', alpha=0.1, label='Below VWAP')
    
    ax2.set_title('VWAP Analysis')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Volume on third subplot
    ax3.bar(df.index, df['Volume'], label='Volume', alpha=0.3, color='blue')
    ax3.plot(df.index, volume_sma, label=f'Volume SMA ({strategy.params.sma_period})', 
             color='orange', alpha=0.7)
    ax3.set_title('Volume Analysis')
    ax3.legend()
    ax3.grid(True)
    
    # Plot Relative Volume on bottom subplot
    ax4.plot(df.index, relative_volume, label='Relative Volume', color='purple', alpha=0.7)
    ax4.axhline(y=strategy.params.relative_volume_multiplier, color='red', linestyle='--', 
                label=f'Threshold ({strategy.params.relative_volume_multiplier}x)', alpha=0.5)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_title('Relative Volume Analysis')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"VWAP Period: {strategy.params.vwap_period}")
    print(f"Volume SMA Period: {strategy.params.sma_period}")
    print(f"Stop Loss: {strategy.params.stop_loss_pct*100}%")
    print(f"Take Profit Ratio: {strategy.params.take_profit_ratio}x")
    print(f"Relative Volume Multiplier: {strategy.params.relative_volume_multiplier}x")
    
    # Add VWAP-specific statistics
    print("\nVWAP Analysis:")
    vwap_distance_pct = ((df['Close'] - vwap) / vwap) * 100
    print(f"Average Distance from VWAP: {np.nanmean(abs(vwap_distance_pct)):.2f}%")
    print(f"Max Distance Above VWAP: {np.nanmax(vwap_distance_pct):.2f}%")
    print(f"Max Distance Below VWAP: {np.nanmin(vwap_distance_pct):.2f}%")
    print(f"Time Above VWAP: {np.mean(df['Close'] > vwap)*100:.1f}%")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in VWAP: {np.isnan(vwap).any()}")
    print(f"NaN in Volume SMA: {np.isnan(volume_sma).any()}")
    print(f"NaN in Relative Volume: {np.isnan(relative_volume).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"VWAP Range: {np.nanmin(vwap):.2f} to {np.nanmax(vwap):.2f}")
    print(f"Average VWAP: {np.nanmean(vwap):.2f}")
    print(f"Volume SMA Range: {np.nanmin(volume_sma):.2f} to {np.nanmax(volume_sma):.2f}")
    print(f"Relative Volume Range: {np.nanmin(relative_volume):.2f} to {np.nanmax(relative_volume):.2f}")
    print(f"Average Relative Volume: {np.nanmean(relative_volume):.2f}")
    print(f"High Volume Periods: {np.sum(relative_volume > strategy.params.relative_volume_multiplier)} candles")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_vwap_breakout_strategy(file_path) 