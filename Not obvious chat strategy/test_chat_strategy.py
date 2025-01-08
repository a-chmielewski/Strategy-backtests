import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from chat_strategy import StrategyTemplate, VwapIntradayIndicator, KeltnerChannels, StochRSI

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_chat_strategy(file_path: str):
    """Test Chat Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(StrategyTemplate)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    vwap = np.array(strategy.vwap_intraday.array)
    kc_upper = np.array(strategy.keltner.kc_upper.array)
    kc_middle = np.array(strategy.keltner.kc_middle.array)
    kc_lower = np.array(strategy.keltner.kc_lower.array)
    stochrsi_k = np.array(strategy.stochrsi.stochrsi_k.array)
    stochrsi_d = np.array(strategy.stochrsi.stochrsi_d.array)
    rsi = np.array(strategy.stochrsi.rsi.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    vwap = vwap[-valid_length:]
    kc_upper = kc_upper[-valid_length:]
    kc_middle = kc_middle[-valid_length:]
    kc_lower = kc_lower[-valid_length:]
    stochrsi_k = stochrsi_k[-valid_length:]
    stochrsi_d = stochrsi_d[-valid_length:]
    rsi = rsi[-valid_length:]
    
    # Create visualization with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[2, 1, 1])
    
    # Plot price, VWAP and Keltner Channels on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, vwap, label='VWAP', alpha=0.7, color='blue')
    ax1.plot(df.index, kc_upper, label='KC Upper', alpha=0.5, color='red', linestyle='--')
    ax1.plot(df.index, kc_middle, label='KC Middle', alpha=0.5, color='gray', linestyle='--')
    ax1.plot(df.index, kc_lower, label='KC Lower', alpha=0.5, color='green', linestyle='--')
    ax1.fill_between(df.index, kc_upper, kc_lower, alpha=0.1, color='gray')
    
    ax1.set_title('Price with VWAP and Keltner Channels')
    ax1.legend()
    ax1.grid(True)
    
    # Plot StochRSI on middle subplot
    ax2.plot(df.index, stochrsi_k, label='StochRSI %K', color='blue', alpha=0.7)
    ax2.plot(df.index, stochrsi_d, label='StochRSI %D', color='red', alpha=0.7)
    ax2.axhline(y=80, color='red', linestyle='--', label='Overbought (80)', alpha=0.5)
    ax2.axhline(y=20, color='green', linestyle='--', label='Oversold (20)', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('Stochastic RSI')
    ax2.legend()
    ax2.grid(True)
    
    # Plot RSI on bottom subplot
    ax3.plot(df.index, rsi, label='RSI', color='purple', alpha=0.7)
    ax3.axhline(y=70, color='red', linestyle='--', label='Overbought (70)', alpha=0.5)
    ax3.axhline(y=30, color='green', linestyle='--', label='Oversold (30)', alpha=0.5)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax3.set_ylim(0, 100)
    ax3.set_title('RSI')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Keltner Channel Period: {strategy.keltner.params.period}")
    print(f"Keltner Channel Multiplier: {strategy.keltner.params.multiplier}")
    print(f"Keltner Channel ATR Period: {strategy.keltner.params.atr_period}")
    print(f"StochRSI RSI Period: {strategy.stochrsi.params.rsi_period}")
    print(f"StochRSI Stoch Period: {strategy.stochrsi.params.stoch_period}")
    print(f"StochRSI K Smoothing: {strategy.stochrsi.params.smooth_k}")
    print(f"StochRSI D Smoothing: {strategy.stochrsi.params.smooth_d}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in VWAP: {np.isnan(vwap).any()}")
    print(f"NaN in KC Upper: {np.isnan(kc_upper).any()}")
    print(f"NaN in KC Middle: {np.isnan(kc_middle).any()}")
    print(f"NaN in KC Lower: {np.isnan(kc_lower).any()}")
    print(f"NaN in StochRSI K: {np.isnan(stochrsi_k).any()}")
    print(f"NaN in StochRSI D: {np.isnan(stochrsi_d).any()}")
    print(f"NaN in RSI: {np.isnan(rsi).any()}")
    
    # Print basic statistics
    print("\nIndicator Statistics:")
    print(f"StochRSI %K Range: {np.nanmin(stochrsi_k):.2f} to {np.nanmax(stochrsi_k):.2f}")
    print(f"StochRSI %D Range: {np.nanmin(stochrsi_d):.2f} to {np.nanmax(stochrsi_d):.2f}")
    print(f"RSI Range: {np.nanmin(rsi):.2f} to {np.nanmax(rsi):.2f}")
    print(f"KC Width Average: {np.nanmean(kc_upper - kc_lower):.2f}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_chat_strategy(file_path) 