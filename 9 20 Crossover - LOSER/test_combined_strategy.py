import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from EMA_BB_pSAR_RSI import EMA_BB_PSAR_RSI

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_combined_strategy(file_path: str):
    """Test Combined Strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(EMA_BB_PSAR_RSI)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    ema_short = np.array(strategy.ema_short.array)
    ema_long = np.array(strategy.ema_long.array)
    bb_top = np.array(strategy.bb_upper.array)
    bb_mid = np.array(strategy.bb.mid.array)
    bb_bot = np.array(strategy.bb_lower.array)
    psar = np.array(strategy.sar.array)
    rsi = np.array(strategy.rsi.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    ema_short = ema_short[-valid_length:]
    ema_long = ema_long[-valid_length:]
    bb_top = bb_top[-valid_length:]
    bb_mid = bb_mid[-valid_length:]
    bb_bot = bb_bot[-valid_length:]
    psar = psar[-valid_length:]
    rsi = rsi[-valid_length:]
    
    # Create visualization with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), height_ratios=[2, 1])
    
    # Plot price, EMAs, Bollinger Bands, and PSAR on top subplot
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, ema_short, label=f'EMA {strategy.params.ema_short}', alpha=0.7)
    ax1.plot(df.index, ema_long, label=f'EMA {strategy.params.ema_long}', alpha=0.7)
    ax1.plot(df.index, bb_top, label='BB Upper', alpha=0.5, linestyle='--', color='gray')
    ax1.plot(df.index, bb_mid, label='BB Middle', alpha=0.5, linestyle='--', color='gray')
    ax1.plot(df.index, bb_bot, label='BB Lower', alpha=0.5, linestyle='--', color='gray')
    ax1.scatter(df.index, psar, label='PSAR', alpha=0.5, color='purple', marker='.')
    ax1.fill_between(df.index, bb_top, bb_bot, alpha=0.1, color='gray')
    
    ax1.set_title('Price with EMAs, Bollinger Bands, and Parabolic SAR')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI on bottom subplot
    ax2.plot(df.index, rsi, label='RSI', color='blue', alpha=0.7)
    ax2.axhline(y=strategy.params.rsi_overbought, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=strategy.params.rsi_oversold, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI Indicator')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Short EMA Period: {strategy.params.ema_short}")
    print(f"Long EMA Period: {strategy.params.ema_long}")
    print(f"BB Period: {strategy.params.bb_period}")
    print(f"BB Deviation Factor: {strategy.params.bb_devfactor}")
    print(f"RSI Period: {strategy.params.rsi_period}")
    print(f"RSI Overbought: {strategy.params.rsi_overbought}")
    print(f"RSI Oversold: {strategy.params.rsi_oversold}")
    print(f"PSAR Step: {strategy.params.sar_step}")
    print(f"PSAR Max: {strategy.params.sar_max}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in Short EMA: {np.isnan(ema_short).any()}")
    print(f"NaN in Long EMA: {np.isnan(ema_long).any()}")
    print(f"NaN in BB Upper: {np.isnan(bb_top).any()}")
    print(f"NaN in BB Middle: {np.isnan(bb_mid).any()}")
    print(f"NaN in BB Lower: {np.isnan(bb_bot).any()}")
    print(f"NaN in PSAR: {np.isnan(psar).any()}")
    print(f"NaN in RSI: {np.isnan(rsi).any()}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_combined_strategy(file_path) 