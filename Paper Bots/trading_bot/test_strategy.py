import pandas as pd
import numpy as np
from strategies.double_ema_stoch import DoubleEMAStochStrategy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Ensure we have all required columns
    required_columns = ['datetime', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_strategy_on_market_data(file_path: str):
    # Initialize strategy with original parameters from backtest
    strategy = DoubleEMAStochStrategy(
        ema_slow=50,
        ema_fast=150,  # Back to original value
        stoch_k=5,
        stoch_d=3,
        slowing=3,
        stoch_overbought=80,
        stoch_oversold=20,
        stop_loss=0.0025,
        take_profit=0.005
    )
    
    # Load market data
    print(f"Loading market data from {file_path}...")
    df = load_market_data(file_path)
    print(f"Loaded {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    
    # Lists to store signals and prices
    signals = []
    prices = []
    
    # Process each candle
    print("Processing candles and generating signals...")
    for i in range(len(df)):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        close = df['Close'].iloc[i]
        
        # Update indicators
        strategy.calculate_indicators(high, low, close)
        
        # Generate signal
        signal = strategy.generate_signal(close)
        signals.append(signal)
        prices.append(close)
        
        # Print signal if generated
        # if signal:
        #     print(f"\nSignal at {df['datetime'].iloc[i]}:")
        #     print(f"Price: {close:.2f}")
        #     print(f"Action: {signal['action']}")
        #     print(f"Stop Loss: {signal['stop_loss']:.2f}")
        #     print(f"Take Profit: {signal['take_profit']:.2f}")
    
    # Plot results
    plt.figure(figsize=(20, 12))
    
    # Plot price and EMAs
    plt.subplot(2, 1, 1)
    plt.plot(df['Close'], label='Price', alpha=0.7)
    
    # Calculate the offset for EMAs (they will be shorter than the price series)
    if len(strategy.state.ema_slow) > 0:
        ema_offset = len(df) - len(strategy.state.ema_slow)
        plt.plot(range(ema_offset, len(df)), 
                strategy.state.ema_slow, 
                label=f'EMA {strategy.ema_slow}', 
                alpha=0.7)
    
    if len(strategy.state.ema_fast) > 0:
        ema_offset = len(df) - len(strategy.state.ema_fast)
        plt.plot(range(ema_offset, len(df)), 
                strategy.state.ema_fast, 
                label=f'EMA {strategy.ema_fast}', 
                alpha=0.7)
    
    # Plot buy/sell signals
    for i, signal in enumerate(signals):
        if signal:
            if signal['action'] == 'BUY':
                plt.scatter(i, prices[i], color='green', marker='^', s=100)
            else:
                plt.scatter(i, prices[i], color='red', marker='v', s=100)
    
    plt.title('BTC/USDT Price, EMAs and Signals')
    plt.legend()
    plt.grid(True)
    
    # Plot Stochastic
    plt.subplot(2, 1, 2)
    if len(strategy.state.stoch_k) > 0:
        plt.plot(strategy.state.stoch_k, label='Stoch %K', alpha=0.7)
    if len(strategy.state.stoch_d) > 0:
        plt.plot(strategy.state.stoch_d, label='Stoch %D', alpha=0.7)
    
    # Add overbought/oversold lines
    plt.axhline(y=strategy.stoch_overbought, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=strategy.stoch_oversold, color='g', linestyle='--', alpha=0.5)
    
    plt.title('Stochastic Oscillator')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    buy_signals = len([s for s in signals if s and s['action'] == 'BUY'])
    sell_signals = len([s for s in signals if s and s['action'] == 'SELL'])
    
    print("\nTest Summary:")
    print(f"Total periods processed: {len(df)}")
    print(f"Buy signals generated: {buy_signals}")
    print(f"Sell signals generated: {sell_signals}")
    print(f"Total signals: {buy_signals + sell_signals}")
    
    # Calculate signal frequency
    total_signals = buy_signals + sell_signals
    signals_per_day = total_signals / (len(df) / 1440)  # Assuming 1-minute data
    
    print(f"\nSignal Frequency:")
    print(f"Average signals per day: {signals_per_day:.2f}")
    
    # Verify indicator calculations
    print("\nIndicator Verification:")
    print(f"EMA Slow length: {len(strategy.state.ema_slow)}")
    print(f"EMA Fast length: {len(strategy.state.ema_fast)}")
    print(f"Stochastic %K length: {len(strategy.state.stoch_k)}")
    print(f"Stochastic %D length: {len(strategy.state.stoch_d)}")
    
    # Check for any NaN values in indicators
    print("\nChecking for NaN values:")
    print(f"NaN in EMA Slow: {np.isnan(strategy.state.ema_slow).any()}")
    print(f"NaN in EMA Fast: {np.isnan(strategy.state.ema_fast).any()}")
    print(f"NaN in Stoch %K: {np.isnan(strategy.state.stoch_k).any()}")
    print(f"NaN in Stoch %D: {np.isnan(strategy.state.stoch_d).any()}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_strategy_on_market_data(file_path) 