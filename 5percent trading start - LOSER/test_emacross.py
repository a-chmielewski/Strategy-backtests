import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from EMACrossover import EMACrossStrategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_emacross_strategy(file_path: str):
    """Test EMACrossover strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(EMACrossStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    ema_short_values = np.array(strategy.ema_short.lines[0].array)
    ema_long_values = np.array(strategy.ema_long.lines[0].array)
    stoch_k_values = np.array(strategy.stochastic.lines.percK.array)
    stoch_d_values = np.array(strategy.stochastic.lines.percD.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    ema_short_values = ema_short_values[-valid_length:]
    ema_long_values = ema_long_values[-valid_length:]
    stoch_k_values = stoch_k_values[-valid_length:]
    stoch_d_values = stoch_d_values[-valid_length:]
    
    # Create visualization
    plt.figure(figsize=(20, 12))
    
    # Plot price and EMAs
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Price', alpha=0.7)
    plt.plot(df.index, ema_short_values, label=f'EMA {strategy.p.ema_short}', alpha=0.7)
    plt.plot(df.index, ema_long_values, label=f'EMA {strategy.p.ema_long}', alpha=0.7)
    plt.title('Price and EMAs')
    plt.legend()
    plt.grid(True)
    
    # Plot Stochastic
    plt.subplot(2, 1, 2)
    plt.plot(df.index, stoch_k_values, label='Stoch %K', alpha=0.7)
    plt.plot(df.index, stoch_d_values, label='Stoch %D', alpha=0.7)
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    plt.title('Stochastic Oscillator')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"EMA Short period: {strategy.p.ema_short}")
    print(f"EMA Long period: {strategy.p.ema_long}")
    print(f"Stochastic period: {strategy.p.stochastic_period}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in EMA Short: {np.isnan(ema_short_values).any()}")
    print(f"NaN in EMA Long: {np.isnan(ema_long_values).any()}")
    print(f"NaN in Stochastic K: {np.isnan(stoch_k_values).any()}")
    print(f"NaN in Stochastic D: {np.isnan(stoch_d_values).any()}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_emacross_strategy(file_path)