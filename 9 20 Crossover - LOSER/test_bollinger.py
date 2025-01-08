import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from bollinger import BBStrategy

def load_market_data(file_path: str) -> pd.DataFrame:
    """Load and prepare market data from CSV file"""
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    return df

def test_bollinger_strategy(file_path: str):
    """Test Bollinger Bands strategy indicators calculation and visualization"""
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
    cerebro.addstrategy(BBStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    bb_mid = np.array(strategy.bollinger.mid.array)
    bb_top = np.array(strategy.bb_upper.array)
    bb_bot = np.array(strategy.bb_lower.array)
    
    # Trim any NaN values at the beginning (warmup period)
    valid_length = len(df)
    bb_mid = bb_mid[-valid_length:]
    bb_top = bb_top[-valid_length:]
    bb_bot = bb_bot[-valid_length:]
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    
    # Plot price and Bollinger Bands
    plt.plot(df.index, df['Close'], label='Price', alpha=0.7)
    plt.plot(df.index, bb_mid, label='BB Middle', alpha=0.7)
    plt.plot(df.index, bb_top, label='BB Upper', alpha=0.7)
    plt.plot(df.index, bb_bot, label='BB Lower', alpha=0.7)
    plt.fill_between(df.index, bb_top, bb_bot, alpha=0.1)
    
    plt.title('Price and Bollinger Bands')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"BB Period: {strategy.params.period}")
    print(f"BB Deviation Factor: {strategy.params.devfactor}")
    print(f"Stop Loss: {strategy.params.stop_loss}")
    print(f"Take Profit: {strategy.params.take_profit}")
    
    # Check for any NaN values in indicators
    print("\nIndicator Data Quality Check:")
    print(f"NaN in BB Middle: {np.isnan(bb_mid).any()}")
    print(f"NaN in BB Upper: {np.isnan(bb_top).any()}")
    print(f"NaN in BB Lower: {np.isnan(bb_bot).any()}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_bollinger_strategy(file_path) 