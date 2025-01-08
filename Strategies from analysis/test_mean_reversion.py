import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import backtrader as bt
from mean_reversion import VolatilityMeanReversionStrategy, PercentRank

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
    
    # If there's still concern about data gaps, you can forward-fill:
    # (Uncomment if needed, but only if your data is missing some timestamps)
    # df = df.set_index('datetime')
    # df = df.asfreq('T')  # assumes 1-minute frequency; adjust if needed
    # df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(method='ffill')
    # df = df.reset_index()

    return df

def test_mean_reversion_strategy(file_path: str):
    """Test Mean Reversion Strategy indicators calculation and visualization"""
    # Load and clean market data
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
    cerebro.addstrategy(VolatilityMeanReversionStrategy)
    
    # Run strategy to calculate indicators
    print("Calculating indicators...")
    results = cerebro.run()
    strategy = results[0]
    
    # Extract indicator values for plotting
    bb_top = np.array(strategy.bb.top.array)
    bb_mid = np.array(strategy.bb.mid.array)
    bb_bot = np.array(strategy.bb.bot.array)
    atr = np.array(strategy.atr.array)
    atr_percentile = np.array(strategy.atr_percentile.percentrank.array)
    
    # Debugging prints
    print("\nDebugging ATR Percentile:")
    print(f"ATR Percentile shape: {atr_percentile.shape}")
    print(f"First few ATR values: {atr[:5]}")
    print(f"First few ATR Percentile values: {atr_percentile[:5]}")
    print(f"Number of non-NaN ATR Percentile values: {np.sum(~np.isnan(atr_percentile))}")
    
    # Trim to dataset length
    valid_length = len(df)
    bb_top = bb_top[-valid_length:]
    bb_mid = bb_mid[-valid_length:]
    bb_bot = bb_bot[-valid_length:]
    atr = atr[-valid_length:]
    atr_percentile = atr_percentile[-valid_length:]
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 15), height_ratios=[2, 1, 1])
    
    # Price and BB
    ax1.plot(df.index, df['Close'], label='Price', alpha=0.7, color='black')
    ax1.plot(df.index, bb_top, label='BB Upper', alpha=0.5, color='red', linestyle='--')
    ax1.plot(df.index, bb_mid, label='BB Middle', alpha=0.5, color='gray', linestyle='--')
    ax1.plot(df.index, bb_bot, label='BB Lower', alpha=0.5, color='green', linestyle='--')
    ax1.fill_between(df.index, bb_top, bb_bot, alpha=0.1, color='gray')

    # ATR-based bands
    atr_upper = df['Close'] + (atr * strategy.params.stop_loss_atr_mult)
    atr_lower = df['Close'] - (atr * strategy.params.stop_loss_atr_mult)
    ax1.plot(df.index, atr_upper, '--', label=f'ATR Upper ({strategy.params.stop_loss_atr_mult}x)', 
             alpha=0.3, color='red')
    ax1.plot(df.index, atr_lower, '--', label=f'ATR Lower ({strategy.params.stop_loss_atr_mult}x)', 
             alpha=0.3, color='green')

    ax1.set_title('Price with Bollinger Bands and ATR Bands')
    ax1.legend()
    ax1.grid(True)
    
    # ATR
    ax2.plot(df.index, atr, label=f'ATR ({strategy.params.atr_period})', color='orange', alpha=0.7)
    ax2.set_title('Average True Range (ATR)')
    ax2.legend()
    ax2.grid(True)
    
    # ATR Percentile Rank
    ax3.plot(df.index, atr_percentile, label='ATR Percentile Rank', color='purple', alpha=0.7)
    ax3.axhline(y=strategy.params.atr_threshold_percentile, color='red', linestyle='--', 
                label=f'Threshold ({strategy.params.atr_threshold_percentile})', alpha=0.5)
    ax3.set_ylim(0, 100)
    ax3.set_title('ATR Percentile Rank')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print indicator parameters
    print("\nIndicator Parameters:")
    print(f"Bollinger Period: {strategy.params.bb_period}")
    print(f"Bollinger Std Dev: {strategy.params.bb_dev}")
    print(f"ATR Period: {strategy.params.atr_period}")
    print(f"ATR Percentile Threshold: {strategy.params.atr_threshold_percentile}")
    print(f"Stop Loss ATR Multiple: {strategy.params.stop_loss_atr_mult}")
    print(f"Take Profit ATR Multiple: {strategy.params.take_profit_atr_mult}")
    print(f"Trailing ATR Multiple: {strategy.params.trailing_atr}")
    
    # Check for NaN values
    print("\nIndicator Data Quality Check:")
    print(f"NaN in BB Upper: {np.isnan(bb_top).any()}")
    print(f"NaN in BB Middle: {np.isnan(bb_mid).any()}")
    print(f"NaN in BB Lower: {np.isnan(bb_bot).any()}")
    print(f"NaN in ATR: {np.isnan(atr).any()}")
    print(f"NaN in ATR Percentile: {np.isnan(atr_percentile).any()}")
    
    # Print basic stats
    print("\nIndicator Statistics:")
    if not np.isnan(atr).all():
        print(f"ATR Range: {np.nanmin(atr):.2f} to {np.nanmax(atr):.2f}")
        print(f"Average ATR: {np.nanmean(atr):.2f}")
    bb_width = bb_top - bb_bot
    print(f"BB Width Average: {np.nanmean(bb_width):.2f}")
    if not np.isnan(atr_percentile).all():
        print(f"ATR Percentile Range: {np.nanmin(atr_percentile):.2f} to {np.nanmax(atr_percentile):.2f}")

    if not np.isnan(atr_percentile).all():
        print("\nDetailed ATR Percentile Statistics:")
        print(f"Mean: {np.nanmean(atr_percentile):.2f}")
        print(f"Median: {np.nanmedian(atr_percentile):.2f}")
        print(f"Std Dev: {np.nanstd(atr_percentile):.2f}")
        print(f"Min: {np.nanmin(atr_percentile):.2f}")
        print(f"Max: {np.nanmax(atr_percentile):.2f}")
        print(f"Quartiles: {np.nanpercentile(atr_percentile, [25, 50, 75])}")
    
    # Check PercentRank settings
    print("\nPercentRank Indicator Settings:")
    print(f"Period setting: {strategy.atr_percentile.params.period}")

if __name__ == "__main__":
    file_path = r"F:\Algo Trading TRAINING\Strategy backtests\data\bybit-BTCUSDT-5m-20240929-to-20241128.csv"
    test_mean_reversion_strategy(file_path)
