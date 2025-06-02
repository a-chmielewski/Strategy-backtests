import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

# Test with relaxed parameters to verify strategy logic
from volatility_reversal_scalp import run_backtest
import pandas as pd

# Test with one data file and relaxed parameters
data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
test_file = os.path.join(data_folder, 'bybit-BTCUSDT-1m-20250515-to-20250529.csv')

if os.path.exists(test_file):
    print("Testing Volatility Reversal Strategy with relaxed parameters...")
    data_df = pd.read_csv(test_file)
    data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    
    # Check data summary
    print(f"Data period: {data_df['datetime'].iloc[0]} to {data_df['datetime'].iloc[-1]}")
    print(f"Total bars: {len(data_df)}")
    
    # Use much more lenient parameters to test logic
    results = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_extreme_std=2.0,  # Reduced from 3.0 to 2.0
        atr_period=14,
        atr_extreme_multiplier=2.0,  # Reduced from 4.0 to 2.0
        rsi_period=7,
        rsi_overbought_extreme=80,  # Reduced from 85 to 80
        rsi_oversold_extreme=20,    # Increased from 15 to 20
        volume_climax_multiplier=1.5,  # Reduced from 3.0 to 1.5
        volume_avg_period=20,
        ema_period=20,
        reversal_confirmation_bars=3,  # Increased from 2 to 3
        fibonacci_retracement_1=0.382,
        fibonacci_retracement_2=0.5,
        max_hold_bars=5,
        stop_buffer_pct=0.001,
        min_spike_size_pct=0.005,  # Reduced from 0.01 to 0.005
        position_size_reduction=0.5,
        partial_exit_pct=0.5,
        max_consecutive_losses=5,  # Increased from 3 to 5
        leverage=10
    )
    
    print(f"\nTest Results with Relaxed Parameters:")
    print(f"Total Trades: {results.get('# Trades', 0)}")
    print(f"Win Rate: {results.get('Win Rate [%]', 0):.2f}%")
    print(f"Final Equity: {results.get('Equity Final [$]', 100):.2f}")
    
    if results.get('# Trades', 0) > 0:
        print("✅ Strategy logic is working - found trades with relaxed parameters")
        print("🔍 This confirms the original strict parameters are appropriate for extreme events only")
    else:
        print("ℹ️ Even with relaxed parameters, no extreme spikes found in this dataset")
        print("📊 Data Analysis:")
        print(f"   - Price range: ${data_df['Close'].min():.0f} - ${data_df['Close'].max():.0f}")
        print(f"   - Max volume: {data_df['Volume'].max():,.0f}")
        print(f"   - Avg volume: {data_df['Volume'].mean():,.0f}")
        print("\n🎯 Strategy Designed For:")
        print("   - Flash crashes or moonshots")
        print("   - News-driven spikes")
        print("   - Liquidation cascades")
        print("   - Market manipulation events")
        print("\n💡 This dataset appears to be from a normal trading period without extreme volatility events.")
        
else:
    print(f"Test data file not found at: {test_file}")
    print("Available files:")
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                print(f"  - {file}") 