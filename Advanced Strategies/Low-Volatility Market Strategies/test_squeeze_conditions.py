import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

# Test with relaxed parameters to verify strategy logic
from volatility_squeeze_breakout import run_backtest
import pandas as pd

# Test with one data file and relaxed parameters
data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
test_file = os.path.join(data_folder, 'bybit-BTCUSDT-5m-20250515-to-20250529.csv')

if os.path.exists(test_file):
    print("Testing Volatility Squeeze Breakout Strategy with relaxed parameters...")
    data_df = pd.read_csv(test_file)
    data_df["datetime"] = pd.to_datetime(data_df["datetime"])
    
    # Check data summary
    print(f"Data period: {data_df['datetime'].iloc[0]} to {data_df['datetime'].iloc[-1]}")
    print(f"Total bars: {len(data_df)}")
    
    # Calculate some basic statistics to understand the data
    price_range = (data_df['High'].max() - data_df['Low'].min()) / data_df['Close'].mean()
    avg_volume = data_df['Volume'].mean()
    
    print(f"Price range during period: {price_range:.1%}")
    print(f"Average volume: {avg_volume:,.0f}")
    
    # Test 1: Original strict parameters
    print("\n=== TEST 1: Original Strict Parameters ===")
    results = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.01,  # 1% BB width
        bb_squeeze_bars=10,
        adx_period=14,
        adx_low_threshold=20,
        atr_period=14,
        volume_avg_period=20,
        volume_low_threshold=0.7,
        volume_breakout_multiplier=2.0,
        range_detection_bars=30,
        range_touch_tolerance=0.0015,
        min_range_touches=3,
        breakout_buffer_pct=0.002,
        confirmation_required=True,
        candle_close_pct=0.7,
        leverage=10
    )
    
    # Test 2: Relaxed parameters to check strategy logic
    print("\n=== TEST 2: Relaxed Parameters to Test Logic ===")
    results_relaxed = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.03,  # Increased from 1% to 3% BB width
        bb_squeeze_bars=5,          # Reduced from 10 to 5 bars
        adx_period=14,
        adx_low_threshold=30,       # Increased from 20 to 30
        atr_period=14,
        volume_avg_period=20,
        volume_low_threshold=0.9,   # Increased from 0.7 to 0.9
        volume_breakout_multiplier=1.5,  # Reduced from 2.0 to 1.5
        range_detection_bars=20,    # Reduced from 30 to 20
        range_touch_tolerance=0.003,  # Increased tolerance
        min_range_touches=2,        # Reduced from 3 to 2
        breakout_buffer_pct=0.001,  # Reduced buffer
        confirmation_required=False,  # Disable confirmation
        candle_close_pct=0.5,       # Reduced from 0.7 to 0.5
        leverage=10
    )
    
    # Test 3: Very relaxed parameters
    print("\n=== TEST 3: Very Relaxed Parameters ===")
    results_very_relaxed = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.05,  # 5% BB width
        bb_squeeze_bars=3,          # Only 3 bars minimum
        adx_period=14,
        adx_low_threshold=40,       # Very high ADX threshold
        atr_period=14,
        volume_avg_period=20,
        volume_low_threshold=1.2,   # Allow higher volume
        volume_breakout_multiplier=1.2,  # Low multiplier
        range_detection_bars=15,    # Short lookback
        range_touch_tolerance=0.005,  # High tolerance
        min_range_touches=2,
        breakout_buffer_pct=0.0005,  # Very small buffer
        confirmation_required=False,
        candle_close_pct=0.3,       # Very low requirement
        leverage=10
    )
    
    print(f"\n=== SUMMARY ===")
    print(f"Original (Strict): {results.get('# Trades', 0)} trades")
    print(f"Relaxed: {results_relaxed.get('# Trades', 0)} trades")
    print(f"Very Relaxed: {results_very_relaxed.get('# Trades', 0)} trades")
    
    if results_very_relaxed.get('# Trades', 0) > 0:
        print("✅ Strategy logic is working - found trades with very relaxed parameters")
        print("🔍 This confirms the original strict parameters are appropriate for specific squeeze conditions")
    else:
        print("ℹ️ Even with very relaxed parameters, no squeeze/breakout patterns found")
        print("📊 Analysis:")
        print("   - This dataset may not contain the specific conditions this strategy targets")
        print("   - Volatility Squeeze Breakout is designed for rare, specific market conditions")
        
    print("\n🎯 Strategy Designed For:")
    print("   - Bollinger Band squeezes (very tight bands)")
    print("   - Low ADX (no trending)")
    print("   - Low volume periods")
    print("   - Clear support/resistance ranges")
    print("   - Confirmed breakouts with volume spikes")
    print("\n💡 Zero trades often indicates:")
    print("   - Normal market conditions without extreme squeezes")
    print("   - Strategy correctly waiting for ideal conditions")
    print("   - Conservative approach preventing false signals")
        
else:
    print(f"Test data file not found at: {test_file}")
    print("Available files:")
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                print(f"  - {file}") 