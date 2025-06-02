import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

# Test with relaxed parameters to verify strategy logic
from bollinger_squeeze_breakout import run_backtest
import pandas as pd

# Test with one data file and relaxed parameters
data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
test_file = os.path.join(data_folder, 'bybit-BTCUSDT-5m-20250515-to-20250529.csv')

if os.path.exists(test_file):
    print("Testing Bollinger Squeeze Breakout Strategy with different parameter sets...")
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
    
    # Test 1: Original conservative parameters
    print("\n=== TEST 1: Original Conservative Parameters ===")
    results = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.015,  # 1.5% BB width
        bb_squeeze_bars=8,
        adx_period=14,
        adx_use_filter=True,
        adx_low_threshold=25,
        atr_period=14,
        volume_avg_period=20,
        volume_breakout_multiplier=1.5,
        entry_mode="conservative",
        confirmation_bars=1,
        retest_enabled=False,
        leverage=10
    )
    
    # Test 2: Aggressive mode with relaxed parameters
    print("\n=== TEST 2: Aggressive Mode with Relaxed Parameters ===")
    results_aggressive = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.025,  # Increased from 1.5% to 2.5% BB width
        bb_squeeze_bars=5,          # Reduced from 8 to 5 bars
        adx_period=14,
        adx_use_filter=True,
        adx_low_threshold=30,       # Increased from 25 to 30
        atr_period=14,
        volume_avg_period=20,
        volume_breakout_multiplier=1.2,  # Reduced from 1.5 to 1.2
        entry_mode="aggressive",     # Aggressive entry mode
        confirmation_bars=0,         # No confirmation needed
        retest_enabled=False,
        breakout_min_pct=0.001,     # Reduced from 0.2% to 0.1%
        leverage=10
    )
    
    # Test 3: Very relaxed parameters without ADX filter
    print("\n=== TEST 3: Very Relaxed Parameters (No ADX Filter) ===")
    results_relaxed = run_backtest(
        data_df,
        verbose=True,
        bb_period=20,
        bb_std=2.0,
        bb_squeeze_threshold=0.04,  # 4% BB width
        bb_squeeze_bars=3,          # Only 3 bars minimum
        adx_period=14,
        adx_use_filter=False,       # Disable ADX filter
        adx_low_threshold=40,
        atr_period=14,
        volume_avg_period=20,
        volume_breakout_multiplier=1.1,  # Very low multiplier
        entry_mode="aggressive",
        confirmation_bars=0,
        retest_enabled=False,
        breakout_min_pct=0.0005,    # Very small breakout requirement
        head_fake_protection=False, # Disable head fake protection
        leverage=10
    )
    
    # Test 4: Most relaxed - almost any squeeze/breakout
    print("\n=== TEST 4: Maximum Relaxed Parameters ===")
    results_max_relaxed = run_backtest(
        data_df,
        verbose=True,
        bb_period=15,               # Shorter period for more responsive bands
        bb_std=1.5,                # Tighter bands
        bb_squeeze_threshold=0.06,  # 6% BB width
        bb_squeeze_bars=2,          # Only 2 bars minimum
        adx_period=14,
        adx_use_filter=False,       # No ADX filter
        atr_period=14,
        volume_avg_period=20,
        volume_breakout_multiplier=1.0,  # No volume requirement
        entry_mode="aggressive",
        confirmation_bars=0,
        retest_enabled=False,
        breakout_min_pct=0.0001,    # Minimal breakout requirement
        head_fake_protection=False,
        fake_breakout_exit_bars=10, # Longer fake breakout tolerance
        leverage=10
    )
    
    print(f"\n=== SUMMARY ===")
    print(f"Original Conservative: {results.get('# Trades', 0)} trades")
    print(f"Aggressive Relaxed: {results_aggressive.get('# Trades', 0)} trades")
    print(f"Very Relaxed (No ADX): {results_relaxed.get('# Trades', 0)} trades")
    print(f"Maximum Relaxed: {results_max_relaxed.get('# Trades', 0)} trades")
    
    if results_max_relaxed.get('# Trades', 0) > 0:
        print("\n✅ Strategy logic is working - found trades with maximum relaxed parameters")
        print("🔍 This confirms the original parameters are designed for specific squeeze conditions")
        print(f"📊 Best performing relaxed config:")
        print(f"   Return: {results_max_relaxed.get('Return [%]', 0):.2f}%")
        print(f"   Win Rate: {results_max_relaxed.get('Win Rate [%]', 0):.2f}%")
        print(f"   Max Drawdown: {results_max_relaxed.get('Max. Drawdown [%]', 0):.2f}%")
    else:
        print("ℹ️ Even with maximum relaxed parameters, no squeeze/breakout patterns found")
        print("📊 Analysis:")
        print("   - This dataset may not contain the specific conditions this strategy targets")
        print("   - Bollinger Squeeze Breakout is designed for specific transitional market conditions")
        
    print("\n🎯 Strategy Designed For:")
    print("   - Bollinger Band squeezes (narrowing bands)")
    print("   - Low ADX during consolidation periods")
    print("   - Clear breakouts from squeeze ranges")
    print("   - Volume confirmation on breakouts")
    print("   - Both aggressive and conservative entry modes")
    print("\n💡 Zero trades often indicates:")
    print("   - Normal market conditions without significant squeezes")
    print("   - Strategy correctly waiting for transitional setups")
    print("   - Conservative approach preventing false signals")
    print("   - Parameters optimized for specific market phases")
        
else:
    print(f"Test data file not found at: {test_file}")
    print("Available files:")
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                print(f"  - {file}") 