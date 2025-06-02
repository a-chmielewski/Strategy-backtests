import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '..')))

# Test with relaxed parameters to verify strategy logic
from breakout_and_retest import run_backtest
import pandas as pd

# Test with one data file and relaxed parameters
data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
test_file = os.path.join(data_folder, 'bybit-BTCUSDT-5m-20250515-to-20250529.csv')

if os.path.exists(test_file):
    print("Testing Breakout and Retest Strategy with different parameter sets...")
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
        sr_lookback_period=30,
        sr_min_touches=3,
        sr_tolerance_pct=0.002,
        breakout_min_pct=0.003,
        volume_breakout_multiplier=1.5,
        use_trend_filter=True,
        rsi_pullback_min=40,
        retest_timeout_bars=15,
        leverage=10
    )
    
    # Test 2: Relaxed support/resistance detection
    print("\n=== TEST 2: Relaxed S/R Detection ===")
    results_relaxed_sr = run_backtest(
        data_df,
        verbose=True,
        sr_lookback_period=20,        # Shorter lookback
        sr_min_touches=2,             # Fewer touches required
        sr_tolerance_pct=0.005,       # Wider tolerance (0.5%)
        breakout_min_pct=0.002,       # Smaller breakout requirement (0.2%)
        volume_breakout_multiplier=1.2, # Lower volume requirement
        use_trend_filter=False,       # No trend filter
        rsi_pullback_min=30,          # Lower RSI threshold
        retest_timeout_bars=20,       # Longer timeout
        reversal_confirmation_bars=1, # Faster confirmation
        engulfing_min_ratio=1.1,      # Easier engulfing pattern
        leverage=10
    )
    
    # Test 3: Very relaxed parameters - almost any breakout/retest
    print("\n=== TEST 3: Very Relaxed Parameters ===")
    results_very_relaxed = run_backtest(
        data_df,
        verbose=True,
        sr_lookback_period=15,        # Even shorter lookback
        sr_min_touches=2,             # Minimum touches
        sr_tolerance_pct=0.008,       # Very wide tolerance (0.8%)
        breakout_min_pct=0.001,       # Tiny breakout requirement (0.1%)
        volume_breakout_multiplier=1.0, # No volume requirement
        use_trend_filter=False,       # No trend filter
        rsi_pullback_min=20,          # Very low RSI threshold
        rsi_pullback_max=80,          # Very high RSI threshold
        retest_timeout_bars=25,       # Longer timeout
        retest_tolerance_pct=0.005,   # Wider retest tolerance
        reversal_confirmation_bars=1, # Fast confirmation
        engulfing_min_ratio=1.0,      # Any engulfing pattern
        hammer_ratio=1.5,             # Easier hammer detection
        leverage=10
    )
    
    # Test 4: Maximum relaxed - catch any potential setup
    print("\n=== TEST 4: Maximum Relaxed Parameters ===")
    results_max_relaxed = run_backtest(
        data_df,
        verbose=True,
        sr_lookback_period=10,        # Very short lookback
        sr_min_touches=2,             # Minimum touches
        sr_tolerance_pct=0.01,        # 1% tolerance
        breakout_min_pct=0.0005,      # 0.05% breakout
        volume_breakout_multiplier=0.8, # Even declining volume acceptable
        use_trend_filter=False,       # No filters
        rsi_pullback_min=10,          # Almost no RSI requirement
        rsi_pullback_max=90,          # Almost no RSI requirement
        retest_timeout_bars=30,       # Long timeout
        retest_tolerance_pct=0.008,   # Wide retest tolerance
        reversal_confirmation_bars=1, # Immediate confirmation
        engulfing_min_ratio=0.8,      # Reverse engulfing acceptable
        hammer_ratio=1.0,             # Any hammer/doji
        stop_loss_buffer_pct=0.005,   # Wider stops
        first_target_pct=0.002,       # Smaller first target
        leverage=10
    )
    
    print(f"\n=== SUMMARY ===")
    print(f"Original Conservative: {results.get('# Trades', 0)} trades")
    print(f"Relaxed S/R Detection: {results_relaxed_sr.get('# Trades', 0)} trades")
    print(f"Very Relaxed: {results_very_relaxed.get('# Trades', 0)} trades")
    print(f"Maximum Relaxed: {results_max_relaxed.get('# Trades', 0)} trades")
    
    # Find best performing configuration
    all_tests = [
        ("Original Conservative", results),
        ("Relaxed S/R Detection", results_relaxed_sr),
        ("Very Relaxed", results_very_relaxed),
        ("Maximum Relaxed", results_max_relaxed)
    ]
    
    best_test = max(all_tests, key=lambda x: x[1].get('# Trades', 0))
    
    if best_test[1].get('# Trades', 0) > 0:
        print(f"\n✅ Strategy logic is working - best config: {best_test[0]}")
        print(f"📊 Best performing configuration results:")
        print(f"   Trades: {best_test[1].get('# Trades', 0)}")
        print(f"   Return: {best_test[1].get('Return [%]', 0):.2f}%")
        print(f"   Win Rate: {best_test[1].get('Win Rate [%]', 0):.2f}%")
        print(f"   Max Drawdown: {best_test[1].get('Max. Drawdown [%]', 0):.2f}%")
    else:
        print("ℹ️ Even with maximum relaxed parameters, no breakout/retest patterns found")
        print("📊 Analysis:")
        print("   - This dataset may not contain clear breakout/retest setups")
        print("   - Strategy designed for transitional trend changes")
        
    print("\n🎯 Strategy Designed For:")
    print("   - Clear support/resistance level formation")
    print("   - Volume-confirmed breakouts of key levels")
    print("   - Retests of broken levels (old resistance becomes support)")
    print("   - Reversal confirmation at retest levels")
    print("   - Trend continuation after successful retests")
    print("\n💡 Main strategy characteristics:")
    print("   - Waits for confirmation before entry (safer than immediate breakout)")
    print("   - Tight stops near support/resistance levels")
    print("   - Good risk-to-reward ratio")
    print("   - Suitable for transitional market conditions")
    print("   - Catches trend changes with confirmation")
        
else:
    print(f"Test data file not found at: {test_file}")
    print("Available files:")
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                print(f"  - {file}") 