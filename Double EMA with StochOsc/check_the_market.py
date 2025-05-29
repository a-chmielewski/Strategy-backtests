import argparse
import datetime
import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta  # For technical indicators
import os
import sys
import scipy.signal # For find_peaks
from scipy.signal import find_peaks # Explicit import for clarity

# Attempt to import bybit_keys, assuming it's in the parent directory or accessible via sys.path
try:
    # If bybit_keys.py is in the root of the project (one level up from 'Double EMA with StochOsc')
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import bybit_keys as keys
except ImportError:
    # Fallback if bybit_keys.py is in the same directory or already in PYTHONPATH
    try:
        import bybit_keys as keys
    except ImportError:
        print("Error: bybit_keys.py not found. Please ensure it is in the project root or accessible in your PYTHONPATH.")
        print("It should contain your Bybit API_KEY and API_SECRET.")
        keys = None # Set keys to None so the script can still be loaded but data fetching will fail

# --- Configuration Constants (defaults, can be overridden by strategy params if needed) ---
DEFAULT_SYMBOL = 'LINK/USDT'
DEFAULT_TIMEFRAME = '1m'
DATA_FETCH_LIMIT = 500  # Number of recent candles to fetch

# Parameters based on the strategy description and common use
EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 150
ADX_WINDOW = 25
ATR_WINDOW = 14
CHOP_WINDOW = 14
PULLBACK_SWINGS_LOOKBACK = 20
STOP_LOSS_PCT = 0.01 # From DoubleEMA_StochOsc strategy
MIN_SWING_PROMINENCE = 0.001 # Prominence for swing detection (0.1% of price variation)

# --- Data Fetching ---
def fetch_recent_bybit_data(symbol, timeframe, limit):
    """Fetches the most recent N (limit) OHLCV candles from Bybit."""
    if keys is None:
        print("API keys not loaded. Cannot fetch data.")
        return pd.DataFrame()

    bybit_exchange = ccxt.bybit({
        'apiKey': keys.API_KEY,
        'secret': keys.API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot', # Assuming spot, adjust if futures
        }
    })
    try:
        print(f"Fetching last {limit} candles for {symbol} on {timeframe} timeframe...")
        # Fetch OHLCV data
        ohlcv_data = bybit_exchange.fetch_ohlcv(
            symbol,
            timeframe,
            limit=limit
        )
        if not ohlcv_data:
            print("No data fetched from Bybit.")
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv_data,
            columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Ensure data is sorted by time
        df.sort_index(inplace=True)
        
        print(f"Successfully fetched {len(df)} candles.")
        return df

    except ccxt.NetworkError as e:
        print(f"Bybit API NetworkError: {e}")
    except ccxt.ExchangeError as e:
        print(f"Bybit API ExchangeError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during data fetching: {e}")
    return pd.DataFrame()

# --- Indicator Calculations ---
def calculate_indicators(df):
    """Calculates all necessary technical indicators."""
    if df.empty:
        return df

    print("Calculating indicators...")
    # EMAs
    df['ema_fast'] = ta.ema(df['Close'], length=EMA_FAST_PERIOD)
    df['ema_slow'] = ta.ema(df['Close'], length=EMA_SLOW_PERIOD)

    # ADX
    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=ADX_WINDOW)
    if adx_df is not None and not adx_df.empty:
        df['adx'] = adx_df[f'ADX_{ADX_WINDOW}']
    else:
        df['adx'] = np.nan
        print("Warning: Could not calculate ADX.")
        
    # ATR (as percentage of price)
    atr_series = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_WINDOW)
    if atr_series is not None and not atr_series.empty:
        df['atr_percent'] = (atr_series / df['Close']) * 100
    else:
        df['atr_percent'] = np.nan
        print("Warning: Could not calculate ATR.")

    # Choppiness Index
    chop_series = ta.chop(df['High'], df['Low'], df['Close'], length=CHOP_WINDOW)
    if chop_series is not None and not chop_series.empty:
         # pandas-ta might return a DataFrame for chop, ensure we get the series
        if isinstance(chop_series, pd.DataFrame):
            df['chop'] = chop_series.iloc[:, 0] 
        else:
            df['chop'] = chop_series
    else:
        df['chop'] = np.nan
        print("Warning: Could not calculate Choppiness Index.")
        
    df.dropna(inplace=True) # Remove rows with NaN from indicator calculations
    print(f"Indicators calculated. Shape of df after NaN drop: {df.shape}")
    return df

# --- Condition Checks (Placeholders) ---
def check_trend_condition(df_row):
    """
    Checks for a clear, persistent trend.
    - Fast EMA > Slow EMA (bull) or < (sell)
    - ADX >= 25 (strength)
    Returns: 1 if condition met, 0 otherwise.
    """
    if pd.isna(df_row['ema_fast']) or pd.isna(df_row['ema_slow']) or pd.isna(df_row['adx']):
        return 0
    
    is_bull_trend = df_row['ema_fast'] > df_row['ema_slow']
    is_bear_trend = df_row['ema_fast'] < df_row['ema_slow']
    has_strength = df_row['adx'] >= 25
    
    if (is_bull_trend or is_bear_trend) and has_strength:
        return 1
    return 0

def check_pullback_condition(df, latest_data_row):
    """
    Checks for shallow, orderly pullbacks.
    - Median pullback depth over last N swings <= 0.6 * stop_loss (e.g., 0.6% if SL is 1%).
    Returns: 1 if condition met, 0 otherwise.
    """
    if df.empty or len(df) < PULLBACK_SWINGS_LOOKBACK * 2: # Need enough data for swings
        print("Pullback Check: Not enough data for swing analysis.")
        return 0

    if pd.isna(latest_data_row['ema_fast']) or pd.isna(latest_data_row['ema_slow']):
        print("Pullback Check: EMAs not available for trend determination.")
        return 0

    # Determine trend from the latest data point
    is_uptrend = latest_data_row['ema_fast'] > latest_data_row['ema_slow']
    is_downtrend = latest_data_row['ema_fast'] < latest_data_row['ema_slow']

    if not is_uptrend and not is_downtrend:
        print("Pullback Check: Market is not trending (EMAs are equal or too close).")
        return 0 # No clear trend to define pullbacks against

    # Detect peaks and troughs
    # Prominence can be tuned, e.g., based on ATR or a percentage of price range
    # For simplicity, using a small fraction of the mean price as prominence guide
    # The actual prominence value for find_peaks should be absolute, not percentage based on each point.
    avg_price_for_prominence = df['Close'].mean()
    required_prominence = avg_price_for_prominence * MIN_SWING_PROMINENCE

    peaks_indices, _ = find_peaks(df['High'], prominence=required_prominence)
    troughs_indices, _ = find_peaks(-df['Low'], prominence=required_prominence) # Invert Lows to find troughs as peaks

    if len(peaks_indices) == 0 or len(troughs_indices) == 0:
        print("Pullback Check: Not enough distinct peaks or troughs found.")
        return 0

    # Combine and sort swing points by index
    swings = []
    for p_idx in peaks_indices: swings.append({'type': 'peak', 'idx': p_idx, 'price': df['High'].iloc[p_idx]})
    for t_idx in troughs_indices: swings.append({'type': 'trough', 'idx': t_idx, 'price': df['Low'].iloc[t_idx]})
    
    # Sort by index, then by type (peak first if same index, though unlikely with OHLC)
    swings.sort(key=lambda x: (x['idx'], x['type']))

    # Filter out consecutive same-type swings (e.g. peak, peak -> keep only one, typically the more extreme)
    # For this simplified version, we will assume find_peaks with prominence handles this sufficiently.
    # A more robust way is to alternate: peak, trough, peak, trough...
    # This can be done by iterating and ensuring the type changes.
    filtered_swings = []
    if not swings: return 0
    
    last_swing_type = None
    for swing in swings:
        if not filtered_swings or swing['type'] != last_swing_type:
            filtered_swings.append(swing)
            last_swing_type = swing['type']
        # If same type, potentially update if more extreme (e.g. higher peak or lower trough)
        # For now, keeping it simple: first of consecutive type wins due to sort and prominence filtering.

    swings = filtered_swings
    if len(swings) < 2: # Need at least one peak and one trough to form a pullback
        print("Pullback Check: Not enough alternating swings found.")
        return 0
        
    pullback_depths = []
    # Iterate through swings to find pullbacks relative to the identified trend
    for i in range(len(swings) - 1):
        current_swing = swings[i]
        next_swing = swings[i+1]

        if is_uptrend:
            # In an uptrend, a pullback is from a peak down to a trough
            if current_swing['type'] == 'peak' and next_swing['type'] == 'trough':
                peak_price = current_swing['price']
                trough_price = next_swing['price']
                if peak_price > trough_price: # Ensure it's a downward move
                    depth = (peak_price - trough_price) / peak_price
                    pullback_depths.append(depth)
        elif is_downtrend:
            # In a downtrend, a pullback (rally) is from a trough up to a peak
            if current_swing['type'] == 'trough' and next_swing['type'] == 'peak':
                trough_price = current_swing['price']
                peak_price = next_swing['price']
                if peak_price > trough_price: # Ensure it's an upward move
                    # Depth relative to the start of the rally (trough_price)
                    depth = (peak_price - trough_price) / trough_price 
                    pullback_depths.append(depth)
    
    if not pullback_depths:
        print("Pullback Check: No valid pullbacks identified for the current trend.")
        return 0

    # Consider the last N pullbacks
    recent_pullbacks = pullback_depths[-PULLBACK_SWINGS_LOOKBACK:]
    
    if not recent_pullbacks:
        print("Pullback Check: Not enough recent pullbacks to analyze.")
        return 0

    median_pullback_depth = np.median(recent_pullbacks)
    target_max_depth = 0.6 * STOP_LOSS_PCT # e.g., 0.6 * 0.01 = 0.006 (0.6%)

    print(f"Pullback Check: Median Depth={median_pullback_depth:.4f}, Target Max Depth={target_max_depth:.4f}")

    if median_pullback_depth <= target_max_depth:
        return 1
    else:
        print(f"Pullback Check: Median pullback depth {median_pullback_depth:.4f} exceeds target {target_max_depth:.4f}.")
        return 0

def check_volatility_condition(df_row):
    """
    Checks for moderate volatility.
    - ATR-% of price between 0.25% and 0.9%
    Returns: 1 if condition met, 0 otherwise.
    """
    if pd.isna(df_row['atr_percent']):
        return 0
    
    if 0.25 <= df_row['atr_percent'] <= 0.9:
        return 1
    return 0

def check_choppiness_condition(df_row):
    """
    Checks for low choppiness.
    - Choppiness Index <= 50 (or simply ADX test above)
    Returns: 1 if condition met, 0 otherwise.
    (Using Choppiness Index here as requested, ADX is already in trend)
    """
    if pd.isna(df_row['chop']):
        return 0
        
    if df_row['chop'] <= 50: # Lower CHOP values indicate less choppiness (directional trend)
        return 1
    return 0

def check_liquidity_condition(df_row, df):
    """
    Checks for high liquidity (simplified).
    - Average volume last 30 bars >= session median (using overall median for simplicity)
    Returns: 1 if condition met, 0 otherwise.
    NOTE: Bid-ask spread check is omitted due to data limitations with fetch_ohlcv.
    """
    if 'Volume' not in df.columns or df['Volume'].empty:
        return 0
    
    # Simplified: last 30 bars average volume vs overall median volume in the fetched data
    lookback_period = 30
    if len(df) < lookback_period:
        return 0 # Not enough data for reliable check

    avg_volume_recent = df['Volume'].iloc[-lookback_period:].mean()
    median_volume_session = df['Volume'].median() # Using median of the fetched data as proxy

    if pd.isna(avg_volume_recent) or pd.isna(median_volume_session):
        return 0

    if avg_volume_recent >= median_volume_session:
        return 1
    return 0

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Check market conditions for Double EMA StochOsc strategy.")
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                        help=f"Trading pair symbol (default: {DEFAULT_SYMBOL})")
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                        help=f"Time interval (e.g., '1m', '5m', '1h'; default: {DEFAULT_TIMEFRAME})")
    parser.add_argument('--candles', type=int, default=DATA_FETCH_LIMIT,
                        help=f"Number of recent candles to fetch (default: {DATA_FETCH_LIMIT})")
    
    args = parser.parse_args()

    print(f"Starting market check for {args.symbol} on {args.timeframe} timeframe...")

    # 1. Fetch data
    market_data_df = fetch_recent_bybit_data(args.symbol, args.timeframe, args.candles)

    if market_data_df.empty or len(market_data_df) < max(EMA_SLOW_PERIOD, ADX_WINDOW, ATR_WINDOW, CHOP_WINDOW):
        print("Not enough data to perform checks after fetching or initial indicator calculation. Exiting.")
        return

    # 2. Calculate indicators
    market_data_df = calculate_indicators(market_data_df)

    if market_data_df.empty:
        print("Not enough data after indicator calculation. Exiting.")
        return

    # Get the latest row for checks (current market conditions)
    latest_data_row = market_data_df.iloc[-1]

    # 3. Perform checks
    score = 0
    conditions_met = {}

    print("\n--- Checking Conditions ---")
    
    # Trend
    trend_score = check_trend_condition(latest_data_row)
    score += trend_score
    conditions_met['Trend (EMA_fast/slow, ADX>=25)'] = bool(trend_score)
    print(f"Trend Condition Met: {bool(trend_score)}")
    print(f"  EMA Fast: {latest_data_row.get('ema_fast', 'N/A'):.2f}, EMA Slow: {latest_data_row.get('ema_slow', 'N/A'):.2f}, ADX: {latest_data_row.get('adx', 'N/A'):.2f}")

    # Pullbacks
    pullback_score = check_pullback_condition(market_data_df, latest_data_row) # Pass full df for swing analysis
    score += pullback_score
    conditions_met['Shallow Pullbacks (Median Depth <= 0.6*SL)'] = bool(pullback_score)
    print(f"Shallow Pullbacks Condition Met: {bool(pullback_score)}")
    
    # Volatility
    volatility_score = check_volatility_condition(latest_data_row)
    score += volatility_score
    conditions_met['Moderate Volatility (0.25% <= ATR-% <= 0.9%)'] = bool(volatility_score)
    print(f"Moderate Volatility Condition Met: {bool(volatility_score)}")
    print(f"  ATR-%: {latest_data_row.get('atr_percent', 'N/A'):.2f}%")

    # Choppiness
    choppiness_score = check_choppiness_condition(latest_data_row)
    score += choppiness_score
    conditions_met['Low Choppiness (CHOP <= 50)'] = bool(choppiness_score)
    print(f"Low Choppiness Condition Met: {bool(choppiness_score)}")
    print(f"  Choppiness Index: {latest_data_row.get('chop', 'N/A'):.2f}")

    # Liquidity (optional, simplified)
    # For liquidity, we use the whole dataframe for median calculation
    liquidity_score = check_liquidity_condition(latest_data_row, market_data_df)
    score += liquidity_score # This is an optional filter as per notes, but included in score
    conditions_met['High Liquidity (Avg Vol >= Median Vol)'] = bool(liquidity_score)
    print(f"High Liquidity Condition Met: {bool(liquidity_score)}")
    if 'Volume' in latest_data_row and not market_data_df.empty:
       avg_vol_recent = market_data_df['Volume'].iloc[-30:].mean() if len(market_data_df) >=30 else market_data_df['Volume'].mean()
       median_vol_session = market_data_df['Volume'].median()
       print(f"  Avg Volume (last 30 bars or available): {avg_vol_recent:.2f}, Median Session Volume: {median_vol_session:.2f}")


    # 4. Determine favorability
    print("\n--- Scoring ---")
    print(f"Total Score: {score}")

    if score >= 3:
        print("Market Condition: FAVORABLE for Double EMA StochOsc Strategy")
    else:
        print("Market Condition: UNFAVORABLE for Double EMA StochOsc Strategy")
        
    print("\nDetailed Check Results:")
    for condition, met in conditions_met.items():
        print(f"- {condition}: {'Met' if met else 'Not Met'}")

if __name__ == "__main__":
    main()
