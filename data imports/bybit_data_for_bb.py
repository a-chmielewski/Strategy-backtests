import pandas as pd
import datetime
import numpy as np
import os
import ccxt
import bybit_keys as keys  # Ensure this file contains your Bybit API credentials
from math import ceil
from datetime import timezone
import traceback
from collections import defaultdict

def timeframe_to_sec(timeframe):
    """Convert timeframe string to seconds."""
    unit = timeframe[-1]
    num = int(''.join([char for char in timeframe if char.isnumeric()]))
    if unit == 's':
        return num
    elif unit == 'm':
        return num * 60
    elif unit == 'h':
        return num * 60 * 60
    elif unit == 'd':
        return num * 24 * 60 * 60
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")
    
def get_orderbook_metrics(bybit, symbol):
    """
    Fetch basic order book metrics needed for the BB strategy.
    Returns dictionary with order book metrics.
    """
    try:
        orderbook = bybit.fetch_order_book(symbol, limit=25)  # Reduced from 50 to 25 for efficiency
        
        # Get level 1 data only
        bid_price = orderbook['bids'][0][0] if orderbook['bids'] else 0
        ask_price = orderbook['asks'][0][0] if orderbook['asks'] else 0
        bid_vol_l1 = orderbook['bids'][0][1] if orderbook['bids'] else 0
        ask_vol_l1 = orderbook['asks'][0][1] if orderbook['asks'] else 0
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'bid_vol_l1': bid_vol_l1,
            'ask_vol_l1': ask_vol_l1
        }
    except Exception as e:
        print(f"Error fetching orderbook: {e}")
        return defaultdict(float)

def get_bybit_data(symbol, timeframe, start_date, end_date=None):
    """
    Fetch raw historical data from Bybit for BB strategy.
    """
    # Generate filename with date range
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d') if end_date else 'now'
    filename = f'data/bybit-{symbol.replace("/", "")}-{timeframe}-{start_str}-to-{end_str}.csv'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Return existing data if available
    if os.path.exists(filename):
        print(f"Loading existing data from {filename}")
        return pd.read_csv(filename, index_col='datetime', parse_dates=True)
    
    end_date = end_date or datetime.datetime.now(timezone.utc)
    
    # Initialize Bybit exchange with updated configuration
    bybit = ccxt.bybit({
        'apiKey': keys.API_KEY,
        'secret': keys.API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'recvWindow': 60000,  # Increased receive window
            'adjustForTimeDifference': True  # Add this line to automatically adjust timestamps
        }
    })
    
    # Synchronize time with server
    bybit.load_time_difference()
    
    # Calculate number of iterations needed
    granularity = timeframe_to_sec(timeframe)
    total_time = int((end_date - start_date).total_seconds())
    run_times = ceil(total_time / (granularity * 200))
    
    dataframe = pd.DataFrame()
    current_date = end_date

    print(f"Fetching {symbol} data from Bybit...")
    
    for i in range(run_times):
        since = current_date - datetime.timedelta(seconds=granularity * 200)
        since_timestamp = int(since.timestamp()) * 1000

        try:
            # Fetch OHLCV data
            ohlcv_data = bybit.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since_timestamp, 
                limit=2000
            )
            
            if not ohlcv_data:
                print("\nNo more data fetched. Exiting loop.")
                break
                
            # Create DataFrame with basic OHLCV data
            df = pd.DataFrame(
                ohlcv_data, 
                columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )

            # Get basic order book data
            ob_metrics = get_orderbook_metrics(bybit, symbol)
            for metric, value in ob_metrics.items():
                df[metric] = value

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')

            # Get trade history for basic volume analysis
            trades = bybit.fetch_trades(symbol, since=since_timestamp, limit=1000)
            buy_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'buy')
            sell_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'sell')
            
            df['buy_volume'] = buy_volume
            df['sell_volume'] = sell_volume
            
            # Merge with existing data
            dataframe = pd.concat([df, dataframe])
            
            print(f"Progress: {i+1}/{run_times} iterations completed", end='\r')
            
        except Exception as e:
            print(f"\nError: {e}")
            continue

        current_date = since

    # Clean and prepare final dataset
    dataframe = dataframe.set_index('datetime')
    dataframe = dataframe.sort_index()
    dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
    dataframe = dataframe.dropna()
    
    # Save to CSV
    dataframe.to_csv(filename)
    print(f"\nData saved to {filename}")

    return dataframe

def prepare_bb_strategy_data(symbol='BTC/USDT', timeframe='5m', days=30):
    """
    Prepare raw data for the Bollinger Bands strategy.
    """
    end_date = datetime.datetime.now(timezone.utc)
    start_date = end_date - datetime.timedelta(days=days)
    
    # Get historical data
    df = get_bybit_data(symbol, timeframe, start_date, end_date)
    
    # Add basic time features that might be useful for analysis
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    # Verify required columns
    required_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'buy_volume', 'sell_volume', 'bid_price', 'ask_price',
        'bid_vol_l1', 'ask_vol_l1'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

if __name__ == "__main__":
    # Example usage
    symbol = 'ETH/USDT'
    timeframe = '1m'  # Changed to 5m as it's more suitable for BB strategy
    days = 60
    
    try:
        df = prepare_bb_strategy_data(symbol, timeframe, days)
        
        print("\nData Sample Statistics:")
        print(f"Total Records: {len(df)}")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print("\nSample Data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Full error traceback:", traceback.format_exc())
