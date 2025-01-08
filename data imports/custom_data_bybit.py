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
    
def get_bybit_data(symbol, timeframe, start_date, end_date=None):
    """
    Fetch historical OHLCV data from Bybit.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Time interval (e.g., '1m', '5m', '1h')
        start_date (datetime): Start date for historical data
        end_date (datetime, optional): End date for historical data
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
    
    # Initialize Bybit exchange
    bybit = ccxt.bybit({
        'apiKey': keys.API_KEY,
        'secret': keys.API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    })

    # Calculate number of iterations needed
    granularity = timeframe_to_sec(timeframe)
    total_time = int((end_date - start_date).total_seconds())
    run_times = ceil(total_time / (granularity * 200))
    
    dataframe = pd.DataFrame()
    current_date = end_date

    print(f"Fetching {symbol} data from Bybit...")
    
    for i in range(run_times):
        since = current_date - datetime.timedelta(seconds=granularity * 200)
        since_timestamp = int(since.timestamp()) * 1000  # Convert to milliseconds

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
                
            # Create DataFrame with OHLCV data
            df = pd.DataFrame(
                ohlcv_data, 
                columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            )
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            
            # Merge with existing data
            dataframe = pd.concat([df, dataframe])
            
            print(f"Progress: {i+1}/{run_times} iterations completed", end='\r')
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Full error traceback:", traceback.format_exc())
            continue

        current_date = since

    # Clean and prepare final dataset
    dataframe = dataframe.set_index('datetime')
    dataframe = dataframe.sort_index()
    
    # Remove duplicates and NaN values
    dataframe = dataframe[~dataframe.index.duplicated(keep='first')]
    dataframe = dataframe.dropna()
    
    # Save to CSV
    dataframe.to_csv(filename)
    print(f"\nData saved to {filename}")

    return dataframe

if __name__ == "__main__":
    # Example usage
    symbol = 'ETH/USDT'
    timeframe = '5m'
    days = 14
    
    try:
        end_date = datetime.datetime.now(timezone.utc)
        start_date = end_date - datetime.timedelta(days=days)

        # end_date = datetime.datetime.strptime("2024-11-28 01:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        # start_date = datetime.datetime.strptime("2024-09-29 01:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        
        
        df = get_bybit_data(symbol, timeframe, start_date, end_date)
        
        print("\nSample Data:")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")
        print("Full error traceback:", traceback.format_exc())
