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
import shutil

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
    
def get_top_usdt_perpetual_futures(exchange, top_n=10):
    """
    Get top N USDT perpetual futures pairs by 24h volume from Bybit.
    
    Args:
        exchange: ccxt exchange instance
        top_n (int): Number of top pairs to return (default: 10)
    
    Returns:
        list: List of symbol names (e.g., ['BTCUSDT', 'ETHUSDT', ...])
    """
    try:
        print(f"Fetching top {top_n} USDT perpetual futures by volume...")
        
        # Fetch all tickers
        tickers = exchange.fetch_tickers()
        print(f"Found {len(tickers)} total tickers")
        
        # Filter for USDT perpetual futures
        usdt_perpetual_pairs = []
        
        for symbol, ticker in tickers.items():
            # Check if it's a USDT pair and has volume data
            if ('USDT' in symbol and 
                ticker.get('quoteVolume') is not None and 
                ticker.get('quoteVolume') > 0):
                
                # Additional check to ensure it's a perpetual futures contract
                market_info = exchange.markets.get(symbol, {})
                if (market_info.get('type') == 'swap' or 
                    market_info.get('linear') == True):
                    
                    # Extract base symbol name (handle formats like ETH/USDT:USDT -> ETHUSDT)
                    clean_symbol = symbol.split(':')[0]  # Remove settlement currency part
                    clean_symbol = clean_symbol.replace('/', '')  # Remove slash
                    
                    # Only keep if it ends with USDT
                    if clean_symbol.endswith('USDT'):
                        usdt_perpetual_pairs.append({
                            'symbol': clean_symbol,
                            'volume': ticker['quoteVolume']
                        })
        
        print(f"Found {len(usdt_perpetual_pairs)} USDT perpetual futures")
        
        # Sort by volume (descending) and get top N
        usdt_perpetual_pairs.sort(key=lambda x: x['volume'], reverse=True)
        top_pairs = [pair['symbol'] for pair in usdt_perpetual_pairs[:top_n]]
        
        print(f"Top {top_n} USDT perpetual futures by volume:")
        for i, pair_data in enumerate(usdt_perpetual_pairs[:top_n], 1):
            print(f"{i}. {pair_data['symbol']} - Volume: ${pair_data['volume']:,.2f}")
        
        print(f"Selected symbols: {top_pairs}")
        
        return top_pairs
        
    except Exception as e:
        print(f"Error fetching top USDT perpetual futures: {e}")
        print("Full error traceback:", traceback.format_exc())
        # Fallback to original hardcoded list
        print("Falling back to hardcoded list...")
        return [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'NXPCUSDT', 'TONUSDT',
            'XRPUSDT', 'MNTUSDT', 'PEPEUSDT', 'DOGEUSDT', 'SUIUSDT'
        ]
    
def get_bybit_data(symbol, timeframe, start_date, end_date=None):
    """
    Fetch historical OHLCV data from Bybit.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        timeframe (str): Time interval (e.g., '1m', '5m', '1h')
        start_date (datetime): Start date for historical data
        end_date (datetime, optional): End date for historical data
    """
    # Generate filename with date range
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d') if end_date else 'now'
    filename = f'data/bybit-{symbol}-{timeframe}-{start_str}-to-{end_str}.csv'
    
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
            'defaultType': 'linear',
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
    # Delete all Bybit data CSV files in the data folder
    data_folder = 'data'
    for filename in os.listdir(data_folder):
        if filename.startswith('bybit-') and filename.endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    # Initialize Bybit exchange for market data
    bybit = ccxt.bybit({
        'apiKey': keys.API_KEY,
        'secret': keys.API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'linear',
        }
    })

    # Get top 10 USDT perpetual futures by volume
    pairs = get_top_usdt_perpetual_futures(bybit, top_n=10)
    timeframes = ['1m', '5m']

    # Date range: last 2 weeks before 12/06/2025 10:00 AM UTC
    end_date = datetime.datetime(2025, 6, 12, 10, 0, 0, tzinfo=timezone.utc)
    start_date = end_date - datetime.timedelta(days=14)

    for symbol in pairs:
        for timeframe in timeframes:
            try:
                print(f"\nFetching {symbol} {timeframe}...")
                df = get_bybit_data(symbol, timeframe, start_date, end_date)
                print(f"Fetched {len(df)} rows for {symbol} {timeframe}")
            except Exception as e:
                print(f"Error fetching {symbol} {timeframe}: {e}")
                print("Full error traceback:", traceback.format_exc())
