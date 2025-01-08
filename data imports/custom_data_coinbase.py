import pandas as pd
import datetime
import os
import ccxt
import cdp_api_key as d
from math import ceil
from datetime import timezone  # Import timezone

# Function to convert timeframe to seconds
def timeframe_to_sec(timeframe):
    if 'm' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60
    elif 'h' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60 * 60
    elif 'd' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 24 * 60 * 60

def get_historical_data(symbol, timeframe, start_date, end_date=None):
    """
    Fetch historical data for a specific date range
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'SOL/USDT')
        timeframe (str): Time interval (e.g., '1m', '1h', '1d')
        start_date (datetime): Start date for historical data
        end_date (datetime, optional): End date for historical data. Defaults to current time
    """
    # Generate filename with date range
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d') if end_date else 'now'
    filename = f'{symbol.replace("/", "")}-{timeframe}-{start_str}-to-{end_str}.csv'
    
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col='datetime', parse_dates=True)

    end_date = end_date or datetime.datetime.now(timezone.utc)
    
    # Initialize exchange
    coinbase = ccxt.coinbase({
        'apiKey': d.name,
        'secret': d.privateKey,
        'enableRateLimit': True,
    })

    granularity = timeframe_to_sec(timeframe)
    total_time = int((end_date - start_date).total_seconds())
    run_times = ceil(total_time / (granularity * 200))

    dataframe = pd.DataFrame()
    current_date = end_date

    for i in range(run_times):
        since = current_date - datetime.timedelta(seconds=granularity * 200)
        since_timestamp = int(since.timestamp()) * 1000  # Convert to milliseconds

        try:
            data = coinbase.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=200)
            if not data:
                break
            df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            dataframe = pd.concat([df, dataframe])
        except ccxt.NetworkError as e:
            print(f"Network error: {e}. Retrying...")
            continue
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}. Skipping...")
            continue

        current_date = since

    dataframe = dataframe.set_index('datetime')
    dataframe = dataframe[["open", "high", "low", "close", "volume"]]
    dataframe.to_csv(filename)

    return dataframe

# Example usage:
if __name__ == "__main__":
    symbol = 'BTC/USDT'
    timeframe = '1m'
    
    # Get data from 2023-09-01 to 2023-11-30
    start_date = datetime.datetime(2023, 9, 1, tzinfo=timezone.utc)
    end_date = datetime.datetime(2023, 11, 30, tzinfo=timezone.utc)
    
    df = get_historical_data(symbol, timeframe, start_date, end_date)
    print(df)
