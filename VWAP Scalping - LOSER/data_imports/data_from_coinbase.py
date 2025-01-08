import pandas as pd
import datetime
import os
import ccxt
import data_imports.cdp_api_key as d
from math import ceil
from datetime import timezone  # Import timezone

symbol = 'SOL/USDT'
timeframe = '1m'
weeks = 10

# Function to convert timeframe to seconds
def timeframe_to_sec(timeframe):
    if 'm' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60
    elif 'h' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 60 * 60
    elif 'd' in timeframe:
        return int(''.join([char for char in timeframe if char.isnumeric()])) * 24 * 60 * 60

def get_historical_data(symbol, timeframe, weeks):
    filename = f'{symbol.replace("/", "")}-{timeframe}-{weeks}wks-data.csv'
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col='datetime', parse_dates=True)

    now = datetime.datetime.now(timezone.utc)  # Use timezone-aware datetime
    coinbase = ccxt.coinbase({
        'apiKey': d.name,
        'secret': d.privateKey,
        'enableRateLimit': True,
    })

    granularity = timeframe_to_sec(timeframe)  # Convert timeframe to seconds
    total_time = weeks * 7 * 24 * 60 * 60
    run_times = ceil(total_time / (granularity * 200))

    dataframe = pd.DataFrame()

    for i in range(run_times):
        since = now - datetime.timedelta(seconds=granularity * 200 * (i + 1))
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

    dataframe = dataframe.set_index('datetime')
    dataframe = dataframe[["open", "high", "low", "close", "volume"]]
    dataframe.to_csv(filename)

    return dataframe

print(get_historical_data(symbol, timeframe, weeks))
