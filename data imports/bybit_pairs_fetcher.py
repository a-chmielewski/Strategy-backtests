from pybit.unified_trading import HTTP
import pandas as pd
from datetime import datetime
import time
import os
from bybit_keys import API_KEY, API_SECRET

# Initialize the HTTP client
session = HTTP(
    testnet=False,
    api_key=API_KEY,
    api_secret=API_SECRET
)

def fetch_futures_pairs():
    try:
        # Fetch linear (USDT) perpetual futures
        linear_response = session.get_instruments_info(
            category="linear",
        )
        
        # Fetch inverse perpetual futures
        inverse_response = session.get_instruments_info(
            category="inverse",
        )
        
        # Extract the trading pairs data
        linear_pairs = linear_response['result']['list']
        inverse_pairs = inverse_response['result']['list']
        
        # Combine both types of futures
        all_pairs = linear_pairs + inverse_pairs
        
        # Convert to DataFrame
        df = pd.DataFrame(all_pairs)
        
        # Add timestamp column
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV
        filename = f'F:\Algo Trading TRAINING\Strategy backtests\data\\bybit_futures_pairs_{datetime.now().strftime("%Y%m%d")}.csv'
        df.to_csv(filename, index=False)
        print(f"Successfully saved {len(df)} trading pairs to {filename}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching trading pairs: {str(e)}")
        return None

def fetch_tick_data(symbol, category="linear", limit=100000):
    """
    Fetch recent trades (tick data) for a specific symbol
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        category (str): Category of the instrument ('linear' or 'inverse')
        limit (int): Number of trades to fetch (max 1000)
    """
    try:
        # Fetch recent trades
        response = session.get_public_trade_history(
            category=category,
            symbol=symbol,
            limit=limit
        )
        
        if response and 'result' in response and 'list' in response['result']:
            trades = response['result']['list']
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            
            # Organize columns
            df = df.rename(columns={
                'price': 'price',
                'size': 'quantity',
                'side': 'side',
                'time': 'unix_timestamp'
            })
            
            # Create directory if it doesn't exist
            save_dir = f'F:\Algo Trading TRAINING\Strategy backtests\data\\tick_data\{symbol}'
            os.makedirs(save_dir, exist_ok=True)
            
            # Save to CSV
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{save_dir}\\{symbol}_ticks_{current_time}.csv'
            df.to_csv(filename, index=False)
            print(f"Successfully saved {len(df)} ticks for {symbol} to {filename}")
            
            return df
            
    except Exception as e:
        print(f"Error fetching tick data for {symbol}: {str(e)}")
        return None

def fetch_all_ticks(symbols=None, categories=None):
    """
    Fetch tick data for multiple symbols
    
    Args:
        symbols (list): List of symbols to fetch. If None, fetches all available pairs
        categories (list): List of categories to fetch. If None, uses ['linear', 'inverse']
    """
    if categories is None:
        categories = ['linear', 'inverse']
    
    if symbols is None:
        # Fetch all available pairs first
        pairs_df = fetch_futures_pairs()
        if pairs_df is not None:
            symbols = pairs_df['symbol'].tolist()
    
    for symbol in symbols:
        print(f"Fetching tick data for {symbol}...")
        for category in categories:
            fetch_tick_data(symbol, category)
        time.sleep(0.5)  # Add delay to avoid rate limiting

if __name__ == "__main__":
    # Example usage:
    # 1. Fetch all available pairs
    pairs_df = fetch_futures_pairs()
    
    # 2. Fetch tick data for specific symbols
    # fetch_tick_data('BTCUSDT', 'linear')  # For single symbol
    
    # 3. Fetch tick data for all available pairs
    # fetch_all_ticks()
    
    # 4. Fetch tick data for specific symbols
    # specific_symbols = ['BTCUSDT', 'ETHUSDT']
    # fetch_all_ticks(symbols=specific_symbols, categories=['linear']) 