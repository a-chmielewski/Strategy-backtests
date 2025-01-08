import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from logging_setup.logger import logger
from config.config import Config
from exchange.exchange_interface import ExchangeInterface
from utils.helpers import timeframe_to_ms

class DataHandler:
    def __init__(self, exchange: ExchangeInterface, config: Config):
        """Initialize DataHandler with exchange interface and config"""
        self.exchange = exchange
        self.config = config
        
        # Get trading parameters from config
        trading_config = config.get('trading')
        self.symbol = trading_config.get('symbol', 'BTC/USDT')
        self.timeframe = trading_config.get('timeframe', '1m')
        
        # Get indicator parameters from config
        indicator_config = config.get('indicators')
        self.vwap_period = indicator_config.get('vwap_period', 20)
        self.volatility_lookback = indicator_config.get('volatility', {}).get('lookback', 100)
        
        # Initialize containers
        self.historical_data: Optional[pd.DataFrame] = None
        self.indicators: Dict[str, np.ndarray] = {}

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical OHLCV data from exchange"""
        try:
            # Convert timeframe to milliseconds for since parameter
            timeframe_ms = timeframe_to_ms(self.timeframe)
            limit = 250  # Default limit
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Add volume validation
            if df['volume'].isnull().any() or (df['volume'] == 0).all():
                logger.warning("Invalid volume data received from exchange")
            
            # Fill any NaN volumes with previous valid values
            df['volume'] = df['volume'].replace(0, np.nan).fillna(method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise


    def calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate VWAP indicator"""
        try:
            # Calculate typical price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            pv = typical_price * df['volume']
            
            # Calculate cumulative values
            cumulative_pv = np.cumsum(pv)
            cumulative_volume = np.cumsum(df['volume'])
            
            # Initialize VWAP array
            vwap = np.empty_like(cumulative_pv)
            vwap[:self.vwap_period-1] = np.nan
            
            # Calculate VWAP using vectorized operations
            vwap[self.vwap_period-1:] = (
                cumulative_pv[self.vwap_period-1:] - 
                np.concatenate(([0], cumulative_pv[:-self.vwap_period]))
            ) / (
                cumulative_volume[self.vwap_period-1:] - 
                np.concatenate(([0], cumulative_volume[:-self.vwap_period]))
            )
            
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            raise

    def verify_data_freshness(self) -> bool:
        """Verify that the data is fresh enough to trade on"""
        try:
            if self.historical_data is None:
                return False
            
            # Get the last candle time (in UTC)
            last_candle_time = self.historical_data.index[-1]
            if last_candle_time.tz is None:
                last_candle_time = last_candle_time.tz_localize('UTC')
            
            # Get current time in UTC
            current_time = pd.Timestamp.now(tz='UTC')
            
            # Calculate maximum allowed age based on timeframe
            timeframe_minutes = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 60
            }
            minutes = timeframe_minutes.get(self.timeframe, 1)
            buffer_minutes = minutes * 0.1
            max_age = pd.Timedelta(minutes=minutes * 2 + buffer_minutes)
            
            # Check if data is fresh
            time_difference = current_time - last_candle_time
            is_fresh = time_difference <= max_age
            
            if not is_fresh:
                logger.warning(
                    f"Data is not fresh. "
                    f"Last candle (UTC): {last_candle_time}, "
                    f"Current (UTC): {current_time}, "
                    f"Difference: {time_difference}, "
                    f"Max allowed: {max_age}"
                )
            
            return is_fresh
            
        except Exception as e:
            logger.error(f"Error verifying data freshness: {str(e)}")
            return False

    def verify_indicator_data(self) -> bool:
        """Verify that all required indicators have valid data"""
        try:
            required_indicators = ['vwap', 'rsi', 'macd', 'volatility', 'atr']
            for indicator in required_indicators:
                if (indicator not in self.indicators or 
                    len(self.indicators[indicator]) < 20 or 
                    np.all(np.isnan(self.indicators[indicator][-20:]))):
                    logger.warning(f"Invalid or missing data for indicator: {indicator}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error verifying indicator data: {str(e)}")
            return False
