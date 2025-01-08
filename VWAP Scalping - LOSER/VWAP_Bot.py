import ccxt
import pandas as pd
import numpy as np
import talib
import time
from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import pytz
import re
import math
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)
# Import API credentials
from bybit_keys import API_KEY, API_SECRET, TEST_API_KEY, TEST_API_SECRET

class RefreshingConsoleHandler(logging.StreamHandler):
    """
    A custom logging handler that refreshes multiple lines in the console for certain log messages.
    Each log message intended for refreshing should contain '\n' characters to indicate multiple lines.
    """
    def __init__(self, stream=None):
        super().__init__(stream)
        self.previous_line_count = 0

    def emit(self, record):
        try:
            msg = self.format(record)
            if getattr(record, 'refresh', False):
                # Split message into lines
                lines = msg.split('\n')
                line_count = len(lines)
                
                # Move cursor up by the previous number of lines
                if self.previous_line_count > 0:
                    sys.stdout.write(f'\033[{self.previous_line_count}A')
                
                # Clear each of the previous lines
                for _ in range(self.previous_line_count):
                    sys.stdout.write('\033[K\n')  # Clear line and move to next line
                if self.previous_line_count > 0:
                    # Move cursor up again to overwrite
                    sys.stdout.write(f'\033[{self.previous_line_count}A')
                
                # Write the new lines
                for line in lines:
                    sys.stdout.write(line + '\n')
                sys.stdout.flush()
                
                # Update the previous_line_count
                self.previous_line_count = line_count
            else:
                # Normal log message, append as new line
                sys.stdout.write(msg + '\n')
                sys.stdout.flush()
                self.previous_line_count = 0
        except Exception:
            self.handleError(record)

# ==========================
# Logger Configuration
# ==========================
# Create primary logger
logger = logging.getLogger('bot')
logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs

# File handler for primary logger
file_handler = logging.FileHandler('vwap_bot.log')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Stream handler for primary logger (console)
console_handler = logging.StreamHandler(sys.stdout)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Create refreshed logger
refresh_logger = logging.getLogger('bot.refresh')
refresh_logger.setLevel(logging.INFO)  # Set appropriate level

# Refreshing console handler for refreshed logger
refresh_console_handler = RefreshingConsoleHandler(sys.stdout)
refresh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
refresh_console_handler.setFormatter(refresh_formatter)
refresh_logger.addHandler(refresh_console_handler)

# Prevent the refreshed logger from propagating to the primary logger
refresh_logger.propagate = False

class VWAPScalpingBot:
    def __init__(self, 
                 symbol: str = 'BTC/USDT',
                 timeframe: str = '1m',
                 test_mode: bool = True):
        """
        Initialize the VWAP Scalping Bot
        
        Parameters:
        - symbol: Trading pair
        - timeframe: Candlestick timeframe
        - test_mode: If True, runs on testnet
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.test_mode = test_mode
        
        # Strategy Parameters (matching the backtest)
        self.vwap_period: int = 20
        self.rsi_period: int = 27
        self.macd_fast: int = 9
        self.macd_slow: int = 17
        self.macd_signal: int = 8
        self.bb_period: int = 25
        self.bb_std_dev: float = 2.5
        self.atr_period: int = 20
        self.sar_step: float = 0.03
        self.sar_max: float = 0.221
        self.stop_loss_pct: float = 0.01
        self.take_profit_pct: float = 0.03
        
        # Risk Management Parameters
        self.max_position_size: float = 5000  # Maximum position size in USDT
        self.risk_per_trade: float = 0.01  # 1% risk per trade
        self.trailing_stop_atr_multiplier: float = 2.0
        self.risk_atr_multiplier: float = 1.5
        
        # Trading Schedule Parameters
        self.min_trading_hour: int = 2
        self.max_trading_hour: int = 22
        
        # Add these parameters
        self.rsi_fixed_overbought: float = 70
        self.rsi_fixed_oversold: float = 30
        self.rsi_atr_multiplier: float = 2
        
        # Add volume parameters
        self.min_volume_multiplier: float = 0.5
        self.volume_ma_period: int = 20
        
        # Add volatility parameters
        self.high_volatility_percentile: float = 75.0
        self.volatility_lookback: int = 100
        self.volatility_recent_window: int = 20
        self.volatility_moving_window: int = 100
        self.volatility_threshold_multiplier: float = 2.0
        
        # Initialize exchange connection
        self.initialize_exchange()
        
        # Initialize containers for market data
        self.historical_data: Optional[pd.DataFrame] = None
        self.current_position: Dict = {'size': 0, 'side': None, 'entry_price': None}
        self.indicators: Dict = {}
        
    def initialize_exchange(self):
        """Initialize connection to ByBit exchange"""
        try:
            # Select appropriate API credentials based on test_mode
            api_key = TEST_API_KEY if self.test_mode else API_KEY
            api_secret = TEST_API_SECRET if self.test_mode else API_SECRET
            
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'defaultCategory': 'linear',  
                    'category': 'linear',         
                }
            })
            
            if self.test_mode:
                self.exchange.set_sandbox_mode(True)
                logger.info("Running in testnet mode with test API keys")
            else:
                logger.info("Running in mainnet mode with production API keys")
            
            # Test the connection by fetching account balance
            try:
                balance = self.exchange.fetch_balance({'category': 'linear'})  # Add category parameter
                total_equity = float(balance['info']['result']['list'][0]['totalEquity'])
                logger.info(f"Successfully connected to ByBit {'testnet' if self.test_mode else 'mainnet'}")
                logger.info(f"Account balance fetched successfully. Current balance: {total_equity}")
            except Exception as e:
                logger.error(f"Failed to fetch account balance: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise

            
    def fetch_historical_data(self) -> pd.DataFrame:
        """
        Fetch historical market data with enough periods to calculate all indicators.
        Returns DataFrame with OHLCV data.
        """
        try:
            # Calculate required number of candles
            # We need extra candles for accurate indicator calculation
            # Taking the maximum of all periods and adding buffer
            required_periods = max(
                self.vwap_period,
                self.rsi_period,
                self.macd_slow,  # MACD needs the slow period
                self.bb_period,
                self.atr_period,
                self.volatility_lookback
            )
            
            # Add buffer periods to ensure accurate calculations
            fetch_periods = required_periods + 50 # Extra 100 candles for safety
            
            # Convert timeframe to milliseconds for since parameter
            timeframe_ms = self.timeframe_to_ms(self.timeframe)
            
            # Calculate since timestamp
            since = int(time.time() * 1000) - (fetch_periods * timeframe_ms)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol='BTC/USDT:USDT',
                timeframe=self.timeframe,
                since=since,
                limit=fetch_periods
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            if df.empty or df['volume'].isnull().all():
                logger.warning("Fetched data is empty or contains only NaN volumes.")
                return pd.DataFrame()  # Return empty DataFrame if data is invalid

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp to ensure correct order
            df.sort_index(inplace=True)
            
            # Change log level from info to debug
            logger.debug(f"Successfully fetched {len(df)} candles of historical data")
            
            # Verify data quality
            if len(df) < required_periods:
                logger.warning(f"Fetched data ({len(df)} periods) is less than required ({required_periods} periods)")
                
            # Check for missing periods
            expected_periods = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq=self.timeframe
            )
            missing_periods = expected_periods.difference(df.index)
            if len(missing_periods) > 0:
                logger.warning(f"Found {len(missing_periods)} missing periods in the data")
            
            # Add volume validation
            if df['volume'].isnull().any() or (df['volume'] == 0).all():
                logger.warning("Invalid volume data received from exchange")
                
            # Fill any NaN volumes with previous valid values
            df['volume'] = df['volume'].replace(0, np.nan).fillna(method='ffill')
            
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Network error while fetching data: {str(e)}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error while fetching data: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching data: {str(e)}")
            raise

    def timeframe_to_ms(self, timeframe: str) -> int:
        match = re.match(r'(\d+)([mhd])', timeframe)
        if not match:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
        value, unit = match.groups()
        value = int(value)
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    def calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate VWAP using the same method as in the backtest strategy.
        
        Parameters:
        - df: DataFrame with OHLCV data
        
        Returns:
        - np.ndarray: Array of VWAP values
        """
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
            
            # Calculate VWAP for each point starting from period-1
            # for i in range(self.vwap_period-1, len(df)):
            #     start_idx = max(0, i - self.vwap_period + 1)
            #     vwap[i] = cumulative_pv[i] - cumulative_pv[start_idx]
            #     vol_sum = cumulative_volume[i] - cumulative_volume[start_idx]
            #     vwap[i] = vwap[i] / vol_sum if vol_sum != 0 else np.nan
                
            vwap[self.vwap_period-1:] = (cumulative_pv[self.vwap_period-1:] - np.concatenate(([0], cumulative_pv[:-self.vwap_period]))) / \
            (cumulative_volume[self.vwap_period-1:] - np.concatenate(([0], cumulative_volume[:-self.vwap_period])))    

            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            raise
    
    def calculate_volatility(self, close_prices: np.ndarray, lookback: int) -> np.ndarray:
        returns = pd.Series(close_prices).pct_change()
        if self.timeframe.endswith('m'):
            minutes = int(self.timeframe.rstrip('m'))
        elif self.timeframe.endswith('h'):
            minutes = int(self.timeframe.rstrip('h')) * 60
        elif self.timeframe.endswith('d'):
            minutes = int(self.timeframe.rstrip('d')) * 60 * 24
        else:
            minutes = 1  # Default to 1 minute
        trading_minutes = minutes * 1440  # 1440 minutes in a day
        return returns.rolling(window=lookback, min_periods=1).std() * np.sqrt(trading_minutes)
    
    def calculate_indicators(self):
        """Calculate all technical indicators used in the strategy"""
        try:
            if self.historical_data is None or len(self.historical_data) < self.macd_slow:
                logger.error("Not enough data to calculate indicators")
                return False
                
            df = self.historical_data

            if df.empty or df['volume'].isnull().all():
                logger.warning("Fetched data is empty or contains only NaN volumes.")
                return pd.DataFrame()  # Return empty DataFrame if data is invalid
            
            # Store indicators in a dictionary
            self.indicators = {}
            
            # Calculate VWAP
            self.indicators['vwap'] = self.calculate_vwap(df)
            
            # Calculate RSI
            self.indicators['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            self.indicators['macd'] = macd
            self.indicators['macd_signal'] = macd_signal
            self.indicators['macd_hist'] = macd_hist
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'].values,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std_dev,
                nbdevdn=self.bb_std_dev
            )
            self.indicators['bb_upper'] = bb_upper
            self.indicators['bb_middle'] = bb_middle
            self.indicators['bb_lower'] = bb_lower
            
            # Calculate ATR
            self.indicators['atr'] = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=self.atr_period
            )
            
            # Calculate Parabolic SAR
            self.indicators['sar'] = talib.SAR(
                df['high'].values,
                df['low'].values,
                acceleration=self.sar_step,
                maximum=self.sar_max
            )
            
            # Calculate dynamic RSI levels based on ATR
            self.indicators['rsi_dynamic_overbought'] = np.minimum(
                70 + (self.risk_atr_multiplier * self.indicators['atr']),
                100
            )
            self.indicators['rsi_dynamic_oversold'] = np.maximum(
                30 - (self.risk_atr_multiplier * self.indicators['atr']),
                0
            )
            
            # Calculate Volume MA for volume checks
            self.indicators['volume_ma'] = talib.SMA(
                df['volume'].values,
                timeperiod=self.volume_ma_period
            )
            
            # Add volatility calculation
            self.indicators['volatility'] = self.calculate_volatility(
                self.historical_data['close'].values,
                self.volatility_lookback
            )
            self.indicators['volatility'] = np.array(self.indicators['volatility'])
            # Change log level from info to debug
            logger.debug("Successfully calculated all indicators")
            
            # Verify indicator quality
            self._verify_indicators()
            
            return True
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return False
            
    def _verify_indicators(self):
        """Verify the quality of calculated indicators"""
        try:
            # Get the last values of key indicators
            last_idx = -1
            
            # Check for NaN values in critical indicators
            critical_indicators = ['vwap', 'rsi', 'macd', 'bb_upper', 'atr', 'sar']
            for indicator in critical_indicators:
                if np.isnan(self.indicators[indicator][last_idx]):
                    logger.warning(f"Last value of {indicator} is NaN")
                    
            # Log last values for debugging
            logger.debug(
                f"Last indicator values:\n"
                f"VWAP: {self.indicators['vwap'][last_idx]:.2f}\n"
                f"RSI: {self.indicators['rsi'][last_idx]:.2f}\n"
                f"MACD: {self.indicators['macd'][last_idx]:.2f}\n"
                f"BB Upper: {self.indicators['bb_upper'][last_idx]:.2f}\n"
                f"ATR: {self.indicators['atr'][last_idx]:.2f}\n"
                f"SAR: {self.indicators['sar'][last_idx]:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error verifying indicators: {str(e)}")
    
    def get_position_size(self, current_price: float) -> float:
        """
        Calculate position size in USDT based on risk management rules.
        Adjusted for $100 initial equity and 50x leverage.
        """
        try:
            # Fetch current balance
            balance = self.exchange.fetch_balance()
            equity = float(balance['info']['result']['list'][0]['totalEquity'])
            
            # Risk amount (1% of equity)
            risk_amount = equity * self.risk_per_trade
            
            # Calculate position size in USDT with leverage
            position_size_usdt = risk_amount * 50  # 50x leverage
            
            # Calculate maximum position size based on leverage and equity
            max_position_usdt = equity * 50
            
            # Apply position limits
            position_size_usdt = min(position_size_usdt, max_position_usdt, self.max_position_size * current_price)
            
            # Ensure minimum position size (Bybit requirement)
            MIN_ORDER_SIZE_USDT = 1.0
            position_size_usdt = max(position_size_usdt, MIN_ORDER_SIZE_USDT)
            
            # Round down to 2 decimal places to ensure valid order size
            position_size_usdt = math.floor(position_size_usdt * 100) / 100            
            
            logger.info(f"Calculated position size: {position_size_usdt} USDT (Equity: ${equity:.2f})")
            return position_size_usdt
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def check_entry_conditions(self) -> Tuple[bool, str]:
        """
        Check if entry conditions are met based on the strategy rules.
        Returns (should_enter, signal_type)
        """
        try:
            # First check if we have enough data
            if (self.historical_data is None or 
                len(self.historical_data) < 2 or 
                any(indicator is None for indicator in self.indicators.values())):
                logger.info("Not enough data for entry conditions check")
                return False, ""
            
            # Add volume validation check
            if np.isnan(self.indicators['volume_ma'][-1]) or self.indicators['volume_ma'][-1] == 0:
                logger.info("Volume MA is invalid (NaN or 0)")
                return False, ""

            current_volume = self.historical_data['volume'].iloc[-1]
            if np.isnan(current_volume) or current_volume == 0:
                logger.info("Current volume is invalid (NaN or 0)")
                return False, ""
            
            # Check if any indicator has NaN values at the current position
            for name, indicator in self.indicators.items():
                if isinstance(indicator, pd.Series):
                    if len(indicator) < 2:
                        logger.info(f"Indicator {name} has insufficient data")
                        return False, ""
                    if indicator.iloc[-1] is None or indicator.iloc[-2] is None:
                        logger.info(f"Indicator {name} has None values")
                        return False, ""
                    if np.isnan(indicator.iloc[-1]) or np.isnan(indicator.iloc[-2]):
                        logger.info(f"Indicator {name} has NaN values")
                        return False, ""
                else:  # numpy array
                    if len(indicator) < 2:
                        logger.info(f"Indicator {name} has insufficient data")
                        return False, ""
                    if np.isnan(indicator[-1]) or np.isnan(indicator[-2]):
                        logger.info(f"Indicator {name} has NaN values")
                        return False, ""
        
            # Get current values
            try:
                current_price = self.historical_data['close'].iloc[-1]
                previous_price = self.historical_data['close'].iloc[-2]
                current_volume = self.historical_data['volume'].iloc[-1]
                
                # Get indicator values for current and previous candles
                # Handle both pandas Series and numpy arrays
                def get_value(indicator, idx):
                    if isinstance(indicator, pd.Series):
                        return indicator.iloc[idx]
                    return indicator[idx]
                
                current_vwap = get_value(self.indicators['vwap'], -1)
                previous_vwap = get_value(self.indicators['vwap'], -2)
                current_rsi = get_value(self.indicators['rsi'], -1)
                current_macd = get_value(self.indicators['macd'], -1)
                current_macd_signal = get_value(self.indicators['macd_signal'], -1)
                previous_macd = get_value(self.indicators['macd'], -2)
                previous_macd_signal = get_value(self.indicators['macd_signal'], -2)
                current_bb_upper = get_value(self.indicators['bb_upper'], -1)
                current_bb_lower = get_value(self.indicators['bb_lower'], -1)
                current_sar = get_value(self.indicators['sar'], -1)
                current_volume_ma = get_value(self.indicators['volume_ma'], -1)
                
                # Log current values for debugging
                # logger.info("\nChecking entry conditions:")
                # logger.info(f"Current Price: {current_price:.2f}")
                # logger.info(f"Current VWAP: {current_vwap:.2f}")
                # logger.info(f"Current RSI: {current_rsi:.2f}")
                # logger.info(f"Current MACD: {current_macd:.4f}, Signal: {current_macd_signal:.4f}")
                # logger.info(f"Current BB Upper: {current_bb_upper:.2f}, Lower: {current_bb_lower:.2f}")
                # logger.info(f"Current SAR: {current_sar:.2f}")
                # logger.info(f"Volume: {current_volume:.2f}, MA: {current_volume_ma:.2f}")
                
                # Check volume condition first
                min_volume_threshold = current_volume_ma * self.min_volume_multiplier
                if current_volume < max(min_volume_threshold, 0.001):
                    refresh_logger.info(f"[X] Volume too low: {current_volume:.2f} < {min_volume_threshold:.2f}")
                    return False, ""
                else:
                    refresh_logger.info("[+] Volume condition met")
                
                # Check long conditions with detailed logging
                long_conditions = {
                    "VWAP crossover": (previous_price < previous_vwap and current_price > current_vwap),
                    "RSI not overbought": (current_rsi < self.indicators['rsi_dynamic_overbought'][-1]),
                    "MACD bullish crossover": (previous_macd < previous_macd_signal and current_macd > current_macd_signal),
                    "Price within BB": (current_price > current_bb_lower and current_price < current_bb_upper),
                    "SAR confirms uptrend": (current_sar < current_price)
                }
                
                # Check short conditions with detailed logging
                short_conditions = {
                    "VWAP crossover": (previous_price > previous_vwap and current_price < current_vwap),
                    "RSI not oversold": (current_rsi > self.indicators['rsi_dynamic_oversold'][-1]),
                    "MACD bearish crossover": (previous_macd > previous_macd_signal and current_macd < current_macd_signal),
                    "Price within BB": (current_price > current_bb_lower and current_price < current_bb_upper),
                    "SAR confirms downtrend": (current_sar > current_price)
                }
                
                # Check and log long conditions
                refresh_logger.info("\nLong conditions check:")
                all_long_conditions_met = True
                for condition_name, condition_met in long_conditions.items():
                    if condition_met:
                        refresh_logger.info(f"[+] {condition_name}")
                    else:
                        refresh_logger.info(f"[X] {condition_name}")
                        all_long_conditions_met = False
                
                # Check and log short conditions
                refresh_logger.info("\nShort conditions check:")
                all_short_conditions_met = True
                for condition_name, condition_met in short_conditions.items():
                    if condition_met:
                        refresh_logger.info(f"[+] {condition_name}")
                    else:
                        refresh_logger.info(f"[X] {condition_name}")
                        all_short_conditions_met = False
                
                # Final decision
                if all_long_conditions_met:
                    logger.info("[+] All long conditions met - Opening long position")
                    return True, "long"
                elif all_short_conditions_met:
                    logger.info("[+] All short conditions met - Opening short position")
                    return True, "short"
                else:
                    logger.debug("[X] Not all conditions met for either long or short entry")
                    return False, ""
                
            except IndexError as e:
                logger.error(f"Index error while accessing indicator values: {str(e)}")
                return False, ""
            
        except Exception as e:
            logger.error(f"Error checking entry conditions: {str(e)}")
            logger.error(f"Full error traceback:", exc_info=True)
            return False, ""
    
    def check_exit_conditions(self, dynamic_stop_loss: float = None, dynamic_take_profit: float = None) -> Tuple[bool, str]:
        """
        Check if exit conditions are met using dynamic risk parameters
        """
        try:
            if not self.current_position['size'] or not self.current_position['side']:
                return False, ""
            
            current_price = self.historical_data['close'].iloc[-1]
            entry_price = self.current_position['entry_price']
            
            # Calculate dynamic parameters if not provided
            # if dynamic_stop_loss is None:
            #     dynamic_stop_loss, dynamic_take_profit = self.get_dynamic_risk_parameters(
            #         current_price,
            #         self.indicators['atr'][-1]
            #     )
            
            # Calculate current profit/loss
            if self.current_position['side'] == 'long':
                pnl = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) / entry_price
            
            # Check dynamic stop loss
            if pnl < -dynamic_stop_loss:
                return True, f"dynamic_stop_loss_{dynamic_stop_loss:.4f}"
            
            # Check dynamic take profit
            if pnl > dynamic_take_profit:
                return True, f"dynamic_take_profit_{dynamic_take_profit:.4f}"
            
            # Check trailing stop
            if self.current_position['side'] == 'long':
                if current_price < self.trailing_stop:
                    return True, "trailing_stop"
            else:
                if current_price > self.trailing_stop:
                    return True, "trailing_stop"
                
            # Check reversal signals with dynamic parameters
            current_vwap = self.indicators['vwap'][-1]
            if self.current_position['side'] == 'long' and current_price < current_vwap:
                # Add additional confirmation for reversal with dynamic parameters
                if pnl > dynamic_stop_loss * 0.5:  # Only exit on reversal if in profit
                    return True, "reversal_with_profit"
            elif self.current_position['side'] == 'short' and current_price > current_vwap:
                if pnl > dynamic_stop_loss * 0.5:
                    return True, "reversal_with_profit"
                
            return False, ""
        
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            return False, ""
    
    def determine_order_type(self, side: str, current_price: float) -> Tuple[str, float, dict]:
        """
        Determine the best order type based on market conditions.
        Returns (order_type, limit_price, additional_params)
        """
        try:
            # Check if we have valid volatility data
            if 'volatility' not in self.indicators or len(self.indicators['volatility']) == 0:
                logger.warning("Insufficient volatility data, defaulting to market order")
                return 'market', None, {'category': 'linear', 'position_idx': 0}
            logger.info(f"Determining order type for side: {side}, current_price: {current_price}")
            
            # Get current market conditions
            # Check if volatility is available before using
            if 'volatility' in self.indicators and len(self.indicators['volatility']) > 0:
                current_volatility = self.indicators['volatility'][-1]
            else:
                current_volatility = np.nan  # Default if unavailable
                logger.warning("Volatility data is not available.")
        
            current_spread = self.get_market_spread()
            avg_volatility = np.nanmean(self.indicators['volatility'][-20:])
            logger.debug(f"Current Volatility: {current_volatility}, Average Volatility: {avg_volatility}")
        
            if np.isnan(current_volatility) or np.isnan(avg_volatility):
                logger.warning("NaN values in volatility data, defaulting to market order")
                return 'market', None, {'category': 'linear', 'position_idx': 0}
            
            # Get order book to check liquidity
            order_book = self.exchange.fetch_order_book('BTC/USDT:USDT')
            bid = order_book['bids'][0][0] if order_book['bids'] else current_price
            ask = order_book['asks'][0][0] if order_book['asks'] else current_price
            
            # Calculate spread percentage
            spread_pct = (ask - bid) / current_price if current_price != 0 else 0.0
            logger.debug(f"Spread Percentage: {spread_pct}")
        
            # Base parameters
            params = {
                'category': 'linear',
                'position_idx': 0
            }
            
            # High volatility or wide spread conditions
            if current_volatility > avg_volatility * 1.5 or spread_pct > 0.001:
                logger.info(f"Using market order due to high volatility/spread")
                return 'market', None, params
            
            # Normal market conditions - use post-only limit orders
            if side == 'long':
                limit_price = bid - (current_price * 0.001)  # 0.01% above bid
                params['postOnly'] = True
            else:
                limit_price = ask + (current_price * 0.001)  # 0.01% below ask
                params['postOnly'] = True
            
            # For tight spreads, use IOC instead of post-only
            if spread_pct < 0.001:  # 0.05% spread threshold
                params.pop('postOnly', None)  # Remove postOnly if it exists
                params['timeInForce'] = 'IOC'
                
            logger.info(f"Using limit order: Price: {limit_price:.2f}")
            return 'limit', limit_price, params
            
        except Exception as e:
            logger.error(f"Error determining order type: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return 'market', None, {'category': 'linear', 'position_idx': 0}

    def get_market_spread(self) -> float:
        """Calculate current market spread"""
        try:
            ticker = self.exchange.fetch_ticker('BTC/USDT:USDT')
            return (ticker['ask'] - ticker['bid']) / ticker['last']
        except Exception as e:
            logger.error(f"Error calculating market spread: {str(e)}")
            return float('inf')

    def place_order(self, side: str, size_usdt: float, order_type: str = None):
        """Modified place_order method with dynamic order type selection and graceful leverage handling"""
        try:
            # Check minimum order size (Bybit requirement)
            MIN_ORDER_SIZE_USDT = 1.0
            if size_usdt < MIN_ORDER_SIZE_USDT:
                logger.warning(f"Order size ({size_usdt} USDT) is below minimum requirement of {MIN_ORDER_SIZE_USDT} USDT")
                return False
            
            if size_usdt <= 0:
                logger.warning("Invalid position size, skipping order")
                return False
            
            # Round size to ensure it meets Bybit's requirements
            size_usdt = math.floor(size_usdt * 100) / 100  # Round down to 2 decimal places
            
            current_price = self.historical_data['close'].iloc[-1]
            # Convert USDT to BTC
            size_btc = round(size_usdt / current_price, 3)
            
            current_atr = self.indicators['atr'][-1]
            
            # Log the order attempt
            logger.info(f"Attempting to place {side} order: Size={size_btc} BTC at price {current_price:.2f}")
            
            # Determine best order type based on market conditions
            determined_type, limit_price, _ = self.determine_order_type(side, current_price)
            order_type = order_type or determined_type  # Use provided type or determined type
            
            # Get dynamic risk parameters
            stop_loss_pct, take_profit_pct = self.get_dynamic_risk_parameters(current_price, current_atr)

            # Prepare leverage parameters correctly
            leverage_symbol = 'BTC/USDT:USDT'
            params_leverage = {
                'category': 'linear',
            }
            
            try:
                # **Refactored Leverage Setting Logic**
                leverage_response = self.exchange.fetch_leverage(leverage_symbol)
                current_leverage = leverage_response['info']['info']['leverage']  # or shortLeverage, they're the same
                
                logger.debug(f"Current leverage for {self.symbol}: {current_leverage}")
                
                if current_leverage is None:
                    logger.error(f"Unable to fetch current leverage for {self.symbol}. Proceeding to set leverage to 50x.")
                    # If leverage information is unavailable, attempt to set it
                    self.exchange.set_leverage(50, 'BTC/USDT:USDT', params={'category': 'linear'})
                    logger.info(f"Leverage set to 50x for {self.symbol}")
                elif current_leverage != '50':
                    # Only set leverage if it's not already 50x
                    logger.info(f"Current leverage ({current_leverage}x) is different from desired leverage (50x). Updating leverage.")
                    self.exchange.set_leverage(50, 'BTC/USDT:USDT', params={'category': 'linear'})
                    logger.info(f"Leverage updated to 50x for {self.symbol}")
                else:
                    logger.info(f"Leverage is already set to 50x for {self.symbol}. No changes made.")
                
                # Place the main order with appropriate type
                order_params = {
                    'timeInForce': 'GTC',
                    'mmp': False,  # Market Maker Protection
                    'category': 'linear',
                }

                if order_type == 'limit':
                    if side == 'long':
                        limit_price = current_price * 1.001  # Place slightly above market
                    else:
                        limit_price = current_price * 0.999  # Place slightly below market
                    order_params['postOnly'] = True
                    main_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='limit',
                        side='buy' if side == 'long' else 'sell',
                        amount=size_btc,
                        price=limit_price,
                        params=order_params
                    )
                else:  # 'market' order
                    main_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='market',
                        side='buy' if side == 'long' else 'sell',
                        amount=size_btc,
                        params=order_params
                    )
                
                # logger.info(f"Main order placed: {main_order}")
                # Wait for the order to be filled with timeout
                filled_order = self.wait_for_fill(main_order['id'], timeout_seconds=30)
                if not filled_order:
                    logger.warning("Order not filled within timeout, cancelling")
                    self.exchange.cancel_all_orders('BTC/USDT:USDT')
                    return False

                # Get the actual fill price
                fill_price = float(filled_order['average'])

                # Calculate stop loss and take profit prices
                if side == 'long':
                    stop_loss_price = fill_price * (1 - stop_loss_pct)
                    take_profit_price = fill_price * (1 + take_profit_pct)
                else:
                    stop_loss_price = fill_price * (1 + stop_loss_pct)
                    take_profit_price = fill_price * (1 - take_profit_pct)

                # Round prices to market precision
                stop_loss_price = float(self.exchange.price_to_precision('BTC/USDT:USDT', stop_loss_price))
                take_profit_price = float(self.exchange.price_to_precision('BTC/USDT:USDT', take_profit_price))

                # Place stop loss order
                if side == 'long':
                    # For long position, stop-loss is a sell order triggered when price falls to stop_loss_price
                    sl_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='market',
                        side='sell',
                        amount=size_btc,
                        price=None,
                        params={
                            'stopPrice': stop_loss_price,
                            'triggerPrice': stop_loss_price,
                            'triggerDirection': 2,  # Price falls to stop_price
                            'reduceOnly': True,
                            'timeInForce': 'GTC',
                            'closeOnTrigger': True,
                            'category': 'linear',
                        }
                    )
                else:
                    # For short position, stop-loss is a buy order triggered when price rises to stop_loss_price
                    sl_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='market',
                        side='buy',
                        amount=size_btc,
                        price=None,
                        params={
                            'stopPrice': stop_loss_price,
                            'triggerPrice': stop_loss_price,
                            'triggerDirection': 1,  # Price rises to stop_price
                            'reduceOnly': True,
                            'timeInForce': 'GTC',
                            'closeOnTrigger': True,
                            'category': 'linear',
                        }
                    )

                # Place take profit order
                if side == 'long':
                    # For long position, take-profit is a sell order triggered when price rises to take_profit_price
                    tp_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='market',
                        side='sell',
                        amount=size_btc,
                        price=None,
                        params={
                            'stopPrice': take_profit_price,
                            'triggerPrice': take_profit_price,
                            'triggerDirection': 1,  # Price rises to take_profit_price
                            'reduceOnly': True,
                            'timeInForce': 'GTC',
                            'closeOnTrigger': True,
                            'category': 'linear',
                        }
                    )
                else:
                    # For short position, take-profit is a buy order triggered when price falls to take_profit_price
                    tp_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='market',
                        side='buy',
                        amount=size_btc,
                        price=None,
                        params={
                            'stopPrice': take_profit_price,
                            'triggerPrice': take_profit_price,
                            'triggerDirection': 2,  # Price falls to take_profit_price
                            'reduceOnly': True,
                            'timeInForce': 'GTC',
                            'closeOnTrigger': True,
                            'category': 'linear',
                        }
                    )

                # Update current position information
                self.current_position = {
                    'size': size_btc,
                    'side': side,
                    'entry_price': fill_price,
                    'stop_loss_order_id': sl_order['id'],
                    'take_profit_order_id': tp_order['id'],
                    'entry_time': pd.Timestamp.now(tz='UTC')
                }

                logger.info(f"Successfully placed {side} position: "
                            f"Size={size_btc}, Entry={fill_price:.2f}, "
                            f"SL={stop_loss_price}, TP={take_profit_price}")

                return True
            
            except Exception as e:
                logger.error(f"Error placing orders: {str(e)}")
                logger.error("Traceback:", exc_info=True)
                # Try to cancel any orders that were placed
                self.exchange.cancel_all_orders('BTC/USDT:USDT')
                return False
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order in place_order: {str(e)}")
            logger.error("Traceback:", exc_info=True)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error in place_order: {str(e)}")
            logger.error("Traceback:", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in place_order: {str(e)}")
            logger.error("Traceback:", exc_info=True)

    
    def update_trailing_stop(self, dynamic_stop_loss: float = None):
        """
        Update trailing stop based on ATR and dynamic risk parameters
        """
        try:
            if not self.current_position['size']:
                return
            
            current_price = self.historical_data['close'].iloc[-1]
            current_atr = self.indicators['atr'][-1]
            
            # Use provided dynamic stop loss or calculate new one
            if dynamic_stop_loss is None:
                dynamic_stop_loss, _ = self.get_dynamic_risk_parameters(current_price, current_atr)
            
            # Calculate minimum stop distance using dynamic parameters
            min_stop_distance = current_price * dynamic_stop_loss
            
            # Initialize trailing stop if it doesn't exist
            if not hasattr(self, 'trailing_stop') or self.trailing_stop is None:
                if self.current_position['side'] == 'long':
                    self.trailing_stop = current_price - max(
                        min_stop_distance,
                        current_atr * self.trailing_stop_atr_multiplier
                    )
                else:
                    self.trailing_stop = current_price + max(
                        min_stop_distance,
                        current_atr * self.trailing_stop_atr_multiplier
                    )
                logger.info(f"Initialized trailing stop at: {self.trailing_stop:.2f}")
                return
            
            if self.current_position['side'] == 'long':
                # For long positions, calculate new stop with dynamic parameters
                new_stop = current_price - max(
                    min_stop_distance,
                    current_atr * self.trailing_stop_atr_multiplier
                )
                # Use max to ensure stop only moves up
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    refresh_logger.info(
                        f"Updated long trailing stop up to: {self.trailing_stop:.2f} "
                        f"(Price: {current_price:.2f}, Dynamic SL: {dynamic_stop_loss:.4f})"
                    )
            else:  # short position
                # For short positions, calculate new stop with dynamic parameters
                new_stop = current_price + max(
                    min_stop_distance,
                    current_atr * self.trailing_stop_atr_multiplier
                )
                # Use min to ensure stop only moves down
                if new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                    refresh_logger.info(
                        f"Updated short trailing stop down to: {self.trailing_stop:.2f} "
                        f"(Price: {current_price:.2f}, Dynamic SL: {dynamic_stop_loss:.4f})"
                    )
            
            # Calculate and log distance to trailing stop with dynamic parameters
            stop_distance = abs(current_price - self.trailing_stop)
            stop_distance_pct = (stop_distance / current_price) * 100
            refresh_logger.info(
                f"Trailing stop distance: {stop_distance:.2f} ({stop_distance_pct:.2f}%) "
                f"Dynamic SL: {dynamic_stop_loss:.4f}"
            )
                
        except Exception as e:
            logger.error(f"Error updating trailing stop: {str(e)}")
    
    def sync_position(self):
        """
        Sync the current position with the exchange.
        This ensures our position tracking stays accurate even if orders are filled directly on the exchange.
        """
        try:
            # Fetch positions from exchange with correct category
            positions = self.exchange.fetch_positions(['BTC/USDT:USDT'])

            # Find our symbol's position
            current_position = next((pos for pos in positions if pos['symbol'] == 'BTC/USDT:USDT'), None)
            if current_position is None or float(current_position['info']['size']) == 0:
                # No position exists
                if self.current_position['size'] != 0:
                    logger.warning("Local position tracking was out of sync with exchange. Resetting position.")
                self.current_position = {'size': 0, 'side': None, 'entry_price': None}
                self.trailing_stop = None
                return
                
            # Position exists on exchange
            position_size = abs(float(current_position['info']['size']))
            position_side = 'long' if current_position['info']['side'] == 'Buy' else 'short'
            entry_price = float(current_position['entryPrice'])
            
            # Convert createdTime to UTC timestamp
            created_time = pd.Timestamp(int(current_position['info']['createdTime']), unit='ms', tz='UTC')
            
            # Check if our local tracking matches
            if (self.current_position['size'] != position_size or 
                self.current_position['side'] != position_side or 
                self.current_position['entry_price'] != entry_price):
                
                logger.warning(
                    f"Position sync mismatch - Local: {self.current_position}, "
                    f"Exchange: Size={position_size}, Side={position_side}, Entry={entry_price}"
                )
                
                # Update local tracking
                self.current_position = {
                    'size': position_size,
                    'side': position_side,
                    'entry_price': entry_price,
                    'entry_time': created_time  # Use exchange's creation time
                }
                
                # Get current stop loss and take profit if they exist
                if current_position['info']['stopLoss']:
                    self.current_position['stop_loss'] = float(current_position['info']['stopLoss'])
                if current_position['info']['takeProfit']:
                    self.current_position['take_profit'] = float(current_position['info']['takeProfit'])
                
                # Recalculate trailing stop based on current position
                current_price = self.historical_data['close'].iloc[-1]
                current_atr = self.indicators['atr'][-1]
                
                if position_side == 'long':
                    self.trailing_stop = current_price - (current_atr * self.trailing_stop_atr_multiplier)
                else:
                    self.trailing_stop = current_price + (current_atr * self.trailing_stop_atr_multiplier)
                    
                logger.info(f"Position synced with exchange: {self.current_position}")
                
        except Exception as e:
            logger.error(f"Error syncing position with exchange: {str(e)}")
    
    def manage_open_positions(self):
        """Manage existing positions with dynamic risk parameters"""
        try:
            # Sync position with exchange first
            self.sync_position()
            
            if not self.current_position['size']:
                return
            
            current_price = self.historical_data['close'].iloc[-1]
            current_atr = self.indicators['atr'][-1]
            
            # Get dynamic risk parameters for current market conditions
            dynamic_stop_loss, dynamic_take_profit = self.get_dynamic_risk_parameters(
                current_price, 
                current_atr
            )
            
            # Update trailing stop with dynamic parameters
            self.update_trailing_stop(dynamic_stop_loss)
            
            # Check exit conditions with dynamic parameters
            should_exit, exit_reason = self.check_exit_conditions(
                dynamic_stop_loss,
                dynamic_take_profit
            )
            
            if should_exit:
                try:
                    # Cancel existing stop loss and take profit orders
                    self.exchange.cancel_all_orders('BTC/USDT:USDT')
                    
                    # Close position with market order
                    close_side = 'sell' if self.current_position['side'] == 'long' else 'buy'
                    
                    close_order = self.exchange.create_order(
                        symbol='BTC/USDT:USDT',
                        type='market',
                        side=close_side,
                        amount=self.current_position['size'],
                        params={'reduceOnly': True}
                    )
                    
                    # Calculate P&L
                    entry_price = self.current_position['entry_price']
                    if close_order['average'] is not None:
                        exit_price = float(close_order['average'])
                        pnl_pct = ((exit_price - entry_price) / entry_price * 100 * 
                              (1 if self.current_position['side'] == 'long' else -1))
                        logger.info(
                            f"Position closed - Reason: {exit_reason}, "
                            f"Entry: {entry_price:.2f}, Exit: {exit_price:.2f}, "
                            f"P&L: {pnl_pct:.2f}%, "
                            f"Dynamic SL: {dynamic_stop_loss:.4f}, TP: {dynamic_take_profit:.4f}"
                        )
                    else:
                        logger.warning("Can't calculate P&L because average is None")
                        pnl_pct = 0

                    
                    # Reset position information
                    self.current_position = {'size': 0, 'side': None, 'entry_price': None}
                    self.trailing_stop = None
                    
                except Exception as e:
                    logger.error(f"Error closing position: {str(e)}")
                    logger.error("Traceback:", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error in manage_open_positions: {str(e)}")
    
    def cancel_all_orders(self):
        """Cancel all open orders for the trading pair"""
        try:
            open_orders = self.exchange.fetch_open_orders('BTC/USDT:USDT')
            for order in open_orders:
                try:
                    self.exchange.cancel_order(order['info']['orderId'], 'BTC/USDT:USDT')
                    logger.info(f"Cancelled order {order['info']['orderId']}")
                except Exception as e:
                    logger.error(f"Error cancelling order {order['info']['orderId']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching open orders: {str(e)}")
    
    # def is_valid_trading_time(self) -> bool:
    #     """
    #     Check if current time is within valid trading hours (UTC).
    #     """
    #     try:
    #         # Get current UTC time
    #         current_time = pd.Timestamp.now(tz='UTC')
            
    #         # Check trading hours (using UTC hours)
    #         current_hour = current_time.hour
            
    #         # Correct the hour comparison logic
    #         if self.min_trading_hour <= current_hour < self.max_trading_hour:
    #             logger.debug(
    #                 f"Within trading hours. Current hour (UTC): {current_hour}, "
    #                 f"Valid hours: {self.min_trading_hour}:00-{self.max_trading_hour}:00 UTC"
    #             )
    #             return True
    #         else:
    #             logger.info(
    #                 f"Outside trading hours. Current hour (UTC): {current_hour}, "
    #                 f"Valid hours: {self.min_trading_hour}:00-{self.max_trading_hour}:00 UTC"
    #             )
    #             return False
    #     except Exception as e:
    #         logger.error(f"Error checking trading time: {str(e)}")
    #         return False
    
    def is_market_volatile(self) -> bool:
        """
        Check if the market is too volatile to trade.
        Returns True if volatility is too high, False otherwise.
        """
        try:
            # Ensure sufficient data is available
            if self.historical_data is not None and len(self.historical_data) > self.volatility_lookback:
                
                # Calculate percentage changes
                pct_changes = self.historical_data['close'].pct_change().dropna()
                
                # Recent volatility: Standard deviation over the last 20 periods
                recent_volatility = pct_changes[-self.volatility_recent_window:]
                recent_std = np.std(recent_volatility)
                
                # Moving volatility: Standard deviation over the last 100 periods
                moving_volatility = np.std(pct_changes[-self.volatility_moving_window:])
                
                logger.debug(
                    f"Recent Volatility (20 periods): {recent_std:.6f}, "
                    f"Moving Volatility (100 periods): {moving_volatility:.6f}"
                )
                
                # Define a threshold multiplier
                threshold_multiplier = 3.0
                
                # Determine if recent volatility exceeds the threshold
                if moving_volatility > 0 and recent_std > moving_volatility * threshold_multiplier:
                    logger.debug(
                        f"Market volatility too high. "
                        f"Recent Std: {recent_std:.6f}, "
                        f"Moving Std: {moving_volatility:.6f}"
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking market volatility: {str(e)}")
            return False  # Default to not volatile to avoid unnecessary skipping
    
    def run(self):
        """Main bot loop"""
        logger.info("Starting VWAP Scalping Bot...")
        retry_delay = 5
        max_delay = 300
        
        try:
            # Initial position sync
            self.sync_position()
            logger.info("Initial position sync completed")
            
            # Log initial setup information
            logger.info(f"Trading {self.symbol} on {'testnet' if self.test_mode else 'mainnet'}")
            logger.info(f"Timeframe: {self.timeframe}")
            logger.info("Bot initialization completed successfully")
            
            while True:
                try:
                    current_time = pd.Timestamp.now(tz='UTC')
                    
                    # Fetch latest market data
                    self.historical_data = self.fetch_historical_data()
                    
                    # Verify data freshness
                    if not self.is_data_fresh():
                        logger.warning("Data is not fresh, skipping this iteration")
                        time.sleep(5)
                        continue
                    
                    # Calculate indicators
                    if not self.calculate_indicators():
                        logger.warning("Failed to calculate indicators, skipping this iteration")
                        time.sleep(5)
                        continue
                    
                    # Verify that we have valid indicator data
                    if not self.verify_indicator_data():
                        logger.warning("Invalid indicator data, skipping this iteration")
                        time.sleep(5)
                        continue
                    
                    # Check if we're in valid trading hours
                    # if not self.is_valid_trading_time():
                    #     logger.info(
                    #         f"Outside trading hours. Current hour (UTC): {current_time.hour}, "
                    #         f"Valid hours: {self.min_trading_hour}:00-{self.max_trading_hour}:00 UTC"
                    #     )
                    #     time.sleep(60)
                    #     continue
                    
                    # Check market volatility
                    if self.is_market_volatile():
                        logger.info("Market is too volatile, skipping this iteration")
                        time.sleep(60)
                        continue
                    
                    # Manage existing positions (includes position sync)
                    self.manage_open_positions()
                    
                    # Check for new trade opportunities
                    if self.current_position['size'] == 0:
                        logger.debug("Looking for entry opportunities...")
                        should_enter, signal = self.check_entry_conditions()
                        if should_enter:
                            size = self.get_position_size(self.historical_data['close'].iloc[-1])
                            self.place_order(signal, size)
                        else:
                            logger.debug("No entry conditions met, waiting for next opportunity.")
                    
                    time.sleep(5)  # Avoid hitting rate limits

                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"Rate limit exceeded: {str(e)}. Retrying after delay.")
                    time.sleep(retry_delay)
                    retry_delay = min(max_delay, retry_delay * 2)  # Exponential backoff
                except ccxt.NetworkError as e:
                    logger.warning(f"Network error: {str(e)}. Retrying after delay.")
                    # logger.error("Traceback:", exc_info=True)
                    time.sleep(retry_delay)
                    retry_delay = min(max_delay, retry_delay * 2)
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay = min(max_delay, retry_delay * 2)
                    
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            # Clean up any open orders before exiting
            self.exchange.cancel_all_orders('BTC/USDT:USDT')

    def is_data_fresh(self) -> bool:
        """
        Check if the current data is fresh enough to trade on.
        Handles timezone conversion between UTC (exchange) and Europe/Paris (local).
        """
        if self.historical_data is None:
            return False
        
        # Get the last candle time (in UTC)
        last_candle_time = self.historical_data.index[-1]
        if last_candle_time.tz is None:
            last_candle_time = last_candle_time.tz_localize('UTC')
        
        # Get current time in UTC
        current_time = pd.Timestamp.now(tz='UTC')
        
        # Calculate maximum allowed age based on timeframe with buffer
        timeframe_minutes = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '1h': 60
        }
        minutes = timeframe_minutes.get(self.timeframe, 1)
        # Add a small buffer (10% of timeframe) to prevent edge case warnings
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
        else:
            logger.debug(
                f"Data is fresh. "
                f"Time difference: {time_difference}, "
                f"Max allowed: {max_age}"
            )
        
        return is_fresh

    def get_dynamic_risk_parameters(self, current_price: float, current_atr: float) -> Tuple[float, float]:
        """
        Calculate dynamic risk parameters based on market volatility
        """
        try:
            # Check if we have valid volatility data
            if 'volatility' not in self.indicators or len(self.indicators['volatility']) < self.volatility_lookback:
                logger.warning("Insufficient volatility data for dynamic risk calculation, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            refresh_logger.info(f"Calculating dynamic risk parameters for price: {current_price}, ATR: {current_atr}")
        
            recent_volatility = self.indicators['volatility'][-self.volatility_lookback:]
            
            # Check for valid data
            if np.all(np.isnan(recent_volatility)):
                logger.warning("All NaN values in volatility data, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            # Convert recent_volatility to a NumPy array if it's not already
            recent_volatility = np.array(recent_volatility)
            
            # Calculate volatility factor with additional checks
            mean_recent_volatility = np.nanmean(recent_volatility)
            if mean_recent_volatility == 0 or np.isnan(mean_recent_volatility):
                logger.warning("Invalid mean volatility, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            current_vol = self.indicators['volatility'][-1]
            if np.isnan(current_vol):
                logger.warning("Current volatility is NaN, using default values")
                return self.stop_loss_pct, self.take_profit_pct
            
            volatility_factor = (self.indicators['volatility'][-1] / mean_recent_volatility) if mean_recent_volatility != 0 else 1.0
            logger.debug(f"Mean Recent Volatility: {mean_recent_volatility}, Current Volatility: {current_vol}, Volatility Factor: {volatility_factor}")
        
            # Adjust stop loss and take profit based on volatility
            dynamic_stop_loss = max(
                self.stop_loss_pct,
                self.stop_loss_pct * volatility_factor
            )
            dynamic_take_profit = max(
                self.take_profit_pct,
                self.take_profit_pct * volatility_factor
            )
            
            # Use ATR for minimum stop distance
            min_stop_distance = current_atr * self.risk_atr_multiplier
            price_based_stop = min_stop_distance / current_price
            
            # Use the larger of the two stops
            final_stop_loss = max(dynamic_stop_loss, price_based_stop)
            final_take_profit = max(dynamic_take_profit, price_based_stop * 2)

            # Ensure that stop loss and take profit are within logical bounds
            final_stop_loss = min(final_stop_loss, 1.0)  # Stop loss cannot exceed 100%
            final_take_profit = min(final_take_profit, 3.0)  # Arbitrary upper limit for take profit

            
            return final_stop_loss, final_take_profit
            
        except Exception as e:
            logger.error(f"Error calculating dynamic risk parameters: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return self.stop_loss_pct, self.take_profit_pct

    def wait_for_fill(self, order_id: str, timeout_seconds: int = 30) -> Optional[dict]:
        """Wait for an order to be filled with timeout"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                order = self.exchange.fetch_open_order(order_id, 'BTC/USDT:USDT')
                logger.info("Waiting for order to be filled...")
                logger.info(f"Order status: {order['info']['orderStatus']}")
                if order['info']['orderStatus'] == 'Filled':
                    return order
                elif order['info']['orderStatus'] == 'Cancelled':
                    logger.warning(f"Reason: {order['info']['rejectReason']}")
                    return None
                time.sleep(1)
            return None
        except Exception as e:
            logger.error(f"Error waiting for order fill: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            return None

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

if __name__ == "__main__":
    # Initialize and run the bot
    bot = VWAPScalpingBot(
        symbol='BTC/USDT',
        timeframe='1m',
        test_mode=True  # Start in test mode first
    )
    bot.run()

