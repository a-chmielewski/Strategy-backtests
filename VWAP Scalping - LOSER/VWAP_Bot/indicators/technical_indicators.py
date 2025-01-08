import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional, List
from logging_setup.logger import logger

class TechnicalIndicators:
    def __init__(self, config: Dict):
        """Initialize with configuration parameters"""
        self.config = config
        
        # Extract indicator parameters from config
        indicator_config = config.get('indicators', {})
        self.vwap_period = indicator_config.get('vwap_period', 20)
        self.rsi_period = indicator_config.get('rsi_period', 14)
        self.macd_fast = indicator_config.get('macd_fast', 12)
        self.macd_slow = indicator_config.get('macd_slow', 26)
        self.macd_signal = indicator_config.get('macd_signal', 9)
        self.bb_period = indicator_config.get('bb_period', 20)
        self.bb_std_dev = indicator_config.get('bb_std_dev', 2.0)
        self.atr_period = indicator_config.get('atr_period', 14)
        self.sar_step = indicator_config.get('sar_step', 0.02)
        self.sar_max = indicator_config.get('sar_max', 0.2)
        self.volume_ma_period = indicator_config.get('volume_ma_period', 20)
        self.volatility_lookback = indicator_config.get('volatility', {}).get('lookback', 100)
        self.risk_atr_multiplier = config.get('trading', {}).get('risk_atr_multiplier', 1.5)

    def calculate_vwap(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate VWAP using vectorized operations"""
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

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            # Calculate VWAP
            indicators['vwap'] = self.calculate_vwap(df)
            
            # Calculate RSI
            indicators['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'].values,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'].values,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std_dev,
                nbdevdn=self.bb_std_dev
            )
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Calculate ATR
            indicators['atr'] = talib.ATR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=self.atr_period
            )
            
            # Calculate Parabolic SAR
            indicators['sar'] = talib.SAR(
                df['high'].values,
                df['low'].values,
                acceleration=self.sar_step,
                maximum=self.sar_max
            )
            
            # Calculate Volume MA
            indicators['volume_ma'] = talib.SMA(
                df['volume'].values,
                timeperiod=self.volume_ma_period
            )
            
            # Calculate Volatility
            returns = pd.Series(df['close']).pct_change()
            indicators['volatility'] = returns.rolling(
                window=self.volatility_lookback,
                min_periods=1
            ).std() * np.sqrt(1440)  # Annualize for 1-minute data
            
            # Calculate dynamic RSI levels based on ATR
            indicators['rsi_dynamic_overbought'] = np.minimum(
                70 + (self.risk_atr_multiplier * indicators['atr']),
                100
            )
            indicators['rsi_dynamic_oversold'] = np.maximum(
                30 - (self.risk_atr_multiplier * indicators['atr']),
                0
            )
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def verify_indicators(self, indicators: Dict[str, np.ndarray]) -> bool:
        """Verify the quality of calculated indicators"""
        try:
            required_indicators = ['vwap', 'rsi', 'macd', 'volatility', 'atr']
            
            # Check for required indicators
            for indicator in required_indicators:
                if (indicator not in indicators or 
                    len(indicators[indicator]) < 20 or 
                    np.all(np.isnan(indicators[indicator][-20:]))):
                    logger.warning(f"Invalid or missing data for indicator: {indicator}")
                    return False
            
            # Check last values of critical indicators
            critical_indicators = ['vwap', 'rsi', 'macd', 'bb_upper', 'atr', 'sar']
            for indicator in critical_indicators:
                if np.isnan(indicators[indicator][-1]):
                    logger.warning(f"Last value of {indicator} is NaN")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying indicators: {str(e)}")
            return False
