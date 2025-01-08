import ccxt
from typing import Any, Dict, List, Optional, Tuple
from .exchange_interface import ExchangeInterface
from config.config import Config
from logging_setup.logger import logger
import math
import time

class ByBitExchange(ExchangeInterface):
    def __init__(self, config: Config):
        try:
            exchange_config = config.get('exchange')
            
            # Determine which API keys to use based on test_mode
            is_test_mode = exchange_config.get('test_mode', True)  # Default to test mode for safety
            api_key = exchange_config['test_api_key'] if is_test_mode else exchange_config['api_key']
            api_secret = exchange_config['test_api_secret'] if is_test_mode else exchange_config['api_secret']
            
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

            if is_test_mode:
                self.exchange.set_sandbox_mode(True)
                logger.info("ByBit Exchange initialized in testnet mode")
            else:
                logger.info("ByBit Exchange initialized in mainnet mode")

            # Test connection and fetch balance
            try:
                balance = self.fetch_balance()
                total_equity = float(balance['info']['result']['list'][0]['totalEquity'])
                logger.info(f"Successfully connected to ByBit {'testnet' if is_test_mode else 'mainnet'}")
                logger.info(f"Account balance fetched successfully. Current balance: {total_equity}")
            except Exception as e:
                logger.error(f"Failed to fetch account balance: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Failed to initialize ByBitExchange: {str(e)}")
            raise

    def place_order(self, symbol: str, order_type: str, side: str, 
                   amount: float, price: Optional[float] = None, 
                   params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Place an order with enhanced error handling and parameter validation
        """
        try:
            # Ensure minimum order size
            MIN_ORDER_SIZE_USDT = 1.0
            if amount * (price or self.fetch_ticker(symbol)['last']) < MIN_ORDER_SIZE_USDT:
                raise ValueError(f"Order size too small. Minimum is {MIN_ORDER_SIZE_USDT} USDT")

            # Prepare base parameters
            base_params = {
                'category': 'linear',
                'timeInForce': 'GTC',
            }
            
            # Merge with provided params
            if params:
                base_params.update(params)

            # Place the order
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=base_params
            )

            logger.info(f"Order placed successfully: {order['id']}")
            return order

        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for order: {str(e)}")
            raise
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order parameters: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    def wait_for_fill(self, order_id: str, symbol: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for an order to be filled with timeout"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                order = self.fetch_order(order_id, symbol)
                if order['status'] == 'closed':
                    return order
                elif order['status'] == 'canceled':
                    logger.warning(f"Order {order_id} was cancelled")
                    return None
                time.sleep(1)
            logger.warning(f"Order {order_id} timed out after {timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Error waiting for order fill: {str(e)}")
            return None

    def fetch_balance(self) -> Dict[str, Any]:
        """Fetch balance with linear category parameter"""
        return self.exchange.fetch_balance({'category': 'linear'})

    def fetch_positions(self, symbols: Optional[List[str]] = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch positions with enhanced error handling"""
        try:
            base_params = {'category': 'linear'}
            if params:
                base_params.update(params)
            return self.exchange.fetch_positions(symbols, base_params)
        except Exception as e:
            logger.error(f"Error fetching positions: {str(e)}")
            raise

    def set_leverage(self, leverage: int, symbol: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Set leverage with proper parameters"""
        try:
            base_params = {'category': 'linear'}
            if params:
                base_params.update(params)
            return self.exchange.set_leverage(leverage, symbol, base_params)
        except Exception as e:
            logger.error(f"Error setting leverage: {str(e)}")
            raise

    def cancel_all_orders(self, symbol: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Cancel all orders with proper parameters"""
        try:
            base_params = {'category': 'linear'}
            if params:
                base_params.update(params)
            return self.exchange.cancel_all_orders(symbol, base_params)
        except Exception as e:
            logger.error(f"Error cancelling all orders: {str(e)}")
            raise

    def fetch_order_book(self, symbol: str, limit: Optional[int] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch order book with proper parameters"""
        try:
            base_params = {'category': 'linear'}
            if params:
                base_params.update(params)
            return self.exchange.fetch_order_book(symbol, limit, base_params)
        except Exception as e:
            logger.error(f"Error fetching order book: {str(e)}")
            raise

    def fetch_ticker(self, symbol: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fetch ticker with proper parameters"""
        try:
            base_params = {'category': 'linear'}
            if params:
                base_params.update(params)
            return self.exchange.fetch_ticker(symbol, base_params)
        except Exception as e:
            logger.error(f"Error fetching ticker: {str(e)}")
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, 
                   limit: Optional[int] = None, params: Dict[str, Any] = None) -> List[List[Any]]:
        """Fetch OHLCV data with proper parameters"""
        try:
            base_params = {'category': 'linear'}
            if params:
                base_params.update(params)
            return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit, base_params)
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise

