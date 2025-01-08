from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

class ExchangeInterface(ABC):
    @abstractmethod
    def initialize_exchange(self, test_mode: bool = True) -> None:
        """Initialize exchange connection with test mode option"""
        pass

    @abstractmethod
    def fetch_balance(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch account balance"""
        pass

    @abstractmethod
    def fetch_positions(self, symbols: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch current positions"""
        pass

    @abstractmethod
    def set_leverage(self, leverage: int, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Set leverage for a symbol"""
        pass

    @abstractmethod
    def create_order(self, symbol: str, order_type: str, side: str, 
                    amount: float, price: Optional[float] = None, 
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new order"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Cancel a specific order"""
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Cancel all open orders"""
        pass

    @abstractmethod
    def fetch_open_orders(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch open orders"""
        pass

    @abstractmethod
    def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch a specific order"""
        pass

    @abstractmethod
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None, 
                        params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch order book"""
        pass

    @abstractmethod
    def fetch_ticker(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch ticker"""
        pass

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, since: Optional[int] = None, 
                    limit: Optional[int] = None, params: Optional[Dict[str, Any]] = None) -> List[List[Any]]:
        """Fetch OHLCV data"""
        pass

    @abstractmethod
    def wait_for_fill(self, order_id: str, symbol: str, timeout_seconds: int = 30) -> Optional[Dict[str, Any]]:
        """Wait for an order to be filled with timeout"""
        pass

    @abstractmethod
    def sync_position(self, symbol: str) -> Dict[str, Any]:
        """Sync position with exchange"""
        pass

    @abstractmethod
    def close_position(self, symbol: str, position_data: Dict[str, Any], 
                      exit_reason: str) -> Tuple[bool, Optional[float]]:
        """Close an open position"""
        pass

    @abstractmethod
    def place_stop_loss(self, symbol: str, position_data: Dict[str, Any], 
                       stop_price: float, amount: float) -> Dict[str, Any]:
        """Place a stop loss order"""
        pass

    @abstractmethod
    def place_take_profit(self, symbol: str, position_data: Dict[str, Any], 
                         take_profit_price: float, amount: float) -> Dict[str, Any]:
        """Place a take profit order"""
        pass

    @abstractmethod
    def update_trailing_stop(self, symbol: str, position_data: Dict[str, Any], 
                           new_stop_price: float) -> Dict[str, Any]:
        """Update trailing stop loss"""
        pass

    @abstractmethod
    def get_position_size(self, current_price: float, balance: Optional[float] = None) -> float:
        """Calculate position size based on risk parameters"""
        pass

    @abstractmethod
    def determine_order_type(self, side: str, current_price: float) -> Tuple[str, Optional[float], Dict[str, Any]]:
        """Determine order type and parameters based on market conditions"""
        pass

    @abstractmethod
    def verify_order_parameters(self, symbol: str, order_type: str, side: str, 
                              amount: float, price: Optional[float] = None) -> bool:
        """Verify order parameters are valid"""
        pass

    @abstractmethod
    def handle_order_error(self, error: Exception, order_data: Dict[str, Any]) -> None:
        """Handle order placement errors"""
        pass
