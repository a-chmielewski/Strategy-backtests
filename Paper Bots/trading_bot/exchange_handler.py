from typing import Dict, Optional, List
from pybit.unified_trading import HTTP
from datetime import datetime
from trading_bot.logger import setup_logger

class ExchangeHandler:
    def __init__(self, api_key: str, api_secret: str, initial_balance: float = 100.0):
        self.session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret
        )
        self.paper_balance = initial_balance
        self.open_orders = []
        self.positions = []
        self.logger = setup_logger('exchange_handler')
        self.current_leverage = 50
        
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for paper trading"""
        try:
            self.current_leverage = leverage
            self.logger.info(f"Setting leverage to {leverage}x for {symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            return False
            
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            response = self.session.get_tickers(
                category="linear",
                symbol=symbol
            )
            return float(response['result']['list'][0]['lastPrice'])
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return 0.0
        
    def get_balance(self) -> float:
        """Get current paper trading balance"""
        return self.paper_balance
        
    def place_order(self, symbol: str, side: str, qty: float, 
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Dict:
        """Place a paper trading order"""
        try:
            current_price = self.get_current_price(symbol)
            
            # Calculate position value considering leverage
            position_value = qty * current_price
            margin_required = position_value / self.current_leverage
            
            # Check if we have enough margin
            if margin_required > self.paper_balance:
                self.logger.error(f"Insufficient margin: Required=${margin_required:.2f}, Available=${self.paper_balance:.2f}")
                return None
            
            # Create paper trade
            order = {
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'price': current_price,
                'leverage': self.current_leverage,
                'timestamp': datetime.now(),
                'status': 'FILLED'
            }
            
            # Update paper balance
            self.paper_balance -= margin_required
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error placing paper order: {str(e)}")
            return None
            
    def check_order_status(self) -> List[tuple]:
        """Check if any stop loss or take profit levels have been hit"""
        if not self.positions:
            return []
            
        executed_orders = []
        for position in self.positions[:]:
            current_price = self.get_current_price(position['symbol'])
            
            # Calculate PnL with leverage
            price_diff = current_price - position['entry_price']
            pnl_pct = (price_diff / position['entry_price']) * position['leverage']
            
            # Check stop loss (adjusted for leverage)
            if position['stop_loss'] and current_price <= position['stop_loss']:
                order = self.place_order(
                    symbol=position['symbol'],
                    side="SELL",
                    qty=position['size']
                )
                if order:
                    executed_orders.append(('stop_loss', order))
                    
            # Check take profit (adjusted for leverage)
            elif position['take_profit'] and current_price >= position['take_profit']:
                order = self.place_order(
                    symbol=position['symbol'],
                    side="SELL",
                    qty=position['size']
                )
                if order:
                    executed_orders.append(('take_profit', order))
        
        return executed_orders