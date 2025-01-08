from typing import Callable, Optional
import time
from datetime import datetime
import pandas as pd
from typing import Dict
import traceback

from trading_bot.config import BotConfig
from trading_bot.exchange_handler import ExchangeHandler
from trading_bot.performance_tracker import PerformanceTracker, Trade
from trading_bot.logger import setup_logger

class TradingBot:
    def __init__(self, config: BotConfig, strategy: Callable):
        self.config = config
        self.strategy = strategy
        self.exchange = ExchangeHandler(
            config.API_KEY, 
            config.API_SECRET,
            initial_balance=config.INITIAL_BALANCE
        )
        self.performance_tracker = PerformanceTracker(config.INITIAL_BALANCE)
        self.logger = setup_logger('trading_bot')
        self.current_position: Optional[Trade] = None
        
    def calculate_position_size(self, current_price: float) -> float:
        """
        Calculate position size based on current equity and risk management rules
        
        Args:
            current_price (float): Current price of the asset
            
        Returns:
            float: Position size in units of the asset
        """
        try:
            # Get current equity
            current_equity = self.exchange.get_balance()
            
            # Determine position value based on equity
            if current_equity < 100:
                position_value = current_equity
            else:
                position_value = 100.0
                
            # Set leverage
            position_value = position_value * self.config.LEVERAGE
            
            # Calculate position size
            position_size = position_value / current_price
            
            self.logger.info(
                f"Position size calculation: "
                f"Equity=${current_equity:.2f}, "
                f"Risk Amount=${position_value:.2f}, "
                f"Leverage={self.config.LEVERAGE}x, "
                f"Position Value=${position_value:.2f}, "
                f"Size={position_size:.6f}"
            )
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
        
    def execute_trade_signal(self, signal: Dict):
        # Check if we're out of equity
        if self.exchange.get_balance() <= 0:
            self.logger.error("Account has run out of equity! Stopping bot...")
            raise SystemExit("Bot stopped due to insufficient equity")
        
        # First check if any existing positions need to be closed (SL/TP hit)
        executed_orders = self.exchange.check_order_status()
        for order_type, order in executed_orders:
            if self.current_position:
                self.current_position.exit_time = order['timestamp']
                self.current_position.exit_price = order['price']
                self.current_position.pnl = (
                    (order['price'] - self.current_position.entry_price) 
                    * self.current_position.position_size
                )
                
                self.performance_tracker.add_trade(self.current_position)
                self.logger.info(
                    f"Position closed by {order_type} at {order['price']}, "
                    f"PNL: {self.current_position.pnl}"
                )
                self.current_position = None
        
        # Process new signals
        if signal['action'] == 'BUY' and not self.current_position:
            current_price = self.exchange.get_current_price(self.config.SYMBOL)
            position_size = self.calculate_position_size(current_price)
            
            stop_loss = current_price * (1 - self.config.STOP_LOSS_PCT / 100)
            take_profit = current_price * (1 + self.config.TAKE_PROFIT_PCT / 100)
            
            order = self.exchange.place_order(
                symbol=self.config.SYMBOL,
                side="BUY",
                qty=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if order:
                self.current_position = Trade(
                    entry_time=order['timestamp'],
                    entry_price=order['price'],
                    position_size=position_size,
                    side="LONG",
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                self.logger.info(
                    f"Opened paper LONG position at {order['price']}, "
                    f"SL: {stop_loss}, TP: {take_profit}"
                )

        elif signal['action'] == 'SELL' and not self.current_position:
            current_price = self.exchange.get_current_price(self.config.SYMBOL)
            position_size = self.calculate_position_size(current_price)

            stop_loss = current_price * (1 + self.config.STOP_LOSS_PCT / 100)
            take_profit = current_price * (1 - self.config.TAKE_PROFIT_PCT / 100)

            order = self.exchange.place_order(
                symbol=self.config.SYMBOL,
                side="SELL",
                qty=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if order:
                self.current_position = Trade(
                    entry_time=order['timestamp'],
                    entry_price=order['price'],
                    position_size=position_size,
                    side="SHORT",
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
        elif signal['action'] == 'SELL' and self.current_position:
            order = self.exchange.place_order(
                symbol=self.config.SYMBOL,
                side="SELL",
                qty=self.current_position.position_size
            )
            
            if order:
                self.current_position.exit_time = order['timestamp']
                self.current_position.exit_price = order['price']
                self.current_position.pnl = (
                    (order['price'] - self.current_position.entry_price) 
                    * self.current_position.position_size
                )
                
                self.performance_tracker.add_trade(self.current_position)
                self.logger.info(
                    f"Closed paper position at {order['price']}, "
                    f"PNL: {self.current_position.pnl}"
                )
                self.current_position = None

        elif signal['action'] == 'BUY' and self.current_position:
            order = self.exchange.place_order(
                symbol=self.config.SYMBOL,
                side="BUY",
                qty=self.current_position.position_size
            )

            if order:
                self.current_position.exit_time = order['timestamp']
                self.current_position.exit_price = order['price']
                self.current_position.pnl = (
                    (order['price'] - self.current_position.entry_price) 
                    * self.current_position.position_size
                )
    
        # After closing a position, update performance metrics
        if self.current_position and self.current_position.exit_price:
            self.performance_tracker.add_trade(self.current_position)
            self.performance_tracker.save_results()  # Save after each trade
            
            # Log performance metrics
            stats = self.performance_tracker.get_statistics()
            self.logger.info(f"Trade closed - Current equity: ${self.exchange.get_balance():.2f}")
            self.logger.info(f"Total Return: {stats['total_return']:.2f}%")
            self.logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
            self.logger.info(f"Total Trades: {stats['total_trades']}")

    def run(self):
        self.logger.info("Starting trading bot...")
        
        while True:
            try:
                # Get latest market data
                current_price = self.exchange.get_current_price(self.config.SYMBOL)
                
                # Generate trading signal using strategy
                signal = self.strategy(current_price)
                
                # Execute signal if any
                if signal:
                    self.execute_trade_signal(signal)
                
                # Sleep to avoid API rate limits
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(5) 