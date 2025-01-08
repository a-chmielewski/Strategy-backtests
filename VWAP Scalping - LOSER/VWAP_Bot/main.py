import time
import pandas as pd
from config.config import Config
from exchange.bybit_interface import ByBitExchange
from data.data_handler import DataHandler
from indicators.technical_indicators import TechnicalIndicators
from strategies.strategy_factory import StrategyFactory
from risk_management.risk_handler import RiskHandler
from logging_setup.logger import logger
from typing import Tuple, Optional

class TradingBot:
    def __init__(self):
        # Load configuration
        self.config = Config()
        
        # Initialize components
        self.exchange = ByBitExchange(self.config)
        self.data_handler = DataHandler(self.exchange, self.config)
        self.indicators = TechnicalIndicators(self.config.get('indicators'))
        self.strategy = StrategyFactory.get_strategy('vwap_scalping', self.config.get('strategies'))
        self.risk_handler = RiskHandler(self.config.get('risk_management'))
        
        # Initialize position tracking
        self.current_position = {}
    
    def run(self):
        logger.info("Starting Trading Bot...")
        try:
            while True:
                try:
                    # Fetch and process data
                    df = self.data_handler.get_latest_data()
                    
                    if df.empty:
                        logger.warning("No data fetched. Skipping iteration.")
                        time.sleep(5)
                        continue
                    
                    # Calculate indicators
                    df = self.indicators.compute_all_indicators(df)
                    
                    # Determine entry conditions
                    if not self.current_position:
                        should_enter, signal = self.strategy.check_entry_conditions(df)
                        if should_enter:
                            # Get account equity
                            balance = self.exchange.fetch_balance()
                            equity = float(balance['total']['USDT'])
                            
                            # Calculate position size
                            size_usdt = self.risk_handler.get_position_size(equity, df['close'].iloc[-1])
                            
                            # Place order
                            order_success = self.place_order(signal, size_usdt, df)
                            if order_success:
                                logger.info(f"Order placed: {signal} {size_usdt} USDT")
                    else:
                        # Check exit conditions
                        should_exit, reason = self.strategy.check_exit_conditions(df, self.current_position)
                        if should_exit:
                            self.exit_position()
                            logger.info(f"Exited position due to: {reason}")
                    
                    time.sleep(5)  # Adjust as per timeframe and rate limits
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            self.cleanup()
    
    def place_order(self, signal: str, size_usdt: float, df: pd.DataFrame) -> bool:
        try:
            # Check minimum order size
            MIN_ORDER_SIZE_USDT = 1.0
            if size_usdt < MIN_ORDER_SIZE_USDT:
                logger.warning(f"Order size ({size_usdt} USDT) is below minimum requirement of {MIN_ORDER_SIZE_USDT} USDT")
                return False

            current_price = df['close'].iloc[-1]
            size_btc = round(size_usdt / current_price, 3)  # Convert USDT to BTC
            
            # Get dynamic risk parameters based on ATR
            current_atr = self.indicators.compute_atr(df)[-1]
            stop_loss_pct, take_profit_pct = self.risk_handler.get_dynamic_risk_parameters(current_price, current_atr)
            
            # Determine order type based on market conditions
            order_type, limit_price, order_params = self.determine_order_type(signal, current_price)
            
            try:
                # Place main order
                if order_type == 'limit':
                    main_order = self.exchange.create_order(
                        symbol=self.config.get('trading', 'symbol'),
                        type='limit',
                        side='buy' if signal == 'long' else 'sell',
                        amount=size_btc,
                        price=limit_price,
                        params=order_params
                    )
                else:
                    main_order = self.exchange.create_order(
                        symbol=self.config.get('trading', 'symbol'),
                        type='market',
                        side='buy' if signal == 'long' else 'sell',
                        amount=size_btc,
                        params=order_params
                    )

                # Wait for fill
                filled_order = self.wait_for_fill(main_order['id'])
                if not filled_order:
                    logger.warning("Order not filled within timeout, cancelling")
                    self.exchange.cancel_all_orders(self.config.get('trading', 'symbol'))
                    return False

                # Get fill price
                fill_price = float(filled_order['average'])

                # Calculate stop loss and take profit prices
                if signal == 'long':
                    stop_loss_price = fill_price * (1 - stop_loss_pct)
                    take_profit_price = fill_price * (1 + take_profit_pct)
                else:
                    stop_loss_price = fill_price * (1 + stop_loss_pct)
                    take_profit_price = fill_price * (1 - take_profit_pct)

                # Place stop loss order
                sl_order = self.exchange.create_order(
                    symbol=self.config.get('trading', 'symbol'),
                    type='market',
                    side='sell' if signal == 'long' else 'buy',
                    amount=size_btc,
                    params={
                        'stopPrice': stop_loss_price,
                        'triggerPrice': stop_loss_price,
                        'triggerDirection': 2 if signal == 'long' else 1,
                        'reduceOnly': True,
                        'timeInForce': 'GTC',
                        'closeOnTrigger': True,
                        'category': 'linear',
                    }
                )

                # Place take profit order
                tp_order = self.exchange.create_order(
                    symbol=self.config.get('trading', 'symbol'),
                    type='market',
                    side='sell' if signal == 'long' else 'buy',
                    amount=size_btc,
                    params={
                        'stopPrice': take_profit_price,
                        'triggerPrice': take_profit_price,
                        'triggerDirection': 1 if signal == 'long' else 2,
                        'reduceOnly': True,
                        'timeInForce': 'GTC',
                        'closeOnTrigger': True,
                        'category': 'linear',
                    }
                )

                # Update position tracking
                self.current_position = {
                    'size': size_btc,
                    'side': signal,
                    'entry_price': fill_price,
                    'stop_loss_order_id': sl_order['id'],
                    'take_profit_order_id': tp_order['id'],
                    'entry_time': pd.Timestamp.now(tz='UTC')
                }

                logger.info(f"Successfully placed {signal} position: Size={size_btc}, Entry={fill_price:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}")
                return True

            except Exception as e:
                logger.error(f"Error placing orders: {str(e)}")
                self.exchange.cancel_all_orders(self.config.get('trading', 'symbol'))
                return False

        except Exception as e:
            logger.error(f"Error in place_order: {str(e)}")
            return False

    def determine_order_type(self, side: str, current_price: float) -> Tuple[str, float, dict]:
        try:
            # Get market conditions
            order_book = self.exchange.fetch_order_book(self.config.get('trading', 'symbol'))
            bid = order_book['bids'][0][0] if order_book['bids'] else current_price
            ask = order_book['asks'][0][0] if order_book['asks'] else current_price
            spread_pct = (ask - bid) / current_price if current_price != 0 else 0.0

            # Base parameters
            params = {
                'category': 'linear',
                'position_idx': 0
            }

            # Use market order for high spread
            if spread_pct > 0.001:
                return 'market', None, params

            # Use limit order with post-only for normal conditions
            if side == 'long':
                limit_price = bid + (current_price * 0.0001)  # Slightly above bid
            else:
                limit_price = ask - (current_price * 0.0001)  # Slightly below ask

            params['postOnly'] = True
            
            # For very tight spreads, use IOC instead
            if spread_pct < 0.0005:
                params.pop('postOnly', None)
                params['timeInForce'] = 'IOC'

            return 'limit', limit_price, params

        except Exception as e:
            logger.error(f"Error determining order type: {str(e)}")
            return 'market', None, {'category': 'linear', 'position_idx': 0}
    
    def exit_position(self):
        try:
            side = 'sell' if self.current_position['side'] == 'long' else 'buy'
            self.exchange.create_order(
                symbol=self.config.get('trading', 'symbol'),
                order_type='market',
                side=side,
                amount=self.current_position['size']
            )
            self.current_position = {}
        except Exception as e:
            logger.error(f"Failed to exit position: {str(e)}")
    
    def cleanup(self):
        try:
            if self.current_position:
                self.exit_position()
            self.exchange.cancel_all_orders(self.config.get('trading', 'symbol'))
            logger.info("Cleaned up open orders and positions.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
