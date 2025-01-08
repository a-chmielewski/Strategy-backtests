import time
import logging
from VWAP_Bot import VWAPScalpingBot  # Ensure this path is correct based on your setup

# Configure logging to display only critical messages to keep the test output clean
logging.basicConfig(level=logging.CRITICAL)

def test_place_order():
    # Initialize the bot in test mode
    bot = VWAPScalpingBot(
        symbol='BTC/USDT',
        timeframe='1m',
        test_mode=True  # Ensure you're using testnet
    )
    
    # Fetch historical data
    historical_data = bot.fetch_historical_data()
    if historical_data.empty:
        print("Failed to fetch historical data. Exiting test.")
        return
    
    bot.historical_data = historical_data
    
    # Calculate indicators
    indicators_calculated = bot.calculate_indicators()
    if not indicators_calculated:
        print("Failed to calculate indicators. Exiting test.")
        return
    
    # Get the latest close price
    current_price = bot.historical_data['close'].iloc[-1]
    print(f"Current Price: {current_price}")
    
    # Define the size in USDT for 0.001 BTC
    size_btc = 0.001
    size_usdt = size_btc * current_price
    print(f"Placing LONG order for {size_btc} BTC (~{size_usdt:.2f} USDT)")
    
    # Place a LONG order with take profit and stop loss
    order_success = bot.place_order('long', size_usdt)
    if order_success:
        print("Order placed successfully.")
    else:
        print("Failed to place order.")
        return  # Exit if order placement failed
    
    # Wait for a short duration to ensure the order is processed
    print("Waiting for order to be processed...")
    time.sleep(10)  # Adjust as necessary
    
    # Check if the stop loss and take profit orders have been placed
    open_orders = bot.exchange.fetch_open_orders('BTC/USDT:USDT')
    print("Open Orders:")
    for order in open_orders:
        print(f"Order ID: {order['id']}, Type: {order['type']}, Side: {order['side']}, Price: {order['price']}, Status: {order['status']}")
    
    # Optionally, simulate price movements or wait for market conditions to trigger stop loss/take profit
    
    # For testing purposes, cancel all open orders after the test
    print("Cancelling all open orders...")
    bot.exchange.cancel_all_orders('BTC/USDT:USDT')
    print("All open orders have been cancelled.")
    
    # Close any open positions
    print("Closing any open positions...")
    bot.sync_position()
    if bot.current_position['size'] != 0:
        close_side = 'buy' if bot.current_position['side'] == 'short' else 'sell'
        bot.exchange.create_order(
            symbol='BTC/USDT:USDT',
            type='market',
            side=close_side,
            amount=bot.current_position['size'],
            params={'reduceOnly': True}
        )
        print("Position closed.")
    else:
        print("No open positions.")
        
if __name__ == "__main__":
    test_place_order()
