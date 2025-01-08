from trading_bot.config import BotConfig
from trading_bot.trading_bot import TradingBot

"""
Strategies to try:
double EMA StochOSC ETH 5m
"""

def simple_strategy(current_price: float) -> dict:
    # Implement your strategy logic here
    # Return {'action': 'BUY'} or {'action': 'SELL'} or None
    pass

if __name__ == "__main__":
    config = BotConfig(
        api_key="your_api_key",
        api_secret="your_api_secret",
        symbol="ETHUSDT",
        initial_balance=1000.0,
        leverage=50  # Set to desired leverage
    )
    
    bot = TradingBot(config, simple_strategy)
    bot.exchange.set_leverage(config.SYMBOL, config.LEVERAGE)
    bot.run() 