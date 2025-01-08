from trading_bot.config import BotConfig
from trading_bot.trading_bot import TradingBot
from trading_bot.strategies.double_ema_stoch import DoubleEMAStochStrategy
from pybit.unified_trading import HTTP

def fetch_latest_kline(session: HTTP, symbol: str, interval: str) -> dict:
    """Fetch the latest kline/candlestick data"""
    # Convert interval to Bybit API format
    interval_map = {
        "1": "1",
        "3": "3",
        "5": "5",
        "15": "15",
        "30": "30",
        "60": "60",
        "120": "120",
        "240": "240",
        "360": "360",
        "720": "720",
        "D": "D",
        "W": "W",
        "M": "M"
    }
    
    response = session.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval_map.get(interval, "5"),
        limit=1
    )
    return response['result']['list'][0]

def create_strategy_runner(strategy: DoubleEMAStochStrategy, session: HTTP, symbol: str, interval: str):
    """Create a function that will run the strategy with latest market data"""
    
    def strategy_runner(current_price: float) -> dict:
        try:
            # Fetch latest candlestick data
            kline = fetch_latest_kline(session, symbol, interval)
            
            if not kline:
                print("No kline data received")
                return None
            
            # Update indicators with new data
            strategy.calculate_indicators(
                high=float(kline[2]),    # High price
                low=float(kline[3]),     # Low price
                close=float(kline[4])    # Close price
            )
            
            # Generate trading signal
            return strategy.generate_signal(current_price)
            
        except Exception as e:
            print(f"Error in strategy runner: {str(e)}")
            return None
    
    return strategy_runner

if __name__ == "__main__":
    # Initialize configuration
    config = BotConfig(
        API_KEY=BotConfig.API_KEY,
        API_SECRET=BotConfig.API_SECRET,
        SYMBOL="ETHUSDT",
        TIMEFRAME="5",
        STOP_LOSS_PCT=0.25,
        TAKE_PROFIT_PCT=0.5,
        INITIAL_BALANCE=100,
        LEVERAGE=50
    )
    
    # Initialize strategy
    strategy = DoubleEMAStochStrategy(
        ema_slow=50,
        ema_fast=150,
        stoch_k=5,
        stoch_d=3,
        slowing=3,
        stoch_overbought=80,
        stoch_oversold=20,
        stop_loss=config.STOP_LOSS_PCT/100,
        take_profit=config.TAKE_PROFIT_PCT/100
    )
    
    # Create Bybit session for fetching kline data
    session = HTTP(
        testnet=False,
        api_key=config.API_KEY,
        api_secret=config.API_SECRET
    )
    
    # Create strategy runner
    strategy_runner = create_strategy_runner(
        strategy=strategy,
        session=session,
        symbol=config.SYMBOL,
        interval=config.TIMEFRAME
    )
    
    # Initialize and run the bot
    bot = TradingBot(config, strategy_runner)
    
    print(f"Starting paper trading bot for {config.SYMBOL} on {config.TIMEFRAME}m timeframe...")
    print("Press Ctrl+C to stop the bot")
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot stopped due to error: {str(e)}") 