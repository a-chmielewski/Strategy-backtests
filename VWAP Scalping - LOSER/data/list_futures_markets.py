import ccxt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('list_futures_markets.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Replace these with your actual Bybit API credentials
from bybit_keys import TEST_API_KEY, TEST_API_SECRET, API_KEY, API_SECRET

# ... [previous imports and configurations] ...

def list_futures_markets(test_mode=True):
    try:
        exchange = ccxt.bybit({
            'apiKey': TEST_API_KEY if test_mode else API_KEY,
            'secret': TEST_API_SECRET if test_mode else API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',     # Access futures markets
                'defaultCategory': 'linear', # Specify 'linear' for linear markets
            }
        })

        if test_mode:
            exchange.set_sandbox_mode(True)
            logger.info("Running in testnet mode with test API keys")
        else:
            exchange.set_sandbox_mode(False)
            logger.info("Running in mainnet mode with production API keys")

        exchange.load_markets()
        logger.info("Markets loaded successfully.")

        # Filter futures markets
        futures_markets = {symbol: market for symbol, market in exchange.markets.items() if market.get('linear') or market.get('inverse')}

        if not futures_markets:
            logger.warning("No futures markets found. Ensure you're using the correct symbols and that futures are enabled on your account.")
            return

        logger.info("Available Futures Markets and Their Details:")
        for symbol, market in futures_markets.items():
            logger.info(f"Symbol: {symbol}")
            logger.info(f" - Type: {market['type']}")
            logger.info(f" - Linear: {market.get('linear')}")
            logger.info(f" - Inverse: {market.get('inverse')}")
            logger.info(f" - Settle: {market.get('settle')}")
            logger.info(f" - Leverage Limits: {market.get('limits', {}).get('leverage')}")
            logger.info(f" - Margin Trading: {market.get('info', {}).get('marginTrading')}")
            logger.info("")

    except ccxt.BaseError as e:
        logger.error(f"CCXT Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")


if __name__ == "__main__":
    list_futures_markets(test_mode=True)
