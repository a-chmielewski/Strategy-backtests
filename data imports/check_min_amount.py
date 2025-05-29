from pybit.unified_trading import HTTP
import bybit_keys as keys

SYMBOL = "LINKUSDT"

# Connect to Bybit
session = HTTP(
    api_key=keys.API_KEY,
    api_secret=keys.API_SECRET
)

def get_min_order_amount(symbol: str) -> float:
    """Fetch the minimum order amount for a given symbol from Bybit."""
    try:
        response = session.get_instruments_info(
            category="spot",
            symbol=symbol
        )
        assert response["retCode"] == 0, f"API error: {response.get('retMsg', 'Unknown error')}"
        instruments = response["result"]["list"]
        assert instruments, f"No instrument info found for {symbol}"
        instrument_info = instruments[0]
        min_order_qty = float(instrument_info["lotSizeFilter"]["minOrderQty"])
        return min_order_qty
    except Exception as error:
        print(f"Error fetching minimum order amount: {error}")
        raise

def main():
    min_amount = get_min_order_amount(SYMBOL)
    print(f"Minimum order amount for {SYMBOL}: {min_amount}")

if __name__ == "__main__":
    main()
