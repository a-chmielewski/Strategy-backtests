from dataclasses import dataclass

@dataclass
class BotConfig:
    # API Configuration
    API_KEY: str = "edrfXfb5mQcdwIWuDe"
    API_SECRET: str = "jqPn4ED1ExxIj3b9jkrg99SAp7BGhH7bnq9L"
    
    # Trading Parameters
    SYMBOL: str = "ETHUSDT"
    TIMEFRAME: str = "5"
    MAX_POSITIONS: int = 1
    
    # Risk Management
    STOP_LOSS_PCT: float = 1.0  # 1%
    TAKE_PROFIT_PCT: float = 2.0  # 2%
    MAX_DAILY_LOSS_PCT: float = 50.0  # 5%
    
    # Performance Tracking
    INITIAL_BALANCE: float = 100  # USDT 
    
    # Leverage
    LEVERAGE: int = 50  # Default leverage
