from typing import Dict, Type
from strategies.base_strategy import BaseStrategy
from strategies.vwap_scalping import VWAPScalpingStrategy
from indicators.technical_indicators import TechnicalIndicators

STRATEGIES: Dict[str, Type[BaseStrategy]] = {
    'vwap_scalping': VWAPScalpingStrategy,
    # Add new strategies here
}

def register_strategy(name: str, strategy_cls: Type[BaseStrategy]):
    STRATEGIES[name.lower()] = strategy_cls

def get_strategy(name: str, config: Dict, indicators: TechnicalIndicators) -> BaseStrategy:
    strategy_cls = STRATEGIES.get(name.lower())
    if not strategy_cls:
        raise ValueError(f"Strategy '{name}' is not registered.")
    return strategy_cls(config, indicators)
