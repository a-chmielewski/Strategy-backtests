from typing import Type
from strategies.base_strategy import BaseStrategy
from strategies.vwap_scalping import VWAPScalpingStrategy
from typing import Dict
class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_name: str, config: Dict) -> BaseStrategy:
        strategies = {
            'vwap_scalping': VWAPScalpingStrategy,
            # Add other strategies here
        }
        strategy_class = strategies.get(strategy_name.lower())
        if not strategy_class:
            raise ValueError(f"Strategy '{strategy_name}' is not recognized.")
        return strategy_class(config)
