from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from typing import Dict

class BaseStrategy(ABC):
    @abstractmethod
    def check_entry_conditions(self, data: pd.DataFrame) -> Tuple[bool, str]:
        pass
    
    @abstractmethod
    def check_exit_conditions(self, data: pd.DataFrame, current_position: Dict) -> Tuple[bool, str]:
        pass
