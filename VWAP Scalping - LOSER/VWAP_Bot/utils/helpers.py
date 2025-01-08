import re
import math
from typing import Optional

def timeframe_to_ms(timeframe: str) -> int:
    match = re.match(r'(\d+)([mhd])', timeframe)
    if not match:
        raise ValueError(f"Unsupported timeframe format: {timeframe}")
    value, unit = match.groups()
    value = int(value)
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")

def round_down(value: float, decimals: int = 2) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

# Add more utility functions as needed
