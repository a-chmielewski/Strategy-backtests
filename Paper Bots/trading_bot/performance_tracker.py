from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import json

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    position_size: float
    side: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def to_dict(self):
        return {
            'entry_time': self.entry_time.isoformat(),
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'side': self.side,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }

class PerformanceTracker:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trades: List[Trade] = []
        self.daily_pnl: Dict[str, float] = {}
        
    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        if trade.pnl:
            self.current_balance += trade.pnl
            date = trade.exit_time.date().isoformat()
            self.daily_pnl[date] = self.daily_pnl.get(date, 0) + trade.pnl
    
    def get_statistics(self) -> Dict:
        if not self.trades:
            return {}
            
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'total_pnl': sum(t.pnl for t in self.trades if t.pnl),
            'current_balance': self.current_balance,
            'return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        }
    
    def save_results(self, filename: str = 'trading_results.json'):
        results = {
            'statistics': self.get_statistics(),
            'trades': [trade.to_dict() for trade in self.trades],
            'daily_pnl': self.daily_pnl
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4) 