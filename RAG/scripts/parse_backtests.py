import json
import glob
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BacktestResult:
    # Core info
    timestamp: str
    strategy_name: str
    strategy_type: str
    strategy_description: str
    
    # Data info
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    
    # Performance metrics
    final_equity: float
    total_return_pct: float
    buy_hold_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    win_rate_pct: float
    expectancy: float
    total_trades: int
    exposure_time_pct: float
    
    # Strategy specific
    parameters: Dict[str, Any]
    
    # Risk management
    risk_management: Dict[str, str]
    
    # File info
    file_path: str

def load_backtest(file_path: str) -> BacktestResult:
    """Load a single backtest JSON file and return structured data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return BacktestResult(
        timestamp=data['timestamp'],
        strategy_name=data['strategy_name'],
        strategy_type=data['strategy_type'],
        strategy_description=data['strategy_description'],
        
        symbol=data['data_info']['symbol'],
        timeframe=data['data_info']['timeframe'],
        start_date=data['data_info']['start_date'],
        end_date=data['data_info']['end_date'],
        
        final_equity=data['performance']['final_equity'],
        total_return_pct=data['performance']['total_return_pct'],
        buy_hold_return_pct=data['performance']['buy_hold_return_pct'],
        max_drawdown_pct=data['performance']['max_drawdown_pct'],
        sharpe_ratio=data['performance']['sharpe_ratio'],
        profit_factor=data['performance']['profit_factor'],
        win_rate_pct=data['performance']['win_rate_pct'],
        expectancy=data['performance']['expectancy'],
        total_trades=data['performance']['total_trades'],
        exposure_time_pct=data['performance']['exposure_time_pct'],
        
        parameters=data['parameters'],
        risk_management=data['risk_management'],
        file_path=file_path
    )

def load_all_backtests(folder_path: str) -> List[BacktestResult]:
    """Load all JSON backtest files from the specified folder and its subfolders."""
    backtests = []
    
    # Get all JSON files in all subdirectories
    json_pattern = os.path.join(folder_path, "**/*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    for file_path in json_files:
        try:
            backtest = load_backtest(file_path)
            backtests.append(backtest)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return backtests

def generate_text_for_embedding(backtest: BacktestResult) -> str:
    """Generate a text representation of the backtest for embedding."""
    params_str = ", ".join(f"{k}: {v}" for k, v in backtest.parameters.items())
    risk_str = ", ".join(f"{k}: {v}" for k, v in backtest.risk_management.items())
    
    return f"""
Strategy: {backtest.strategy_name} ({backtest.strategy_type})
Description: {backtest.strategy_description}
Symbol: {backtest.symbol}, Timeframe: {backtest.timeframe}
Period: {backtest.start_date} to {backtest.end_date}

Performance:
- Final Equity: {backtest.final_equity:.2f}
- Total Return: {backtest.total_return_pct:.2f}%
- Buy & Hold Return: {backtest.buy_hold_return_pct:.2f}%
- Win Rate: {backtest.win_rate_pct:.2f}%
- Max Drawdown: {backtest.max_drawdown_pct:.2f}%
- Sharpe Ratio: {backtest.sharpe_ratio:.2f}
- Profit Factor: {backtest.profit_factor:.2f}
- Expectancy: {backtest.expectancy:.2f}
- Total Trades: {backtest.total_trades}
- Exposure Time: {backtest.exposure_time_pct:.2f}%

Parameters: {params_str}
Risk Management: {risk_str}
""".strip()

if __name__ == "__main__":
    # Example usage
    database_path = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\strategy_database"
    print(f"Loading backtests from: {database_path}")
    print("-" * 80)
    
    backtests = load_all_backtests(database_path)
    
    # Group backtests by strategy
    strategy_groups = {}
    for backtest in backtests:
        strategy = backtest.strategy_name
        if strategy not in strategy_groups:
            strategy_groups[strategy] = []
        strategy_groups[strategy].append(backtest)
    
    print(f"\nLoaded {len(backtests)} backtest results across {len(strategy_groups)} strategies:")
    print("-" * 80)
    
    for strategy, tests in strategy_groups.items():
        print(f"\n{strategy}:")
        print(f"  Number of backtests: {len(tests)}")
        if tests:
            example = tests[0]
            print(f"  Symbol: {example.symbol}")
            print(f"  Timeframe: {example.timeframe}")
            print(f"  Average Return: {sum(t.total_return_pct for t in tests)/len(tests):.2f}%")
            print(f"  Average Win Rate: {sum(t.win_rate_pct for t in tests)/len(tests):.2f}%")
    
    print("\nExample text representation for embedding:")
    print("-" * 80)
    if backtests:
        print(generate_text_for_embedding(backtests[0]))
    
    print("\nData is ready for embedding generation.")
    