"""
Trading Strategy Backtest Parser and Text Generator.

This module handles the parsing and processing of trading strategy backtest results from JSON files.
It provides functionality to:
1. Load and validate backtest results
2. Structure the data into strongly-typed objects
3. Generate rich textual descriptions for embedding generation
4. Filter out invalid or empty backtest results

The module is designed to work with a specific JSON format that includes:
- Strategy metadata (name, type, description)
- Performance metrics (returns, drawdowns, ratios)
- Trade details
- Risk management settings
- Parameter configurations

Key Components:
- BacktestResult: Dataclass for structured backtest data
- load_backtest: Single file parser
- load_all_backtests: Bulk file loader
- generate_text_for_embedding: Text representation generator

Usage:
    from parse_backtests import load_all_backtests, generate_text_for_embedding
    
    # Load all valid backtests
    backtests = load_all_backtests("path/to/database")
    
    # Generate text for embedding
    for backtest in backtests:
        text = generate_text_for_embedding(backtest)
"""

import json
import glob
import os
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BacktestResult:
    """
    Structured representation of a trading strategy backtest result.
    
    This dataclass encapsulates all relevant information about a strategy backtest,
    providing strong typing and clear data organization.
    
    Attributes:
        Core Information:
            timestamp (str): Timestamp of the backtest execution
            strategy_name (str): Name of the trading strategy
            strategy_type (str): Category/type of the strategy (e.g., "Mean Reversion")
            strategy_description (str): Detailed description of the strategy logic
        
        Market Data:
            symbol (str): Trading pair or instrument symbol
            timeframe (str): Timeframe of the price data
            start_date (str): Start date of the backtest period
            end_date (str): End date of the backtest period
        
        Performance Metrics:
            final_equity (float): Final account balance
            total_return_pct (float): Total return percentage
            buy_hold_return_pct (float): Buy & hold return percentage
            max_drawdown_pct (float): Maximum drawdown percentage
            sharpe_ratio (float): Sharpe ratio
            profit_factor (float): Profit factor
            win_rate_pct (float): Win rate percentage
            expectancy (float): Average expected return per trade
            total_trades (int): Total number of trades executed
            exposure_time_pct (float): Percentage of time in market
        
        Configuration:
            parameters (Dict[str, Any]): Strategy parameters
            risk_management (Dict[str, str]): Risk management settings
            file_path (str): Path to the source JSON file
    """
    
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
    risk_management: Dict[str, str]
    file_path: str

def load_backtest(file_path: str) -> BacktestResult:
    """
    Load and validate a single backtest JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing backtest results
        
    Returns:
        BacktestResult: Structured backtest data
        
    Raises:
        ValueError: If the backtest has zero trades or empty trades list
        Exception: For any other loading or parsing errors
        
    Note:
        This function performs validation to ensure the backtest contains actual trades.
        Backtests with zero trades are considered invalid and are skipped.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Skip strategies with zero trades
    if data['performance']['total_trades'] == 0 or not data.get('trades'):
        raise ValueError(f"Skipping strategy with zero trades: {file_path}")
    
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
    """
    Load all valid JSON backtest files from the specified folder and its subfolders.
    
    This function walks through the directory tree, attempting to load each JSON file
    as a backtest result. It handles errors gracefully and provides feedback about
    skipped or failed files.
    
    Args:
        folder_path (str): Root directory containing backtest JSON files
        
    Returns:
        List[BacktestResult]: List of successfully loaded backtest results
        
    Note:
        - Files with zero trades are skipped but logged
        - Other loading errors are reported but don't halt execution
        - A summary of skipped files is printed at the end
    """
    backtests = []
    skipped_files = []
    
    # Get all JSON files in all subdirectories
    json_pattern = os.path.join(folder_path, "**/*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    print(f"Found {len(json_files)} JSON files")
    
    for file_path in json_files:
        try:
            backtest = load_backtest(file_path)
            backtests.append(backtest)
        except ValueError as ve:
            # This is for expected skips (zero trades)
            skipped_files.append((file_path, str(ve)))
        except Exception as e:
            # This is for unexpected errors
            print(f"Error loading {file_path}: {str(e)}")
    
    # Print summary of skipped files
    if skipped_files:
        print("\nSkipped files:")
        for file_path, reason in skipped_files:
            print(f"- {os.path.basename(file_path)}: {reason}")
    
    return backtests

def generate_text_for_embedding(backtest: BacktestResult) -> str:
    """
    Generate a detailed text representation of the backtest for embedding.
    
    This function creates a rich textual description of the trading strategy and its
    performance, optimized for semantic search and analysis. It includes:
    - Core strategy information and logic
    - Market context and conditions
    - Performance metrics with interpretation
    - Risk metrics and analysis
    - Technical setup details
    - Risk management insights
    
    The text is structured to maximize semantic relevance while minimizing
    boilerplate content that could dilute the embedding's effectiveness.
    
    Args:
        backtest (BacktestResult): The backtest result to describe
        
    Returns:
        str: A detailed, structured text description of the strategy
        
    Note:
        The text is formatted in a way that emphasizes unique characteristics
        and important metrics while maintaining readability and semantic richness.
    """
    # Calculate derived metrics
    risk_adjusted_return = backtest.total_return_pct / (backtest.max_drawdown_pct if backtest.max_drawdown_pct != 0 else 1)
    avg_trade_return = backtest.total_return_pct / backtest.total_trades if backtest.total_trades > 0 else 0
    
    # Format parameters and risk management settings
    params_list = [f"{k}={v}" for k, v in backtest.parameters.items()]
    risk_list = [f"{k}: {v}" for k, v in backtest.risk_management.items()]
    
    # Build the text representation with minimal boilerplate
    text_parts = []
    
    # Core strategy information
    text_parts.append(f"{backtest.strategy_name} - {backtest.strategy_type}")
    text_parts.append(f"{backtest.strategy_description}")
    
    # Market context
    text_parts.append(
        f"Trading {backtest.symbol} on {backtest.timeframe} timeframe from {backtest.start_date} to {backtest.end_date}. "
        f"Market was {'bullish' if backtest.buy_hold_return_pct > 0 else 'bearish'} with {backtest.buy_hold_return_pct:.1f}% buy & hold return."
    )
    
    # Key performance metrics with context
    text_parts.append(
        f"Strategy achieved {backtest.total_return_pct:.1f}% return ({risk_adjusted_return:.2f} risk-adjusted) "
        f"with {backtest.win_rate_pct:.1f}% win rate across {backtest.total_trades} trades. "
        f"Average trade return: {avg_trade_return:.2f}%."
    )
    
    # Risk metrics with interpretation
    text_parts.append(
        f"Risk profile: Sharpe {backtest.sharpe_ratio:.2f}, "
        f"max drawdown {backtest.max_drawdown_pct:.1f}%, "
        f"profit factor {backtest.profit_factor:.2f}, "
        f"expectancy {backtest.expectancy:.3f}. "
        f"Market exposure: {backtest.exposure_time_pct:.1f}% of time."
    )
    
    # Strategy configuration
    text_parts.append(f"Parameters: {', '.join(params_list)}")
    text_parts.append(f"Risk management: {', '.join(risk_list)}")
    
    # Strategy characteristics and classification
    characteristics = []
    
    # Classify trading frequency
    if backtest.total_trades > 1000:
        characteristics.append("high-frequency trading")
    elif backtest.total_trades > 100:
        characteristics.append("medium-frequency trading")
    else:
        characteristics.append("low-frequency trading")
    
    # Classify win rate
    if backtest.win_rate_pct > 65:
        characteristics.append("high win rate")
    elif backtest.win_rate_pct < 45:
        characteristics.append("low win rate")
    
    # Classify risk-adjusted performance
    if backtest.sharpe_ratio > 2:
        characteristics.append("strong risk-adjusted performance")
    elif backtest.sharpe_ratio < 1:
        characteristics.append("poor risk-adjusted performance")
    
    # Classify drawdown risk
    if backtest.max_drawdown_pct < 10:
        characteristics.append("excellent drawdown control")
    elif backtest.max_drawdown_pct > 25:
        characteristics.append("significant drawdown risk")
    
    # Classify profitability
    if backtest.profit_factor > 2:
        characteristics.append("highly profitable")
    elif backtest.profit_factor < 1.2:
        characteristics.append("marginally profitable")
    
    # Classify market exposure
    if backtest.exposure_time_pct < 30:
        characteristics.append("selective entry")
    elif backtest.exposure_time_pct > 70:
        characteristics.append("high market exposure")
    
    text_parts.append(f"Strategy characteristics: {', '.join(characteristics)}")
    
    # Add parameter-based technical insights
    insights = []
    for param_name, param_value in backtest.parameters.items():
        if 'period' in param_name.lower():
            insights.append(f"Uses {param_value} period for {param_name}")
        elif 'threshold' in param_name.lower():
            insights.append(f"{param_name} set at {param_value}")
        elif any(indicator in param_name.lower() for indicator in ['rsi', 'macd', 'bb', 'ema', 'sma', 'atr']):
            insights.append(f"Incorporates {param_name} indicator")
    
    if insights:
        text_parts.append(f"Technical setup: {', '.join(insights)}")
    
    # Add risk management insights
    risk_insights = []
    for setting, value in backtest.risk_management.items():
        if 'position_sizing' in setting.lower():
            risk_insights.append(f"Position sizing: {value}")
        elif 'stop_loss' in setting.lower():
            risk_insights.append(f"Stop loss: {value}")
        elif 'take_profit' in setting.lower():
            risk_insights.append(f"Take profit: {value}")
    
    if risk_insights:
        text_parts.append(f"Risk controls: {', '.join(risk_insights)}")
    
    return "\n".join(text_parts)

if __name__ == "__main__":
    # Example usage and diagnostic output
    database_path = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\strategy_database"
    print(f"Loading backtests from: {database_path}")
    print("-" * 80)
    
    backtests = load_all_backtests(database_path)
    
    # Group backtests by strategy for analysis
    strategy_groups = {}
    for backtest in backtests:
        strategy = backtest.strategy_name
        if strategy not in strategy_groups:
            strategy_groups[strategy] = []
        strategy_groups[strategy].append(backtest)
    
    # Print summary statistics
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
    
    # Show example text representation
    print("\nExample text representation for embedding:")
    print("-" * 80)
    if backtests:
        print(generate_text_for_embedding(backtests[0]))
    
    print("\nData is ready for embedding generation.")
    