"""
Trading Strategy Retrieval System.

This module provides functionality to search and retrieve similar trading strategies
using semantic similarity and performance-based filtering. It implements a sophisticated
retrieval system that combines:
1. Vector similarity search using Qdrant
2. Performance metric filtering
3. Strategy type matching
4. Natural language query parsing

Key Features:
- Semantic search using OpenAI embeddings
- Flexible filtering based on performance metrics
- Natural language query understanding
- Rich result formatting and presentation
- Automatic query parameter extraction

The system supports various types of queries:
- Strategy type specific (e.g., "find mean reversion strategies")
- Performance based (e.g., "high Sharpe ratio strategies")
- Risk focused (e.g., "low drawdown strategies")
- Market condition specific (e.g., "strategies for volatile markets")
- Technical indicator based (e.g., "strategies using RSI")

Usage:
    from retrieve_strategies import search_similar_strategies
    
    # Simple search
    results = search_similar_strategies("profitable mean reversion strategies")
    
    # Search with filters
    results = search_similar_strategies(
        "high win rate strategies",
        filters={"min_win_rate": 60.0, "max_drawdown": 20.0}
    )
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dataclasses import dataclass
import traceback

# Load API key from APIkeys.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from APIkeys import OpenAI_api_key

# Configure OpenAI client
client = OpenAI(api_key=OpenAI_api_key)

@dataclass
class SearchResult:
    """
    Container for strategy search results with similarity scores and metadata.
    
    This dataclass encapsulates all relevant information about a retrieved strategy,
    including its similarity score to the query and key performance metrics.
    
    Attributes:
        Core Information:
            strategy_name (str): Name of the trading strategy
            strategy_type (str): Category/type of the strategy
            similarity_score (float): Cosine similarity to the query (0-1)
            
        Performance Metrics:
            final_equity (float): Final account balance
            total_return_pct (float): Total return percentage
            win_rate_pct (float): Win rate percentage
            max_drawdown_pct (float): Maximum drawdown percentage
            sharpe_ratio (float): Sharpe ratio
            profit_factor (float): Profit factor
            expectancy (float): Average expected return per trade
            total_trades (int): Total number of trades
            
        Additional Info:
            file_path (str): Path to the source backtest file
            text_content (str): Detailed strategy description
    """
    strategy_name: str
    strategy_type: str
    similarity_score: float
    final_equity: float
    total_return_pct: float
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    expectancy: float
    total_trades: int
    file_path: str
    text_content: str

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the search query using OpenAI's API.
    
    Args:
        text (str): The search query text to embed
        
    Returns:
        List[float]: The embedding vector (1536 dimensions)
        
    Raises:
        Exception: If the API call fails
        
    Note:
        Uses the same model (text-embedding-ada-002) as the strategy embeddings
        to ensure compatibility in the vector space.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def connect_to_qdrant() -> QdrantClient:
    """
    Connect to the local Qdrant vector database.
    
    Returns:
        QdrantClient: Configured Qdrant client
        
    Note:
        Uses a local Qdrant database stored in the scripts directory.
        The database should have been initialized by the embedding process.
    """
    return QdrantClient(path=r"F:\Algo Trading TRAINING\Strategy backtests\RAG\scripts\qdrant_storage")

def search_similar_strategies(
    query: str,
    n_results: int = 5,
    min_score: float = 0.7,
    filters: Dict[str, Any] = None
) -> List[SearchResult]:
    """
    Search for similar trading strategies based on a text query with optional filters.
    
    This function combines semantic similarity search with metric-based filtering
    to find the most relevant trading strategies. It supports various filter types
    including performance thresholds and strategy type matching.
    
    Args:
        query (str): Text description of what to search for
        n_results (int, optional): Maximum number of results to return. Defaults to 5.
        min_score (float, optional): Minimum similarity threshold (0-1). Defaults to 0.7.
        filters (Dict[str, Any], optional): Additional filters to apply, such as:
            {
                "min_sharpe": 1.0,
                "min_profit_factor": 1.5,
                "max_drawdown": 25.0,
                "min_win_rate": 50.0,
                "strategy_type": "mean_reversion"
            }
    
    Returns:
        List[SearchResult]: List of matching strategies with similarity scores
        
    Note:
        The function first performs semantic similarity search, then applies
        any additional filters to the results. Both the query and filters
        contribute to the final ranking of results.
    """
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Connect to Qdrant
    qdrant = connect_to_qdrant()
    
    # Prepare filter conditions if any
    filter_conditions = None
    if filters:
        must_conditions = []
        if "min_sharpe" in filters:
            must_conditions.append(
                models.FieldCondition(key="sharpe_ratio", 
                                    range=models.Range(gte=filters["min_sharpe"]))
            )
        if "min_profit_factor" in filters:
            must_conditions.append(
                models.FieldCondition(key="profit_factor", 
                                    range=models.Range(gte=filters["min_profit_factor"]))
            )
        if "max_drawdown" in filters:
            must_conditions.append(
                models.FieldCondition(key="max_drawdown_pct", 
                                    range=models.Range(lte=filters["max_drawdown"]))
            )
        if "min_win_rate" in filters:
            must_conditions.append(
                models.FieldCondition(key="win_rate_pct", 
                                    range=models.Range(gte=filters["min_win_rate"]))
            )
        if "strategy_type" in filters:
            must_conditions.append(
                models.FieldCondition(key="strategy_type", 
                                    match=models.MatchValue(value=filters["strategy_type"]))
            )
        if must_conditions:
            filter_conditions = models.Filter(must=must_conditions)
    
    # Search for similar vectors with filters
    search_results = qdrant.search(
        collection_name="strategy_embeddings",
        query_vector=query_embedding,
        limit=n_results,
        score_threshold=min_score,
        query_filter=filter_conditions
    )
    
    # Convert results to SearchResult objects
    results = []
    for hit in search_results:
        result = SearchResult(
            strategy_name=hit.payload["strategy_name"],
            strategy_type=hit.payload["strategy_type"],
            similarity_score=hit.score,
            final_equity=hit.payload["final_equity"],
            total_return_pct=hit.payload["total_return_pct"],
            win_rate_pct=hit.payload["win_rate_pct"],
            max_drawdown_pct=hit.payload["max_drawdown_pct"],
            sharpe_ratio=hit.payload["sharpe_ratio"],
            profit_factor=hit.payload["profit_factor"],
            expectancy=hit.payload["expectancy"],
            total_trades=hit.payload["total_trades"],
            file_path=hit.payload["file_path"],
            text_content=hit.payload["text_content"]
        )
        results.append(result)
    
    return results

def parse_query_filters(query: str) -> Dict[str, Any]:
    """
    Extract filter conditions from natural language query text.
    
    This function analyzes the query text to identify implied filters based on
    common phrases and requirements. It supports both strategy type filtering
    and performance metric thresholds.
    
    Args:
        query (str): The natural language query text
        
    Returns:
        Dict[str, Any]: Dictionary of extracted filters
        
    Example:
        >>> parse_query_filters("find profitable mean reversion strategies with high Sharpe")
        {
            'strategy_type': 'mean_reversion',
            'min_profit_factor': 1.5,
            'min_sharpe': 1.5
        }
    """
    filters = {}
    
    # Strategy type filters
    if "mean reversion" in query.lower():
        filters["strategy_type"] = "mean_reversion"
    elif "trend following" in query.lower():
        filters["strategy_type"] = "trend_following"
    elif "momentum" in query.lower():
        filters["strategy_type"] = "momentum"
    
    # Performance filters
    if "high sharpe" in query.lower() or "good sharpe" in query.lower():
        filters["min_sharpe"] = 1.5
    if "profitable" in query.lower() or "high profit" in query.lower():
        filters["min_profit_factor"] = 1.5
    if "low drawdown" in query.lower() or "safe" in query.lower():
        filters["max_drawdown"] = 20.0
    if "high win rate" in query.lower():
        filters["min_win_rate"] = 60.0
    
    return filters

def format_search_results(results: List[SearchResult]) -> str:
    """
    Format search results into a readable string representation.
    
    Creates a detailed text report of the search results, including:
    - Similarity scores
    - Strategy details
    - Performance metrics
    - Strategy descriptions
    
    Args:
        results (List[SearchResult]): List of search results to format
        
    Returns:
        str: Formatted string containing all result details
        
    Note:
        The output is formatted with clear sections and separators
        for easy reading and analysis.
    """
    output = []
    
    for i, result in enumerate(results, 1):
        output.append(f"\nResult {i} (Similarity: {result.similarity_score:.2%})")
        output.append(f"Strategy: {result.strategy_name} ({result.strategy_type})")
        output.append("\nPerformance Metrics:")
        output.append(f"- Total Return: {result.total_return_pct:.2f}%")
        output.append(f"- Win Rate: {result.win_rate_pct:.2f}%")
        output.append(f"- Sharpe Ratio: {result.sharpe_ratio:.2f}")
        output.append(f"- Max Drawdown: {result.max_drawdown_pct:.2f}%")
        output.append(f"- Profit Factor: {result.profit_factor:.2f}")
        output.append(f"- Expectancy: {result.expectancy:.2f}")
        output.append(f"- Total Trades: {result.total_trades}")
        output.append(f"\nStrategy Details:\n{result.text_content}")
        output.append("-" * 80)
    
    return "\n".join(output)

def main():
    """
    Interactive CLI for testing and using the strategy retrieval system.
    
    Provides a command-line interface to:
    - Test different types of queries
    - View example queries
    - Get formatted search results
    - Handle errors gracefully
    
    The interface continues running until the user types 'quit'.
    """
    print("Trading Strategy Retrieval System")
    print("\nAvailable example queries:")
    example_queries = [
        "Find profitable mean reversion strategies with good Sharpe ratio",
        "Show me trend following strategies that work well in volatile markets",
        "Find strategies with high win rate and low drawdown",
        "Show me strategies that use RSI and MACD indicators",
        "Find strategies that perform well on EURUSD"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    while True:
        print("\nEnter your query (or 'quit' to exit):")
        query = input("> ").strip()
        
        if query.lower() == 'quit':
            break
        
        try:
            results = search_similar_strategies(query)
            if results:
                print("\nFound similar strategies:")
                print(format_search_results(results))
            else:
                print("\nNo similar strategies found. Try a different query.")
        except Exception as e:
            print(f"\nError during search: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()