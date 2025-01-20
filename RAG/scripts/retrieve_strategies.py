import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dataclasses import dataclass

# Load API key from APIkeys.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from APIkeys import OpenAI_api_key

# Configure OpenAI
client = OpenAI(api_key=OpenAI_api_key)

@dataclass
class SearchResult:
    """Class to hold search results with strategy details and similarity score."""
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
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def connect_to_qdrant() -> QdrantClient:
    """Connect to the local Qdrant database."""
    return QdrantClient(path="./qdrant_storage")

def search_similar_strategies(
    query: str,
    n_results: int = 5,
    min_score: float = 0.7
) -> List[SearchResult]:
    """
    Search for similar trading strategies based on a text query.
    
    Args:
        query: Text description to search for
        n_results: Number of results to return
        min_score: Minimum similarity score threshold (0-1)
        
    Returns:
        List of SearchResult objects containing similar strategies
    """
    # Get embedding for the query
    query_embedding = get_embedding(query)
    
    # Connect to Qdrant
    qdrant = connect_to_qdrant()
    
    # Search for similar vectors
    search_results = qdrant.search(
        collection_name="strategy_embeddings",
        query_vector=query_embedding,
        limit=n_results,
        score_threshold=min_score
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

def format_search_results(results: List[SearchResult]) -> str:
    """Format search results into a readable string."""
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
    """Example usage of the retrieval system."""
    # Example queries to test the system
    example_queries = [
        "Find profitable mean reversion strategies with good Sharpe ratio",
        "Show me trend following strategies that work well in volatile markets",
        "Find strategies with high win rate and low drawdown",
        "Show me strategies that use RSI and MACD indicators",
        "Find strategies that perform well on EURUSD"
    ]
    
    print("Trading Strategy Retrieval System")
    print("\nAvailable example queries:")
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

if __name__ == "__main__":
    main() 