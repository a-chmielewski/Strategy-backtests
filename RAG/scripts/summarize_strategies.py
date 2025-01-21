"""
Trading Strategy Analysis and Summarization System.

This module provides advanced natural language analysis and summarization of trading
strategies using OpenAI's GPT models. It combines retrieved strategy information
with sophisticated prompting to generate:
1. Detailed strategy analysis
2. Performance comparisons
3. Risk assessments
4. Market condition insights
5. Implementation recommendations

Key Features:
- Contextual strategy analysis
- Performance metric interpretation
- Risk-reward trade-off analysis
- Market condition sensitivity analysis
- Implementation guidance
- Cross-strategy comparison

The system uses carefully crafted prompts to ensure:
- Accurate technical analysis
- Balanced risk assessment
- Clear actionable insights
- Relevant comparisons
- Implementation considerations

Usage:
    from summarize_strategies import generate_strategy_summary
    
    # Generate summary for a single strategy
    summary = generate_strategy_summary(strategy_result)
    
    # Compare multiple strategies
    comparison = compare_strategies(strategy_results)
"""

import os
from typing import List, Dict, Any
from openai import OpenAI
from retrieve_strategies import SearchResult

# Load API key from APIkeys.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from APIkeys import OpenAI_api_key

# Configure OpenAI client
client = OpenAI(api_key=OpenAI_api_key)

def generate_strategy_summary(query: str, results: List[SearchResult]) -> str:
    """
    Generate a comprehensive natural language summary of trading strategies.
    
    This function analyzes the retrieved strategies in the context of the user's
    query, providing detailed insights about their performance, risks, and
    applicability to different market conditions.
    
    Args:
        query (str): The original search query that retrieved the strategies
        results (List[SearchResult]): List of retrieved strategy results to analyze
        
    Returns:
        str: A detailed analysis and summary of the strategies
        
    Note:
        The summary is structured to provide:
        - Overview of matching strategies
        - Performance analysis
        - Risk assessment
        - Market condition considerations
        - Implementation recommendations
    """
    # Prepare context for GPT
    context = []
    
    # Add query context
    context.append(f"Query: {query}")
    context.append("\nRetrieved Strategies:")
    
    # Add strategy details
    for i, result in enumerate(results, 1):
        context.append(f"\nStrategy {i}:")
        context.append(f"Name: {result.strategy_name}")
        context.append(f"Type: {result.strategy_type}")
        context.append(f"Similarity Score: {result.similarity_score:.2%}")
        context.append("\nPerformance Metrics:")
        context.append(f"- Total Return: {result.total_return_pct:.2f}%")
        context.append(f"- Win Rate: {result.win_rate_pct:.2f}%")
        context.append(f"- Sharpe Ratio: {result.sharpe_ratio:.2f}")
        context.append(f"- Max Drawdown: {result.max_drawdown_pct:.2f}%")
        context.append(f"- Profit Factor: {result.profit_factor:.2f}")
        context.append(f"- Total Trades: {result.total_trades}")
        context.append(f"\nStrategy Details:\n{result.text_content}")
    
    # Create analysis prompt
    prompt = f"""
    Analyze the following trading strategies in the context of this search query: {query}
    
    {'\n'.join(context)}
    
    Please provide a comprehensive analysis including:
    
    1. Strategy Overview:
    - Key characteristics of the matching strategies
    - Common patterns and differences
    - Relevance to the search query
    
    2. Performance Analysis:
    - Comparative performance metrics
    - Risk-adjusted returns
    - Trading frequency and efficiency
    
    3. Risk Assessment:
    - Drawdown analysis
    - Risk management approaches
    - Capital requirements
    
    4. Market Conditions:
    - Suitable market environments
    - Potential limitations
    - Adaptability considerations
    
    5. Implementation Insights:
    - Key parameters and settings
    - Technical requirements
    - Operational considerations
    
    6. Recommendations:
    - Best use cases
    - Optimization opportunities
    - Risk mitigation strategies
    """
    
    # Generate summary using GPT-4
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert trading strategy analyst specializing in algorithmic trading systems. Provide detailed, technical analysis with actionable insights."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def compare_strategies(strategies: List[SearchResult]) -> str:
    """
    Generate a detailed comparison between multiple trading strategies.
    
    This function performs a deep comparative analysis of multiple strategies,
    highlighting their relative strengths, weaknesses, and optimal use cases.
    
    Args:
        strategies (List[SearchResult]): List of strategies to compare
        
    Returns:
        str: Detailed comparison analysis
        
    Note:
        The comparison focuses on:
        - Performance metrics
        - Risk profiles
        - Market condition sensitivity
        - Implementation complexity
        - Resource requirements
    """
    # Prepare comparison context
    context = []
    
    for i, strategy in enumerate(strategies, 1):
        context.append(f"\nStrategy {i}: {strategy.strategy_name}")
        context.append(f"Type: {strategy.strategy_type}")
        context.append("\nKey Metrics:")
        context.append(f"- Return: {strategy.total_return_pct:.2f}%")
        context.append(f"- Sharpe: {strategy.sharpe_ratio:.2f}")
        context.append(f"- Win Rate: {strategy.win_rate_pct:.2f}%")
        context.append(f"- Drawdown: {strategy.max_drawdown_pct:.2f}%")
        context.append(f"- Trades: {strategy.total_trades}")
        context.append(f"\nDetails:\n{strategy.text_content}")
    
    # Create comparison prompt
    prompt = f"""
    Compare and contrast the following trading strategies:
    
    {'\n'.join(context)}
    
    Please provide a detailed comparison including:
    
    1. Performance Comparison:
    - Relative performance metrics
    - Risk-adjusted returns
    - Trading efficiency
    
    2. Risk Profile Analysis:
    - Comparative risk levels
    - Drawdown characteristics
    - Capital efficiency
    
    3. Market Condition Sensitivity:
    - Optimal market environments
    - Robustness across conditions
    - Adaptation requirements
    
    4. Implementation Considerations:
    - Relative complexity
    - Technical requirements
    - Operational overhead
    
    5. Recommendations:
    - Portfolio allocation suggestions
    - Combination opportunities
    - Risk management strategies
    """
    
    # Generate comparison using GPT-4
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert in quantitative trading strategy analysis. Provide detailed comparative analysis with practical insights for strategy selection and implementation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def analyze_market_conditions(strategy: SearchResult) -> str:
    """
    Analyze a strategy's performance under different market conditions.
    
    This function examines how a strategy performs across various market
    environments and provides insights into its adaptability and limitations.
    
    Args:
        strategy (SearchResult): The strategy to analyze
        
    Returns:
        str: Detailed market condition analysis
        
    Note:
        The analysis covers:
        - Trend sensitivity
        - Volatility response
        - Volume requirements
        - Seasonal patterns
        - Market regime adaptation
    """
    # Prepare market analysis context
    context = [
        f"Strategy: {strategy.strategy_name}",
        f"Type: {strategy.strategy_type}",
        f"\nPerformance Metrics:",
        f"- Total Return: {strategy.total_return_pct:.2f}%",
        f"- Sharpe Ratio: {strategy.sharpe_ratio:.2f}",
        f"- Win Rate: {strategy.win_rate_pct:.2f}%",
        f"- Max Drawdown: {strategy.max_drawdown_pct:.2f}%",
        f"- Profit Factor: {strategy.profit_factor:.2f}",
        f"- Total Trades: {strategy.total_trades}",
        f"\nStrategy Details:\n{strategy.text_content}"
    ]
    
    # Create market analysis prompt
    prompt = f"""
    Analyze the following trading strategy's performance across different market conditions:
    
    {'\n'.join(context)}
    
    Please provide a detailed market condition analysis including:
    
    1. Trend Analysis:
    - Performance in trending vs ranging markets
    - Directional bias assessment
    - Trend sensitivity metrics
    
    2. Volatility Response:
    - Behavior in high/low volatility
    - Volatility threshold effects
    - Risk adjustment mechanisms
    
    3. Volume Considerations:
    - Volume requirements
    - Liquidity sensitivity
    - Trading costs impact
    
    4. Market Regime Analysis:
    - Performance across different regimes
    - Regime change adaptation
    - Stability characteristics
    
    5. Recommendations:
    - Optimal market conditions
    - Risk management adjustments
    - Parameter adaptation guidelines
    """
    
    # Generate analysis using GPT-4
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are an expert in market microstructure and trading strategy analysis. Provide detailed insights into strategy behavior across different market conditions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    from retrieve_strategies import search_similar_strategies
    
    print("Trading Strategy Analysis System")
    print("\nEnter a query to analyze trading strategies:")
    query = input("> ").strip()
    
    try:
        # Search for strategies
        results = search_similar_strategies(query)
        
        if results:
            print("\nGenerating strategy analysis...")
            summary = generate_strategy_summary(query, results)
            print("\nStrategy Analysis:")
            print("-" * 80)
            print(summary)
            
            if len(results) > 1:
                print("\nGenerating strategy comparison...")
                comparison = compare_strategies(results)
                print("\nStrategy Comparison:")
                print("-" * 80)
                print(comparison)
            
            print("\nGenerating market condition analysis...")
            market_analysis = analyze_market_conditions(results[0])
            print("\nMarket Condition Analysis:")
            print("-" * 80)
            print(market_analysis)
        else:
            print("\nNo strategies found matching your query.")
            
    except Exception as e:
        print(f"\nError during analysis: {str(e)}") 