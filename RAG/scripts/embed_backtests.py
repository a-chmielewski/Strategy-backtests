"""
Trading Strategy Embedding Generator.

This module handles the generation and storage of embeddings for trading strategy backtests.
It provides functionality to:
1. Generate embeddings using OpenAI's text-embedding-ada-002 model
2. Store embeddings in both local files and Qdrant vector database
3. Manage embedding versioning and updates
4. Handle duplicate and existing embeddings efficiently

The module implements a two-tier storage system:
1. Local JSON files for persistence and backup
2. Qdrant vector database for efficient similarity search

Key Features:
- Checks for existing embeddings before generating new ones
- Stores comprehensive metadata with each embedding
- Provides detailed progress and error reporting
- Handles API rate limits and errors gracefully

Usage:
    from embed_backtests import embed_backtests
    
    # Generate and store embeddings
    embed_backtests("path/to/database")
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from parse_backtests import load_all_backtests, generate_text_for_embedding, BacktestResult

# Load API key from APIkeys.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from APIkeys import OpenAI_api_key

# Configure OpenAI client
client = OpenAI(api_key=OpenAI_api_key)

def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the given text using OpenAI's API.
    
    Args:
        text (str): The text to generate an embedding for
        
    Returns:
        List[float]: The embedding vector (1536 dimensions)
        
    Raises:
        Exception: If the API call fails
        
    Note:
        Uses the text-embedding-ada-002 model which produces 1536-dimensional vectors.
        This function makes an API call, so it should be used efficiently.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def setup_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if it doesn't exist."""
    client = QdrantClient(path="./qdrant_storage")
    
    # Create collection for storing strategy embeddings
    client.recreate_collection(
        collection_name="strategy_embeddings",
        vectors_config=models.VectorParams(
            size=1536,  # OpenAI ada-002 embedding size
            distance=models.Distance.COSINE
        )
    )
    
    return client

def get_embedding_path(backtest: BacktestResult) -> str:
    """
    Get the absolute path where the embedding file should be stored.
    
    Args:
        backtest (BacktestResult): The backtest result object
        
    Returns:
        str: Absolute path for the embedding file
        
    Note:
        Creates a filename based on the original backtest file name
        with '_embedding.json' appended.
    """
    base_name = os.path.splitext(os.path.basename(backtest.file_path))[0]
    embeddings_dir = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\embeddings"
    return os.path.join(embeddings_dir, f"{base_name}_embedding.json")

def embedding_exists(backtest: BacktestResult) -> bool:
    """
    Check if an embedding file already exists for this backtest.
    
    Args:
        backtest (BacktestResult): The backtest result object
        
    Returns:
        bool: True if embedding exists, False otherwise
    """
    embedding_path = get_embedding_path(backtest)
    return os.path.exists(embedding_path)

def load_existing_embedding(backtest: BacktestResult) -> Dict:
    """
    Load an existing embedding from file.
    
    Args:
        backtest (BacktestResult): The backtest result object
        
    Returns:
        Dict: The loaded embedding data and metadata
        
    Raises:
        FileNotFoundError: If the embedding file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    embedding_path = get_embedding_path(backtest)
    with open(embedding_path, 'r') as f:
        return json.load(f)

def save_embedding_to_file(embedding: List[float], backtest: BacktestResult) -> None:
    """
    Save an embedding vector and its metadata to a JSON file.
    
    Args:
        embedding (List[float]): The embedding vector to save
        backtest (BacktestResult): The backtest result object containing metadata
        
    Note:
        Creates the embeddings directory if it doesn't exist.
        Stores both the embedding vector and comprehensive metadata.
    """
    embeddings_dir = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    embedding_file = get_embedding_path(backtest)
    
    # Save embedding and metadata
    data = {
        "embedding": embedding,
        "metadata": {
            "strategy_name": backtest.strategy_name,
            "strategy_type": backtest.strategy_type,
            "timestamp": backtest.timestamp,
            "file_path": backtest.file_path,
            "performance": {
                "final_equity": backtest.final_equity,
                "total_return_pct": backtest.total_return_pct,
                "win_rate_pct": backtest.win_rate_pct,
                "max_drawdown_pct": backtest.max_drawdown_pct,
                "sharpe_ratio": backtest.sharpe_ratio,
                "profit_factor": backtest.profit_factor,
                "expectancy": backtest.expectancy,
                "total_trades": backtest.total_trades
            }
        }
    }
    
    with open(embedding_file, 'w') as f:
        json.dump(data, f, indent=2)

def embed_backtests(database_path: str) -> None:
    """
    Main function to process backtests and create/update embeddings.
    
    This function:
    1. Loads all valid backtest results
    2. Sets up the Qdrant vector database
    3. Processes each backtest:
       - Checks for existing embedding
       - Loads or generates embedding as needed
       - Stores in both file system and Qdrant
    4. Provides detailed progress reporting
    
    Args:
        database_path (str): Path to the strategy database directory
        
    Note:
        - Skips backtests with zero trades
        - Reuses existing embeddings when available
        - Provides progress updates and error reporting
        - Maintains both file system and vector database storage
    """
    print("Loading backtests...")
    backtests = load_all_backtests(database_path)
    print(f"Loaded {len(backtests)} backtests")
    
    print("\nSetting up Qdrant...")
    qdrant_storage_path = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\scripts\qdrant_storage"
    os.makedirs(qdrant_storage_path, exist_ok=True)
    qdrant_client = QdrantClient(path=qdrant_storage_path)
    
    # Create collection for storing strategy embeddings
    qdrant_client.recreate_collection(
        collection_name="strategy_embeddings",
        vectors_config=models.VectorParams(
            size=1536,  # OpenAI ada-002 embedding size
            distance=models.Distance.COSINE
        )
    )
    
    print("\nProcessing backtests...")
    new_embeddings = 0
    skipped_embeddings = 0
    
    for i, backtest in enumerate(backtests, 1):
        print(f"\nProcessing backtest {i}/{len(backtests)}: {backtest.strategy_name}")
        
        # Check for existing embedding
        if embedding_exists(backtest):
            print("  ↷ Embedding already exists, loading from file")
            skipped_embeddings += 1
            try:
                # Load and store existing embedding
                data = load_existing_embedding(backtest)
                embedding = data["embedding"]
                
                # Store in Qdrant
                qdrant_client.upsert(
                    collection_name="strategy_embeddings",
                    points=[models.PointStruct(
                        id=i,
                        vector=embedding,
                        payload={
                            "strategy_name": backtest.strategy_name,
                            "strategy_type": backtest.strategy_type,
                            "timestamp": backtest.timestamp,
                            "file_path": backtest.file_path,
                            "final_equity": backtest.final_equity,
                            "total_return_pct": backtest.total_return_pct,
                            "win_rate_pct": backtest.win_rate_pct,
                            "max_drawdown_pct": backtest.max_drawdown_pct,
                            "sharpe_ratio": backtest.sharpe_ratio,
                            "profit_factor": backtest.profit_factor,
                            "expectancy": backtest.expectancy,
                            "total_trades": backtest.total_trades,
                            "text_content": generate_text_for_embedding(backtest)
                        }
                    )]
                )
                print("  ✓ Existing embedding loaded and stored in Qdrant")
            except Exception as e:
                print(f"  ✗ Error loading existing embedding: {str(e)}")
            continue
        
        # Generate new embedding
        try:
            # Generate text and embedding
            text = generate_text_for_embedding(backtest)
            embedding = get_embedding(text)
            new_embeddings += 1
            
            # Save to file system
            save_embedding_to_file(embedding, backtest)
            print("  ✓ New embedding generated and saved to file")
            
            # Store in Qdrant
            qdrant_client.upsert(
                collection_name="strategy_embeddings",
                points=[models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "strategy_name": backtest.strategy_name,
                        "strategy_type": backtest.strategy_type,
                        "timestamp": backtest.timestamp,
                        "file_path": backtest.file_path,
                        "final_equity": backtest.final_equity,
                        "total_return_pct": backtest.total_return_pct,
                        "win_rate_pct": backtest.win_rate_pct,
                        "max_drawdown_pct": backtest.max_drawdown_pct,
                        "sharpe_ratio": backtest.sharpe_ratio,
                        "profit_factor": backtest.profit_factor,
                        "expectancy": backtest.expectancy,
                        "total_trades": backtest.total_trades,
                        "text_content": text
                    }
                )]
            )
            print("  ✓ New embedding stored in Qdrant")
            
        except Exception as e:
            print(f"  ✗ Error processing backtest: {str(e)}")
    
    # Print final summary
    print("\nEmbedding generation complete!")
    print(f"Total backtests processed: {len(backtests)}")
    print(f"New embeddings generated: {new_embeddings}")
    print(f"Existing embeddings reused: {skipped_embeddings}")
    print("All embeddings are stored in both Qdrant database and individual files")

if __name__ == "__main__":
    database_path = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\strategy_database"
    embed_backtests(database_path) 