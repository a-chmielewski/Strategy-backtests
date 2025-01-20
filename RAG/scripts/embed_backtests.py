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

# Configure OpenAI
client = OpenAI(api_key=OpenAI_api_key)

def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
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
    """Get the path where the embedding file should be stored."""
    base_name = os.path.splitext(os.path.basename(backtest.file_path))[0]
    return os.path.join("../embeddings", f"{base_name}_embedding.json")

def embedding_exists(backtest: BacktestResult) -> bool:
    """Check if embedding already exists for this backtest."""
    embedding_path = get_embedding_path(backtest)
    return os.path.exists(embedding_path)

def load_existing_embedding(backtest: BacktestResult) -> Dict:
    """Load existing embedding from file."""
    embedding_path = get_embedding_path(backtest)
    with open(embedding_path, 'r') as f:
        return json.load(f)

def save_embedding_to_file(embedding: List[float], backtest: BacktestResult) -> None:
    """Save embedding to a file in the embeddings directory."""
    os.makedirs("../embeddings", exist_ok=True)
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
    """Main function to process backtests and create embeddings."""
    print("Loading backtests...")
    backtests = load_all_backtests(database_path)
    print(f"Loaded {len(backtests)} backtests")
    
    print("\nSetting up Qdrant...")
    qdrant_client = setup_qdrant()
    
    print("\nGenerating embeddings and storing in Qdrant...")
    new_embeddings = 0
    skipped_embeddings = 0
    
    for i, backtest in enumerate(backtests, 1):
        print(f"\nProcessing backtest {i}/{len(backtests)}: {backtest.strategy_name}")
        
        if embedding_exists(backtest):
            print("  ↷ Embedding already exists, loading from file")
            skipped_embeddings += 1
            try:
                # Load existing embedding
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
        
        # Generate text for embedding
        text = generate_text_for_embedding(backtest)
        
        try:
            # Get embedding from OpenAI
            embedding = get_embedding(text)
            new_embeddings += 1
            
            # Save to file as backup
            save_embedding_to_file(embedding, backtest)
            
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
            print("  ✓ New embedding generated and stored")
            
        except Exception as e:
            print(f"  ✗ Error processing backtest: {str(e)}")
    
    print("\nEmbedding generation complete!")
    print(f"Total backtests processed: {len(backtests)}")
    print(f"New embeddings generated: {new_embeddings}")
    print(f"Existing embeddings reused: {skipped_embeddings}")
    print("All embeddings are stored in both Qdrant database and individual files")

if __name__ == "__main__":
    database_path = r"F:\Algo Trading TRAINING\Strategy backtests\RAG\strategy_database"
    embed_backtests(database_path) 