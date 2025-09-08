# init_chromadb.py

import chromadb
import os
from pathlib import Path

def initialize_chromadb(db_path: str = "./db"):
    """
    Initialize ChromaDB with a persistent client and create necessary directories.
    """
    # Create the database directory if it doesn't exist
    db_path = Path(db_path)
    db_path.mkdir(exist_ok=True)
    print(f"INFO: Created/verified database directory at: {db_path.absolute()}")
    
    # Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path=str(db_path))
    print(f"INFO: Initialized ChromaDB persistent client at: {db_path.absolute()}")
    
    # Optional: Create the memories collection immediately
    try:
        collection = client.create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
        print("INFO: Created 'memories' collection with cosine similarity.")
    except Exception as e:
        if "already exists" in str(e):
            print("INFO: Collection 'memories' already exists.")
        else:
            print(f"WARNING: Could not create collection: {e}")
    
    return client

def main():
    """
    Main initialization script
    """
    print("Initializing ChromaDB...")
    
    # Initialize the database
    client = initialize_chromadb()
    
    # Test the connection
    try:
        collections = client.list_collections()
        print(f"INFO: Successfully connected. Available collections: {[c.name for c in collections]}")
    except Exception as e:
        print(f"ERROR: Failed to connect to ChromaDB: {e}")
        return False
    
    print("INFO: ChromaDB initialization complete!")
    return True

if __name__ == "__main__":
    main()