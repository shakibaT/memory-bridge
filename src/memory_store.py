import chromadb
from src.data_models import Memory
from typing import List, Optional
import datetime

class MemoryStore:
    """
    A wrapper class for our vector database (ChromaDB).
    It handles all database read, write, and update operations.
    """
    def __init__(self, client: chromadb.Client, embedding_dimension: int = 384):
        self.client = client
        self.embedding_dimension = embedding_dimension
        
        # Try to get existing collection first
        try:
            self.collection = self.client.get_collection(name="memories")
            print("INFO: Connected to existing collection 'memories'.")
        except:
            # Collection doesn't exist, create new one with correct dimensions
            self.collection = self.client.create_collection(
                name="memories",
                metadata={"hnsw:space": "cosine"}  # or "l2" or "ip"
            )
            print(f"INFO: Created new collection 'memories' for {embedding_dimension}D embeddings.")
        
        # Check if existing collection has compatible dimensions
        self._validate_collection_dimensions()

    def _validate_collection_dimensions(self):
        """
        Validates that the collection can handle our embedding dimensions.
        Only validates if there are existing embeddings in the collection.
        """
        try:
            # Check if collection has any data first
            result = self.collection.peek(limit=1)
            if not result['ids']:  # Empty collection
                print(f"INFO: Empty collection - ready for {self.embedding_dimension}D embeddings.")
                return
            
            # If collection has data, try a test query to validate dimensions
            test_embedding = [[0.0] * self.embedding_dimension]
            self.collection.query(query_embeddings=test_embedding, n_results=1)
            print(f"INFO: Collection validated for {self.embedding_dimension}D embeddings.")
            
        except Exception as e:
            if "dimension" in str(e).lower():
                print(f"WARNING: Dimension mismatch detected: {e}")
                print("INFO: Resetting collection to match new embedding dimensions...")
                self._reset_collection()
            else:
                # For other errors, just log and continue (might be empty collection)
                print(f"INFO: Collection validation skipped: {e}")
                print(f"INFO: Assuming collection is ready for {self.embedding_dimension}D embeddings.")

    def _reset_collection(self):
        """
        Deletes and recreates the collection with correct dimensions.
        WARNING: This will delete all existing data!
        """
        try:
            self.client.delete_collection(name="memories")
            print("INFO: Deleted existing collection 'memories'.")
        except ValueError:
            pass  # Collection might not exist
        
        self.collection = self.client.create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"INFO: Created new collection 'memories' for {self.embedding_dimension}D embeddings.")

    def _memory_to_chroma_format(self, memory: Memory, embeddings: Optional[List[float]] = None):
        """
        Converts and sanitizes our Pydantic Memory model for ChromaDB.
        Now includes embeddings parameter.
        """
        metadata = memory.dict(exclude={"content"})

        # 1. Convert unsupported types to supported types (e.g., datetime to string)
        if 'timestamp' in metadata and isinstance(metadata['timestamp'], datetime.datetime):
            metadata['timestamp'] = metadata['timestamp'].isoformat()

        # 2. Remove any keys that have a value of `None`.
        # ChromaDB does not support `None` as a metadata value.
        sanitized_metadata = {
            key: value for key, value in metadata.items() if value is not None
        }

        chroma_object = {
            "ids": [memory.fact_id],
            "documents": [memory.content],
            "metadatas": [sanitized_metadata]
        }
        
        # Add embeddings if provided
        if embeddings is not None:
            if len(embeddings) != self.embedding_dimension:
                raise ValueError(f"Expected embedding dimension {self.embedding_dimension}, got {len(embeddings)}")
            chroma_object["embeddings"] = [embeddings]
        
        return chroma_object

    def write(self, memory: Memory, embeddings: Optional[List[float]] = None):
        """
        Write a memory to the store. 
        If embeddings are provided, they will be used. Otherwise, ChromaDB will generate them.
        """
        chroma_object = self._memory_to_chroma_format(memory, embeddings)
        self.collection.add(**chroma_object)
        print(f"INFO: Wrote memory '{memory.fact_id}' to store.")

    def update(self, fact_id: str, new_memory: Memory, embeddings: Optional[List[float]] = None):
        """
        Update/upsert a memory in the store.
        If embeddings are provided, they will be used. Otherwise, ChromaDB will generate them.
        """
        chroma_object = self._memory_to_chroma_format(new_memory, embeddings)
        self.collection.upsert(**chroma_object)
        print(f"INFO: Upserted memory '{fact_id}' in store.")

    def retrieve_by_id(self, fact_id: str):
        return self.collection.get(ids=[fact_id])

    def search(self, query_embeddings: List[List[float]], top_k: int = 3):
        # Validate query embedding dimensions
        for embedding in query_embeddings:
            if len(embedding) != self.embedding_dimension:
                raise ValueError(f"Query embedding dimension {len(embedding)} doesn't match collection dimension {self.embedding_dimension}")
        
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )