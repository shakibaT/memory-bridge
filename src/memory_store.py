# src/memory_store.py

import chromadb
from src.data_models import Memory
from typing import List
import datetime

class MemoryStore:
    """
    A wrapper class for our vector database (ChromaDB).
    It handles all database read, write, and update operations.
    """
    def __init__(self, client: chromadb.Client):
        self.client = client
        self.collection = self.client.get_or_create_collection(name="memories")
        print("INFO: MemoryStore initialized and connected to collection 'memories'.")

    def _memory_to_chroma_format(self, memory: Memory):
        """
        Converts and sanitizes our Pydantic Memory model for ChromaDB.
        *** THIS METHOD CONTAINS THE FIX. ***
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

        return {
            "ids": [memory.fact_id],
            "documents": [memory.content],
            "metadatas": [sanitized_metadata] # Pass the clean, sanitized metadata
        }

    def write(self, memory: Memory):
        chroma_object = self._memory_to_chroma_format(memory)
        self.collection.add(**chroma_object)
        print(f"INFO: Wrote memory '{memory.fact_id}' to store.")

    def update(self, fact_id: str, new_memory: Memory):
        chroma_object = self._memory_to_chroma_format(new_memory)
        self.collection.upsert(**chroma_object)
        print(f"INFO: Upserted memory '{fact_id}' in store.")

    def retrieve_by_id(self, fact_id: str):
        return self.collection.get(ids=[fact_id])

    def search(self, query_embeddings: List[List[float]], top_k: int = 3):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
    