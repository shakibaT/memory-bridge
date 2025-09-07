import chromadb
from src.data_models import Memory

class MemoryStore:
    def __init__(self, db_path: str = "db/chroma.db"):
        # Using a persistent client so data is saved between runs
        self.client = chromadb.PersistentClient(path=db_path)
        # Collection is like a table in a traditional DB
        self.collection = self.client.get_or_create_collection(name="memories")

    def _memory_to_chroma_format(self, memory: Memory):
        """Converts our Pydantic model to the format ChromaDB expects."""
        return {
            "ids": [memory.fact_id],
            "documents": [memory.content],
            "metadatas": [memory.dict(exclude={"content"})] # Store everything else as metadata
        }

    def write(self, memory: Memory):
        """Writes a new memory to the store."""
        chroma_object = self._memory_to_chroma_format(memory)
        self.collection.add(**chroma_object)
        print(f"INFO: Wrote memory {memory.fact_id} to store.")

    def update(self, fact_id: str, new_memory: Memory):
        """Updates an existing memory."""
        chroma_object = self._memory_to_chroma_format(new_memory)
        # Chroma's 'upsert' can handle updates if the ID exists
        self.collection.upsert(**chroma_object)
        print(f"INFO: Updated memory {fact_id} in store.")

    def retrieve_by_id(self, fact_id: str):
        """Retrieves a memory by its unique ID."""
        return self.collection.get(ids=[fact_id])
        
    def search(self, query_embedding, top_k: int):
        """Performs a vector similarity search."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results