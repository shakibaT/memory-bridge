from src.memory_store import MemoryStore

class BaselineRetriever:
    """
    A simple baseline that uses keyword matching instead of vector search.
    It does not use an LLM for extraction, instead treating every user turn as a potential memory.
    """
    def __init__(self, store: MemoryStore):
        self.store = store

    def process_turn(self, conversation_turn: dict, turn_index: int):
        """
        The baseline's 'extractor' is naive: it just saves the entire user turn as a memory.
        """
        if conversation_turn['role'] == 'user':
            # In a real baseline, we would create a Memory object, but for simplicity,
            # we will just add the document directly to the collection for searching.
            self.store.collection.add(
                ids=[f"turn_{turn_index}"],
                documents=[conversation_turn['content']]
            )

    def retrieve(self, query: str) -> list[str]:
        """
        Retrieves memories using simple keyword search.
        """
        all_memories = self.store.collection.get()
        query_words = set(query.lower().split())
        
        # Find all documents that contain any of the query words
        results = [
            doc for doc in all_memories['documents'] 
            if any(word in doc.lower() for word in query_words)
        ]
        
        # Return the top 3 matches (or fewer if not enough results)
        return results[:3]