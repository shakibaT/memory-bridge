# src/tools.py

from src.data_models import Memory
from src.memory_store import MemoryStore

# Note: The embedding logic is simplified for now. In a real implementation,
# this would involve calling the embedding model API.

def write_memory(store: MemoryStore, content: str, confidence: float, extracted_from: str):
    """Tool to write a new fact to the memory store."""
    print(f"TOOL: Executing write_memory with content: '{content}'")
    memory = Memory(
        content=content,
        confidence=confidence,
        extracted_from=extracted_from
    )
    store.write(memory)
    return f"Successfully wrote new memory: {content}"

def update_memory(store: MemoryStore, fact_id: str, new_content: str, confidence: float, extracted_from: str):
    """Tool to update an existing fact in the memory store."""
    print(f"TOOL: Executing update_memory for fact '{fact_id}' with new content: '{new_content}'")
    # In a real system, you'd fetch the old memory to populate 'previous_value'
    old_memory_data = store.retrieve_by_id(fact_id)
    previous_value = old_memory_data['documents'][0] if old_memory_data['ids'] else None
    
    new_memory = Memory(
        fact_id=fact_id,
        content=new_content,
        confidence=confidence,
        extracted_from=extracted_from,
        previous_value=previous_value
    )
    store.update(fact_id, new_memory)
    return f"Successfully updated fact {fact_id} to: {new_content}"

def read_memory(store: MemoryStore, query: str):
    """Tool to search for relevant memories based on a query."""
    print(f"TOOL: Executing read_memory with query: '{query}'")
    # Placeholder for embedding generation
    # In a real scenario: query_embedding = embedding_client.embed(query)
    dummy_embedding = [[0.1, 0.2, 0.3]] # This needs to be replaced
    
    results = store.search(query_embeddings=dummy_embedding, top_k=3)
    return f"Search results for '{query}': {results['documents']}"
