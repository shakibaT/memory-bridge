# src/tools.py

# ... (imports and schemas are the same)
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
class WriteMemoryArgs(BaseModel):
    content: str = Field(..., description="The specific, self-contained fact to be written to memory.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence score from 0.0 to 1.0.")
class UpdateMemoryArgs(BaseModel):
    fact_id: str = Field(..., description="The ID of the fact to be updated. This MUST be retrieved from memory first.")
    new_content: str = Field(..., description="The new, updated content of the fact.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="The confidence score for the updated fact.")
class ReadMemoryArgs(BaseModel):
    query: str = Field(..., description="The question or topic to search for in memory.")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
EMBEDDING_MODEL = "gemini-embedding"
from src.data_models import Memory
from src.memory_store import MemoryStore


def get_embedding(text: str):
    """Generates an embedding for a given text using the specified model."""
    text = text.replace("\n", " ")
    # *** THIS IS THE FIX for the BadRequestError ***
    # We explicitly specify the encoding_format to match what the proxy expects.
    return client.embeddings.create(
        input=[text], 
        model=EMBEDDING_MODEL,
        encoding_format="float"  # <--- ADD THIS LINE
    ).data[0].embedding

# ... (rest of the tool functions and definitions are correct and do not need to change)
def write_memory(store: MemoryStore, args: WriteMemoryArgs, extracted_from: str):
    print(f"TOOL: Executing write_memory with content: '{args.content}'")
    memory = Memory(content=args.content, confidence=args.confidence, extracted_from=extracted_from)
    store.write(memory)
    return f"Successfully wrote new memory (ID: {memory.fact_id}): {args.content}"

def update_memory(store: MemoryStore, args: UpdateMemoryArgs, extracted_from: str):
    print(f"TOOL: Executing update_memory for fact '{args.fact_id}'")
    old_memory_data = store.retrieve_by_id(args.fact_id)
    previous_value = old_memory_data['documents'][0] if old_memory_data['ids'] else None
    
    new_memory = Memory(
        fact_id=args.fact_id,
        content=args.new_content,
        confidence=args.confidence,
        extracted_from=extracted_from,
        previous_value=previous_value
    )
    store.update(args.fact_id, new_memory)
    return f"Successfully updated fact {args.fact_id} to: {args.new_content}"

def read_memory(store: MemoryStore, args: ReadMemoryArgs):
    print(f"TOOL: Executing read_memory with query: '{args.query}'")
    query_embedding = get_embedding(args.query)
    results = store.search(query_embeddings=[query_embedding], top_k=3)
    
    structured_results = []
    if results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            structured_results.append({
                "fact_id": doc_id,
                "content": results['documents'][0][i],
                "relevance_score": results['distances'][0][i]
            })
    return structured_results

tools_definitions = [
    {"type": "function", "function": {"name": "write_memory", "description": "Records a new, self-contained fact about the user.", "parameters": WriteMemoryArgs.model_json_schema()}},
    {"type": "function", "function": {"name": "update_memory", "description": "Updates an existing fact with new information. Requires a fact_id.", "parameters": UpdateMemoryArgs.model_json_schema()}},
    {"type": "function", "function": {"name": "read_memory", "description": "Searches memory for facts relevant to a query to retrieve their content and IDs.", "parameters": ReadMemoryArgs.model_json_schema()}},
]

available_tools = {
    "write_memory": write_memory,
    "update_memory": update_memory,
    "read_memory": read_memory,
}