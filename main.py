# main.py (Rewritten for the new Agentic System)

import chromadb
from dotenv import load_dotenv
from src.agent import Agent
from src.memory_store import MemoryStore

def main():
    """
    A simple demo script to initialize and run the intelligent agent.
    This script simulates a short conversation to demonstrate the agent's
    ability to extract and update memories using an LLM.
    """
    load_dotenv()
    print("--- Initializing Memory Bridge Demo ---")

    client = chromadb.Client()
    store = MemoryStore(client=client)
    memory_agent = Agent(store=store)
    
    conversation_history = []
    
    print("\n--- Starting Demo Conversation ---")

    # --- Turn 1: User introduces two facts ---
    turn_1_content = "Hi, my name is Alice and I am a project manager at Innovate Inc."
    conversation_history.append({"role": "user", "content": turn_1_content})
    memory_agent.process_turn(conversation_history=conversation_history, current_turn_index=0)

    # --- Turn 2: Assistant's turn (should be ignored by agent) ---
    turn_2_content = "It's great to meet you, Alice! How can I help you today?"
    conversation_history.append({"role": "assistant", "content": turn_2_content})
    memory_agent.process_turn(conversation_history=conversation_history, current_turn_index=1)

    # --- Turn 3: User updates one of the facts ---
    turn_3_content = "Actually, I was just promoted to Director. My company is still Innovate Inc though."
    conversation_history.append({"role": "user", "content": turn_3_content})
    memory_agent.process_turn(conversation_history=conversation_history, current_turn_index=2)

    print("\n--- Demo Conversation Finished ---")
    
    print("\n--- Final State of Memory Store ---")
    final_memories = store.collection.get(include=["metadatas", "documents"])
    if final_memories['ids']:
        for i, doc_id in enumerate(final_memories['ids']):
            print(f"ID: {doc_id}")
            print(f"  Content: '{final_memories['documents'][i]}'")
            print(f"  Metadata: {final_memories['metadatas'][i]}")
    else:
        print("No memories were stored.")

    print("\n--- System Shutdown ---")

if __name__ == "__main__":
    main()
    