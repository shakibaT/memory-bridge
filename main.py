# main.py

import chromadb
from src.agent import Agent
from src.memory_store import MemoryStore
from src.tools import write_memory, update_memory, read_memory

def main():
    """
    Main function to initialize and run the agent demo.
    It wires together all the components of the system.
    """
    print("--- Initializing Memory Bridge System ---")

    # 1. Initialize the DB Client.
    # We use the standard in-memory client for simple, clean execution.
    # This client does not start a background server, so the script will exit properly.
    client = chromadb.Client()

    # 2. Initialize the MemoryStore with the client.
    # This is the dependency injection that fixes the original TypeError.
    store = MemoryStore(client=client)

    # 3. Define the list of tools the agent can use.
    tool_belt = [write_memory, update_memory, read_memory]

    # 4. Initialize the Agent, providing it with the tools and the memory store.
    memory_agent = Agent(tools=tool_belt, store=store)

    # 5. Run a sample conversation through the agent.
    print("\n--- Starting Agent Demo Conversation ---")
    memory_agent.process_turn("User: Hi, my name is Alice.") # This won't trigger the keyword
    memory_agent.process_turn("User: My name is Alice, and I work at Google.") # This will
    memory_agent.process_turn("User: Actually, I switched jobs to Anthropic.") # This will trigger update

    print("\n--- System Shutdown ---")


if __name__ == "__main__":
    main()