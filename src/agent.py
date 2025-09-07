# src/agent.py

from src.memory_store import MemoryStore
from typing import List, Callable, Dict

class Agent:
    """
    An agent that orchestrates tools to manage and retrieve memories.
    """
    def __init__(self, tools: List[Callable], store: MemoryStore):
        """
        Initializes the Agent.

        Args:
            tools: A list of tool functions the agent can call.
            store: An initialized MemoryStore instance.
        """
        self.store = store
        self.tools: Dict[str, Callable] = {tool.__name__: tool for tool in tools}
        print("INFO: Agent initialized with tools:", list(self.tools.keys()))

    def process_turn(self, conversation_turn: str):
        """
        Processes a single turn of a conversation.

        This is a placeholder for the real LLM-based tool-use logic.
        It uses simple keyword matching to simulate the LLM's decision.
        """
        print(f"\n--- Processing Turn --- \nInput: '{conversation_turn}'")
        
        # This block simulates the LLM deciding which tool to call
        if "my name is alice" in conversation_turn.lower():
            print("AGENT: Decided to call 'write_memory'")
            # The agent passes the store instance it holds to the tool
            self.tools['write_memory'](
                store=self.store,
                content="User's name is Alice",
                confidence=0.95,
                extracted_from="turn_1"
            )
        elif "switched jobs to anthropic" in conversation_turn.lower():
             print("AGENT: Decided to call 'update_memory'")
             # Here we would need a fact_id. For this demo, we'll hardcode it.
             # A real agent would first need to retrieve the relevant fact to update.
             fact_to_update_id = "fact_001_job" 
             self.tools['update_memory'](
                store=self.store,
                fact_id=fact_to_update_id,
                new_content="User works at Anthropic",
                confidence=0.98,
                extracted_from="turn_2"
             )
        else:
            print("AGENT: Decided no tool was necessary for this turn.")
            