# A very basic agent to start. It doesn't use an LLM yet.
# We are just setting up the structure.

class Agent:
    def __init__(self, tools: list):
        self.tools = {tool.__name__: tool for tool in tools}
        print("Agent initialized with tools:", list(self.tools.keys()))

    def process_turn(self, conversation_turn: str):
        """
        For now, this is a placeholder. In the next phase, this method
        will make an LLM call to decide which tool to use.
        """
        print(f"\n--- Processing Turn --- \n{conversation_turn}")
        # Placeholder logic:
        if "my name is" in conversation_turn.lower():
            # Pretend LLM decided to call this tool
            print("Agent decided to call 'write_memory'")
            # self.tools['write_memory'](...)
        elif "actually" in conversation_turn.lower():
            print("Agent decided to call 'update_memory'")
            # self.tools['update_memory'](...)
        else:
            print("Agent decided to do nothing.")

# In a main.py or demo script:
# from src.tools import write_memory, update_memory, read_memory
# from src.agent import Agent
#
# tool_belt = [write_memory, update_memory, read_memory]
# memory_agent = Agent(tools=tool_belt)
# memory_agent.process_turn("User: Hi, my name is Alex.")