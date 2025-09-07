# src/agent.py

import os
import json
from openai import OpenAI
from src.memory_store import MemoryStore
from src.tools import tools_definitions, available_tools, WriteMemoryArgs, UpdateMemoryArgs, ReadMemoryArgs
from typing import List, Dict

class Agent:
    def __init__(self, store: MemoryStore):
        self.store = store
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        self.model = "gemini-2.5-pro"
        print(f"INFO: Agent initialized with model '{self.model}' and {len(available_tools)} tools.")

    def _get_system_prompt(self):
        # The prompt is good, but we will rely on a guardrail for the most critical rule.
        return """
        You are a highly intelligent and precise memory assistant. Your role is to meticulously manage a memory store based on a conversation.

        **Your Core Directives:**
        1.  **Analyze ONLY the User's Last Turn:** Base your decision *only* on the most recent message from the "user".
        2.  **Extract Atomic Facts:** Identify specific, self-contained facts. A sentence like "I'm Alice, a Google engineer" contains TWO facts: "User's name is Alice" and "User works at Google". Extract them separately.
        3.  **Prevent Duplicates:** Before writing a new fact, consider if it's truly new. If a similar fact already exists, do nothing.
        4.  **Follow the Update Protocol:** If the user corrects or updates information, you MUST follow this two-step process:
            a. **Step 1:** Use the `read_memory` tool to search for the old fact to retrieve its `fact_id`.
            b. **Step 2:** Use the `update_memory` tool with the retrieved `fact_id`.
        5.  **No Action is Default:** If the user's turn contains no new facts, no updates, or is just a simple greeting/question, you MUST NOT call any tools.
        """

    def process_turn(self, conversation_history: List[Dict], current_turn_index: int):
        print(f"\n--- Processing Turn {current_turn_index} ---")
        
        # *** THIS IS THE PROGRAMMATIC GUARDRAIL FIX ***
        # If the last message in the history is not from the user, do nothing.
        if not conversation_history or conversation_history[-1]['role'] != 'user':
            print("AGENT: Skipping turn because it is not a user message.")
            return
        # ----------------------------------------------

        messages = [{"role": "system", "content": self._get_system_prompt()}] + conversation_history

        # The rest of the agent loop logic is now correct
        for i in range(3): 
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools_definitions,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if not tool_calls:
                print("AGENT: Decided no further tools are necessary for this turn.")
                break 

            messages.append(response_message)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)
                
                if function_to_call:
                    function_args_dict = json.loads(tool_call.function.arguments)
                    extracted_from = f"conversation:{current_turn_index}"
                    
                    if function_name == 'write_memory':
                        args = WriteMemoryArgs(**function_args_dict)
                        result = function_to_call(store=self.store, args=args, extracted_from=extracted_from)
                    elif function_name == 'update_memory':
                        args = UpdateMemoryArgs(**function_args_dict)
                        result = function_to_call(store=self.store, args=args, extracted_from=extracted_from)
                    elif function_name == 'read_memory':
                        args = ReadMemoryArgs(**function_args_dict)
                        result = function_to_call(store=self.store, args=args)
                    else:
                        result = f"Error: Tool '{function_name}' not found."

                    print(f"AGENT: Executed '{function_name}'.")
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result),
                    })
                else:
                    print(f"AGENT: Error - Wanted to call unknown tool '{function_name}'.")