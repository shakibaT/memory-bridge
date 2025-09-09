# benchmark/generate_data.py

import json
import os
import uuid
from openai import OpenAI # Using OpenAI's library as per the assignment's proxy compatibility
from tqdm import tqdm
from dotenv import load_dotenv

# --- Configuration ---
# As per the assignment, connect to the provided LiteLLM proxy
# Ensure you have OPENAI_API_KEY and OPENAI_API_BASE set in your environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
MODEL_NAME = "gemini-2.5-pro"
NUM_CONVERSATIONS = 20 # Increased dataset size to meet requirements


# --- New, Schema-Aware Prompt ---
GENERATION_PROMPT_TEMPLATE = """
You are a data generator for testing an AI memory system. Your task is to create a single, realistic, multi-turn conversation and its corresponding ground truth evaluation data in a specific JSON format.

The conversation MUST include:
1.  **Simple Fact Storage**: At least 3-4 basic facts about the user.
2.  **Fact Update/Correction**: The user must correct a piece of information they gave earlier.
3.  **Multi-hop Reasoning Scenario**: Facts that are related and can be used for a multi-hop reasoning query.

Generate a single JSON object with the following structure:
- `conversation_id`: A unique string.
- `scenario`: A brief, one-sentence description of the conversation's purpose (e.g., "User plans a trip and corrects their destination.").
- `turns`: A list of conversation turns, each with a `role` ('user' or 'assistant') and `content`.
- `ground_truth_events`: A list of evaluation events. Each event is a JSON object with:
  - `turn_index`: The zero-based index of the turn where this event should be evaluated.
  - `event_type`: Can be "EXTRACTION", "UPDATE", or "RETRIEVAL".
  - `expected_fact`: The semantic content of the fact. For EXTRACTION, this is the fact that should be saved. For UPDATE, it's the *new* value of the fact. For RETRIEVAL, it's the fact that should be returned.
  - `fact_id_to_update`: (Only for "UPDATE" events) A unique string ID for the fact being updated. The initial EXTRACTION event for this fact must also use this same ID. This can be null for other event types.
  - `retrieval_query`: (Only for "RETRIEVAL" events) The question to test retrieval. This can be null for other event types.

RULES:
- For every new fact introduced by the user, create an "EXTRACTION" event.
- For a fact correction, create an "UPDATE" event. Assign a consistent `fact_id_to_update` to both the original extraction and the update event.
- Create at least one "RETRIEVAL" event that tests the memory.

The final output must be only the JSON object, without any markdown formatting or other text.
"""

def generate_synthetic_dataset():
    """Generates a dataset of conversations with ground truth for evaluation."""
    print(f"Generating {NUM_CONVERSATIONS} conversations...")
    dataset = []
    for _ in tqdm(range(NUM_CONVERSATIONS), desc="Generating Conversations"):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": GENERATION_PROMPT_TEMPLATE}],
                temperature=0.7,
                response_format={"type": "json_object"} # Use JSON mode for reliability
            )
            
            response_text = completion.choices[0].message.content
            generated_data = json.loads(response_text)
            
            # Ensure a unique ID, just in case the model doesn't
            generated_data['conversation_id'] = f"conv_{uuid.uuid4()}"
            dataset.append(generated_data)

        except Exception as e:
            print(f"Error during generation: {e}. Skipping this conversation.")
    
    return dataset

if __name__ == "__main__":
    if not os.path.exists("benchmark"):
        os.makedirs("benchmark")
        
    generated_dataset = generate_synthetic_dataset()
    
    if generated_dataset:
        output_path = os.path.join("benchmark", "dataset.json")
        with open(output_path, "w") as f:
            json.dump(generated_dataset, f, indent=2)
        print(f"\nSuccessfully generated {len(generated_dataset)} conversations.")
        print(f"New dataset saved to {output_path}")
    else:
        print("\nDataset generation failed. Check API keys and model availability.")