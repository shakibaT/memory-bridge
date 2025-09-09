# benchmark/generate_data.py

import json
import os
import uuid
from openai import OpenAI # Using OpenAI's library as per the assignment's proxy compatibility
from dotenv import load_dotenv

# --- Configuration ---
# As per the assignment, connect to the provided LiteLLM proxy
# Ensure you have OPENAI_API_KEY and OPENAI_API_BASE set in your environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
MODEL_NAME = "gemini-2.5-pro"
NUM_CONVERSATIONS = 50 # Increased dataset size to meet requirements

# --- Prompts for Data Generation ---
# This prompt asks the LLM to generate not just the conversation,
# but also the ground truth data we need for evaluation.
GENERATION_PROMPT_TEMPLATE = """
You are a data generator for testing an AI memory system. Your task is to create a realistic, multi-turn conversation and the corresponding ground truth data for evaluation.

The conversation should include:
1.  **Simple Fact Storage**: At least 3-4 basic facts about the user (e.g., name, city, hobbies, job).
2.  **Fact Update/Correction**: The user must correct a piece of information they gave earlier.
3.  **Multi-hop Reasoning**: Include facts that are related, requiring the system to connect them. For example, "I live in Brooklyn" and later "The summers are great here for biking." A query like "Does the user like biking in Brooklyn?" would require multi-hop reasoning.

Generate a JSON object with the following structure:
- `conversation_id`: A unique identifier.
- `turns`: A list of conversation turns, with `role` ('user' or 'assistant') and `content`.
- `ground_truth`:
  - `facts`: A list of all facts that should be extracted. Each fact should have:
    - `content`: The structured fact (e.g., "User's name is Maya").
    - `turn_ids`: A list of turn indices (starting from 0) where this fact is mentioned or updated.
    - `is_update`: A boolean indicating if this is an update to a previous fact.
    - `previous_value` (optional): The old value if `is_update` is true.
  - `queries`: A list of questions to test the memory system. Each query should have:
    - `query`: The question.
    - `expected_facts`: A list of the exact `content` of the facts needed to answer the query.

Ensure the final output is only the JSON object, without any other text or markdown.

Example Fact Update:
{"role": "user", "content": "I work at a startup called 'InnovateAI'."}
...
{"role": "user", "content": "Actually, I switched jobs. I now work at 'Global Tech'."}

Ground truth for this update:
{
  "content": "User works at Global Tech",
  "turn_ids": [X, Y],
  "is_update": true,
  "previous_value": "User works at InnovateAI"
}
"""

def generate_synthetic_dataset():
    """Generates a dataset of conversations with ground truth for evaluation."""
    print(f"Generating {NUM_CONVERSATIONS} conversations...")
    dataset = []
    for i in range(NUM_CONVERSATIONS):
        try:
            print(f"Generating conversation {i+1}/{NUM_CONVERSATIONS}...")
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": GENERATION_PROMPT_TEMPLATE}],
                temperature=0.8,
            )
            
            response_text = completion.choices[0].message.content
            # Clean up potential markdown formatting
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3]
            
            generated_data = json.loads(response_text)
            generated_data['conversation_id'] = f"conv_{uuid.uuid4()}"
            dataset.append(generated_data)

        except Exception as e:
            print(f"Error generating conversation {i+1}: {e}")
            print("Skipping this one.")
    
    return dataset

if __name__ == "__main__":
    # Create benchmark directory if it doesn't exist
    if not os.path.exists("benchmark"):
        os.makedirs("benchmark")
        
    generated_dataset = generate_synthetic_dataset()
    
    if generated_dataset:
        output_path = os.path.join("benchmark", "dataset.json")
        with open(output_path, "w") as f:
            json.dump(generated_dataset, f, indent=2)
        print(f"\nSuccessfully generated {len(generated_dataset)} conversations.")
        print(f"Dataset saved to {output_path}")
    else:
        print("Dataset generation failed.")