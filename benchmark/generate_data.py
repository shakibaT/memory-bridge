# benchmark/generate_data.py

import os
import json
from openai import OpenAI
from benchmark.dataset_schema import Dataset, Conversation
from dotenv import load_dotenv

def get_generation_prompt(scenario: str) -> str:
    # Adding RETRIEVAL to the instructions for the LLM
    return f"""
    You are a synthetic conversation data generator. Your task is to create a single, high-quality conversation in JSON format that fits the following scenario: '{scenario}'.

    The JSON output MUST conform to this exact schema:
    {json.dumps(Conversation.model_json_schema(), indent=2)}

    RULES:
    1.  Create a realistic conversation between a "user" and an "assistant".
    2.  For `EXTRACTION`, `UPDATE`, and `RETRIEVAL` events, the `expected_fact` should be a concise statement.
    3.  For `UPDATE` events, provide a plausible `fact_id_to_update`. The conversation must show the user correcting previous information.
    4.  For `RETRIEVAL` events, provide a `retrieval_query` that the assistant would need to answer. The `expected_fact` is the key piece of information that must be retrieved to answer the query correctly.
    5.  The `turn_index` is zero-based. The JSON output must be a single, complete JSON object.
    """

def generate_dataset(num_conversations: int = 5):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
    model = "gemini-2.5-pro"
    
    scenarios = [
        "A user provides their name and email, then later corrects the email.",
        "A user states their budget is $1500 for a new phone. A few turns later, the assistant needs to recall this budget to make a recommendation.",
        "A user mentions their project's codename is 'Bluebird' and that the deadline is in two weeks. Later, the user asks 'What is the timeline for my project?'",
        "A user says they live in London. They later ask for recommendations for local parks.",
        "A user's favorite programming language is Python. They later change their mind to Rust."
    ]
    
    all_conversations = []
    print(f"Generating {num_conversations} conversations...")
    for i in range(num_conversations):
        scenario = scenarios[i % len(scenarios)] # Cycle through scenarios
        print(f"Generating conversation {i+1}/{num_conversations} for scenario: '{scenario}'")
        prompt = get_generation_prompt(scenario)
        
        try:
            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.8)
            json_response = response.choices[0].message.content.strip().replace('```json', '').replace('```', '')
            conversation_data = json.loads(json_response)
            conversation = Conversation(**conversation_data)
            all_conversations.append(conversation.model_dump())
            print(f"-> Successfully generated and validated conversation '{conversation.conversation_id}'.")
        except Exception as e:
            print(f"-> ERROR: Failed to generate or parse data for scenario. Error: {e}")

    dataset = Dataset(conversations=all_conversations)
    output_path = "benchmark/dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2)
        
    print(f"\nDataset generation complete. Saved {len(all_conversations)} conversations to {output_path}")

if __name__ == "__main__":
    generate_dataset()