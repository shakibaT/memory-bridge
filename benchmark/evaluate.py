import json
import chromadb
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm

from src.agent import Agent
from src.memory_store import MemoryStore
from src.tools import get_embedding
from benchmark.dataset_schema import Dataset
from benchmark.baseline import BaselineRetriever

class Evaluator:
    def __init__(self):
        load_dotenv()
        with open("benchmark/dataset.json", "r") as f:
            self.dataset = Dataset(**json.load(f))
        print(f"Loaded {len(self.dataset.conversations)} conversations for evaluation.")
        self.results = []

    def _is_match(self, text1: str, text2: str, threshold=0.85) -> bool:
        if not text1 or not text2: return False
        try:
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            score = cosine_similarity([emb1], [emb2])[0][0]
            return score > threshold
        except Exception:
            return text1.strip().lower() == text2.strip().lower()

    def evaluate_agent_system(self):
        print("\n--- Evaluating Advanced Agent System ---")
        for convo in tqdm(self.dataset.conversations, desc="Agent System"):
            client = chromadb.Client()
            store = MemoryStore(client=client)
            agent = Agent(store=store)
            history = []
            
            for i, turn in enumerate(convo.turns):
                # The history is built up turn by turn
                history.append(turn.model_dump()) # Use .model_dump() to convert Pydantic model to dict
                
                # --- THIS IS THE FIX ---
                # We simply call the agent on every turn. The agent's internal
                # guardrail is responsible for ignoring non-user turns.
                agent.process_turn(conversation_history=history, current_turn_index=i)
                # -----------------------

                for event in convo.ground_truth_events:
                    if event.turn_index == i:
                        self.calculate_metrics("Agent", convo.conversation_id, event, store)

    def evaluate_baseline_system(self):
        print("\n--- Evaluating Baseline System ---")
        for convo in tqdm(self.dataset.conversations, desc="Baseline System"):
            client = chromadb.Client()
            store = MemoryStore(client=client)
            baseline = BaselineRetriever(store=store)
            
            for i, turn in enumerate(convo.turns):
                baseline.process_turn(turn.model_dump(), i) # Use .model_dump() here as well
                for event in convo.ground_truth_events:
                    if event.turn_index == i:
                        self.calculate_metrics("Baseline", convo.conversation_id, event, store, baseline_retriever=baseline)

    def calculate_metrics(self, system_name, convo_id, event, store, baseline_retriever=None):
        # ... (rest of the function is correct)
        if event.event_type == "EXTRACTION":
            all_memories = store.collection.get()['documents']
            success = any(self._is_match(mem, event.expected_fact) for mem in all_memories)
            self.results.append({"system": system_name, "conversation_id": convo_id, "metric": "Extraction Quality", "success": success})
        elif event.event_type == "UPDATE":
            doc = store.retrieve_by_id(event.fact_id_to_update)
            updated_content = doc['documents'][0] if doc['ids'] else ""
            success = self._is_match(updated_content, event.expected_fact)
            self.results.append({"system": system_name, "conversation_id": convo_id, "metric": "Update Accuracy", "success": success})
        elif event.event_type == "RETRIEVAL":
            if system_name == "Agent":
                retrieved_docs_obj = store.search(query_embeddings=[get_embedding(event.retrieval_query)], top_k=3)
                retrieved_docs = retrieved_docs_obj['documents'][0] if retrieved_docs_obj['ids'][0] else []
            else:
                retrieved_docs = baseline_retriever.retrieve(event.retrieval_query)
            success = any(self._is_match(doc, event.expected_fact) for doc in retrieved_docs)
            self.results.append({"system": system_name, "conversation_id": convo_id, "metric": "Retrieval Precision@3", "success": success})


    def report_results(self):
        # ... (this function is correct)
        if not self.results:
            print("\nNo evaluation events were triggered. Check dataset and evaluation logic.")
            return
        df = pd.DataFrame(self.results)
        df['metric'] = "Overall Success Rate" # Simplified metric for the report
        summary = df.groupby(['system', 'metric'])['success'].value_counts(normalize=True).unstack(fill_value=0)
        summary['success_rate_%'] = summary.get(True, 0) * 100
        print("\n\n--- Benchmark Results Summary ---")
        print(summary[['success_rate_%']].round(2))
        print("---------------------------------")

def main():
    evaluator = Evaluator()
    evaluator.evaluate_agent_system()
    evaluator.evaluate_baseline_system()
    evaluator.report_results()

if __name__ == "__main__":
    main()
