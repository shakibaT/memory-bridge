import json
import time
import chromadb
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm

# --- Import Your System Components ---
from src.agent import Agent
from src.memory_store import MemoryStore
from src.tools import get_embedding
from benchmark.dataset_schema import Dataset, Conversation, GroundTruthEvent
from benchmark.baseline import BaselineRetriever

class Evaluator:
    def __init__(self):
        load_dotenv()
        with open("benchmark/dataset.json", "r") as f:
            data = json.load(f)
            # --- THIS IS THE FIX ---
            # The JSON file is a list of conversations. We need to pass this list
            # to the 'conversations' field of the Dataset Pydantic model.
            self.dataset = Dataset(conversations=data)
            # -----------------------
            
        print(f"Loaded {len(self.dataset.conversations)} conversations for evaluation.")
        self.results = []

    def _is_match(self, text1: str, text2: str, threshold=0.85) -> bool:
        """Checks if two strings are semantically similar using embeddings."""
        if not text1 or not text2: return False
        try:
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            score = cosine_similarity([emb1], [emb2])[0][0]
            return score > threshold
        except Exception:
            return text1.strip().lower() == text2.strip().lower()

    def run_all_evaluations(self):
        """Orchestrates the evaluation for both systems."""
        print("\n--- Evaluating Advanced Agent System ---")
        self._evaluate_system(system_name="Agent")
        
        print("\n--- Evaluating Baseline System ---")
        self._evaluate_system(system_name="Baseline")

    def _evaluate_system(self, system_name: str):
        """
        Generic evaluation function that runs the benchmark for a given system.
        """
        for convo in tqdm(self.dataset.conversations, desc=f"{system_name} System"):
            # --- 1. Simulation Phase ---
            # Run the entire conversation to build the memory store
            client = chromadb.Client()
            store = MemoryStore(client=client)
            system = None
            if system_name == "Agent":
                system = Agent(store=store)
            else:
                system = BaselineRetriever(store=store)

            total_processing_time = 0.0
            history = []
            for i, turn in enumerate(convo.turns):
                history.append(turn.model_dump())
                start_time = time.time()
                # Use a unified interface `process_turn` for both systems
                if system_name == "Agent":
                    system.process_turn(conversation_history=history, current_turn_index=i)
                else:
                    system.process_turn(turn.model_dump(), i)
                total_processing_time += time.time() - start_time
            
            # --- 2. Measurement Phase ---
            # Now, evaluate the final state of the memory against all ground truth events
            self._calculate_all_metrics_for_convo(system_name, convo, store, total_processing_time)

    def _calculate_all_metrics_for_convo(self, system_name: str, convo: Conversation, store: MemoryStore, total_processing_time: float):
        """Calculates all metrics for a single, completed conversation."""
        
        # --- Metric 1: Extraction Quality (Precision, Recall, F1) ---
        gt_extraction_facts = {evt.expected_fact for evt in convo.ground_truth_events if evt.event_type == "EXTRACTION"}
        all_extracted_memories = set(store.collection.get()['documents'])
        
        true_positives = len([gt for gt in gt_extraction_facts if any(self._is_match(mem, gt) for mem in all_extracted_memories)])
        
        precision = true_positives / len(all_extracted_memories) if all_extracted_memories else 0
        recall = true_positives / len(gt_extraction_facts) if gt_extraction_facts else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        self.results.append({"system": system_name, "metric": "Extraction Precision", "score": precision})
        self.results.append({"system": system_name, "metric": "Extraction Recall", "score": recall})
        self.results.append({"system": system_name, "metric": "Extraction F1-Score", "score": f1_score})

        # --- Metrics 2 & 3: Update Accuracy and Retrieval Precision ---
        update_events = [evt for evt in convo.ground_truth_events if evt.event_type == "UPDATE"]
        retrieval_events = [evt for evt in convo.ground_truth_events if evt.event_type == "RETRIEVAL"]

        # Update Accuracy
        if update_events:
            correct_updates = 0
            for event in update_events:
                doc = store.retrieve_by_id(event.fact_id_to_update)
                if doc and doc['ids'] and self._is_match(doc['documents'][0], event.expected_fact):
                    correct_updates += 1
            update_accuracy = correct_updates / len(update_events)
            self.results.append({"system": system_name, "metric": "Update Accuracy", "score": update_accuracy})

        # Retrieval Precision
        if retrieval_events:
            avg_precision = 0
            for event in retrieval_events:
                retrieved_docs = []
                if system_name == "Agent":
                    retrieved_obj = store.search(query_embeddings=[get_embedding(event.retrieval_query)], top_k=3)
                    if retrieved_obj and retrieved_obj['ids'][0]:
                        retrieved_docs = retrieved_obj['documents'][0]
                else: # Baseline
                    baseline_retriever = BaselineRetriever(store=store)
                    retrieved_docs = baseline_retriever.retrieve(event.retrieval_query)
                
                if retrieved_docs:
                    relevant_found = sum(1 for doc in retrieved_docs if self._is_match(doc, event.expected_fact))
                    avg_precision += relevant_found / len(retrieved_docs)
            
            final_avg_precision = avg_precision / len(retrieval_events) if retrieval_events else 0
            self.results.append({"system": system_name, "metric": "Retrieval Precision@3", "score": final_avg_precision})

        # --- Metric 4 (Your Choice): Processing Latency ---
        avg_latency_per_turn = (total_processing_time / len(convo.turns)) * 1000 # in milliseconds
        self.results.append({"system": system_name, "metric": "Avg Latency per Turn (ms)", "score": avg_latency_per_turn})

    def report_results(self):
        """Generates and prints a summary report of all benchmark results."""
        if not self.results:
            print("\nNo evaluation results were recorded. Please check your dataset and evaluation logic.")
            return

        df = pd.DataFrame(self.results)
        
        summary = df.groupby(['system', 'metric'])['score'].mean().reset_index()
        
        summary_pivot = summary.pivot(index='metric', columns='system', values='score')
        
        metric_order = [
            "Extraction Precision", "Extraction Recall", "Extraction F1-Score",
            "Update Accuracy", "Retrieval Precision@3", "Avg Latency per Turn (ms)"
        ]
        summary_pivot = summary_pivot.reindex(metric_order).dropna(how='all')

        print("\n\n--- Benchmark Results Summary ---")
        print("Scores are averaged across all conversations.")
        print(summary_pivot.round(3).to_markdown())
        print("---------------------------------")

def main():
    evaluator = Evaluator()
    evaluator.run_all_evaluations()
    evaluator.report_results()

if __name__ == "__main__":
    main()
    