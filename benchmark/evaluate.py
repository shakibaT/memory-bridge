import json
import time
import chromadb
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional, List, Dict

# --- Import Your System Components ---
from src.agent import Agent
from src.memory_store import MemoryStore
from src.tools import get_embedding
from benchmark.dataset_schema import Dataset, Conversation, GroundTruthEvent
from benchmark.baseline import BaselineRetriever

class Evaluator:
    def __init__(self):
        load_dotenv()
        # --- OPTIMIZATION 1: Add an in-memory cache for embeddings ---
        self.embedding_cache: Dict[str, List[float]] = {}
        # -------------------------------------------------------------
        
        with open("benchmark/dataset.json", "r") as f:
            data = json.load(f)
            self.dataset = Dataset(conversations=data)
            
        print(f"Loaded {len(self.dataset.conversations)} conversations. Ready for evaluation.")
        self.results = []

    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Retrieves an embedding from the cache or computes and caches it if not present.
        This is the core of the speed optimization.
        """
        if not text:
            return None
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        try:
            embedding = get_embedding(text)
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            # Silently fail on embedding error to not crash the whole benchmark
            # print(f"Warning: Could not get embedding for text: '{text}'. Error: {e}")
            return None

    def _are_embeddings_similar(self, emb1: List[float], emb2: List[float], threshold=0.85) -> bool:
        """Helper to compare two pre-computed embeddings."""
        if emb1 is None or emb2 is None:
            return False
        score = cosine_similarity([emb1], [emb2])[0][0]
        return score > threshold

    def run_all_evaluations(self, max_conversations: Optional[int] = None):
        """
        Orchestrates the evaluation.
        Args:
            max_conversations: If set, evaluates only the first N conversations for a quick demo.
        """
        conversations_to_run = self.dataset.conversations
        if max_conversations is not None:
            print(f"\n!!! RUNNING IN DEMO MODE: Evaluating only the first {max_conversations} conversations. !!!")
            conversations_to_run = self.dataset.conversations[:max_conversations]

        print("\n--- Evaluating Advanced Agent System ---")
        self._evaluate_system(system_name="Agent", conversations=conversations_to_run)
        
        print("\n--- Evaluating Baseline System ---")
        self._evaluate_system(system_name="Baseline", conversations=conversations_to_run)

    def _evaluate_system(self, system_name: str, conversations: List[Conversation]):
        """Generic evaluation function for a given system."""
        for convo in tqdm(conversations, desc=f"{system_name} System"):
            client = chromadb.Client()
            store = MemoryStore(client=client)
            system = Agent(store=store) if system_name == "Agent" else BaselineRetriever(store=store)

            total_processing_time = 0.0
            history = []
            for i, turn in enumerate(convo.turns):
                history.append(turn.model_dump())
                start_time = time.time()
                if system_name == "Agent":
                    system.process_turn(conversation_history=history, current_turn_index=i)
                else:
                    system.process_turn(turn.model_dump(), i)
                total_processing_time += time.time() - start_time
            
            self._calculate_all_metrics_for_convo(system_name, convo, store, total_processing_time)

    def _calculate_all_metrics_for_convo(self, system_name: str, convo: Conversation, store: MemoryStore, total_processing_time: float):
        """Calculates all metrics for a single conversation using cached embeddings."""
        
        # --- OPTIMIZATION 2: Pre-compute embeddings before loops ---
        gt_extraction_facts = {evt.expected_fact for evt in convo.ground_truth_events if evt.event_type == "EXTRACTION"}
        gt_embeddings = {fact: self._get_cached_embedding(fact) for fact in gt_extraction_facts}

        all_extracted_memories = set(store.collection.get()['documents'])
        extracted_embeddings = {mem: self._get_cached_embedding(mem) for mem in all_extracted_memories}
        # -------------------------------------------------------------

        # Metric 1: Extraction Quality (Precision, Recall, F1)
        true_positives = sum(1 for gt_emb in gt_embeddings.values() if any(self._are_embeddings_similar(gt_emb, mem_emb) for mem_emb in extracted_embeddings.values()))
        
        precision = true_positives / len(extracted_embeddings) if extracted_embeddings else 0
        recall = true_positives / len(gt_embeddings) if gt_embeddings else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        self.results.append({"system": system_name, "metric": "Extraction Precision", "score": precision})
        self.results.append({"system": system_name, "metric": "Extraction Recall", "score": recall})
        self.results.append({"system": system_name, "metric": "Extraction F1-Score", "score": f1_score})

        # Metrics 2 & 3: Update Accuracy and Retrieval Precision
        update_events = [evt for evt in convo.ground_truth_events if evt.event_type == "UPDATE"]
        retrieval_events = [evt for evt in convo.ground_truth_events if evt.event_type == "RETRIEVAL"]

        if update_events:
            correct_updates = 0
            for event in update_events:
                doc = store.retrieve_by_id(event.fact_id_to_update)
                if doc and doc['ids']:
                    updated_emb = self._get_cached_embedding(doc['documents'][0])
                    expected_emb = self._get_cached_embedding(event.expected_fact)
                    if self._are_embeddings_similar(updated_emb, expected_emb):
                        correct_updates += 1
            update_accuracy = correct_updates / len(update_events)
            self.results.append({"system": system_name, "metric": "Update Accuracy", "score": update_accuracy})

        if retrieval_events:
            total_precision = 0
            for event in retrieval_events:
                retrieved_docs = []
                if system_name == "Agent":
                    retrieved_obj = store.search(query_embeddings=[self._get_cached_embedding(event.retrieval_query)], top_k=3)
                    if retrieved_obj and retrieved_obj['ids'][0]: retrieved_docs = retrieved_obj['documents'][0]
                else:
                    baseline_retriever = BaselineRetriever(store=store)
                    retrieved_docs = baseline_retriever.retrieve(event.retrieval_query)
                
                if retrieved_docs:
                    expected_retrieval_emb = self._get_cached_embedding(event.expected_fact)
                    relevant_found = sum(1 for doc in retrieved_docs if self._are_embeddings_similar(self._get_cached_embedding(doc), expected_retrieval_emb))
                    total_precision += relevant_found / len(retrieved_docs)
            
            avg_precision = total_precision / len(retrieval_events) if retrieval_events else 0
            self.results.append({"system": system_name, "metric": "Retrieval Precision@3", "score": avg_precision})

        # Metric 4 (Your Choice): Processing Latency
        avg_latency_per_turn = (total_processing_time / len(convo.turns)) * 1000 # in ms
        self.results.append({"system": system_name, "metric": "Avg Latency per Turn (ms)", "score": avg_latency_per_turn})

    def report_results(self):
        """Generates and prints a summary report of all benchmark results."""
        if not self.results:
            print("\nNo evaluation results were recorded.")
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
    # Run in DEMO mode on just 2 conversations for speed
    evaluator.run_all_evaluations(max_conversations=2)
    evaluator.report_results()

if __name__ == "__main__":
    main()