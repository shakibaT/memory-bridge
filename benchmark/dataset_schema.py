# benchmark/dataset_schema.py

from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional

class GroundTruthEvent(BaseModel):
    """
    Defines a single evaluation event within a conversation.
    This tells the evaluation script what to check for and when.
    """
    turn_index: int = Field(..., description="The zero-based index of the turn where the event should be evaluated.")
    
    event_type: Literal["EXTRACTION", "UPDATE", "RETRIEVAL"] = Field(..., description="The type of event to evaluate.")
    
    expected_fact: str = Field(..., description="The semantic content of the fact that should exist in memory or be retrieved.")
    
    # This field is specifically for "UPDATE" events
    fact_id_to_update: Optional[str] = Field(default=None, description="The ID of the fact that should have been updated.")
    
    # This field is specifically for "RETRIEVAL" events
    retrieval_query: Optional[str] = Field(default=None, description="The query to send to the retriever to test its performance.")


class Conversation(BaseModel):
    """

    Represents a single conversation test case, including the dialogue
    and the ground truth events for evaluation.
    """
    conversation_id: str = Field(..., description="A unique identifier for the conversation.")
    
    scenario: str = Field(..., description="A human-readable description of what this conversation is testing.")
    
    turns: List[Dict[str, str]] = Field(..., description="The sequence of turns between the 'user' and 'assistant'.")
    
    ground_truth_events: List[GroundTruthEvent] = Field(..., description="A list of events to be evaluated during the benchmark.")


class Dataset(BaseModel):
    """

    The top-level model for the entire benchmark dataset file.
    """
    conversations: List[Conversation]