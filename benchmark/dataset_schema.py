from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional

class Turn(BaseModel):
    """
    A new, stricter model for a single conversation turn.
    This ensures every turn has both a 'role' and 'content'.
    """
    role: Literal["user", "assistant"]
    content: str

class GroundTruthEvent(BaseModel):
    turn_index: int = Field(..., description="The zero-based index of the turn where the event should be evaluated.")
    event_type: Literal["EXTRACTION", "UPDATE", "RETRIEVAL"]
    expected_fact: str = Field(..., description="The semantic content of the fact that should exist in memory or be retrieved.")
    fact_id_to_update: Optional[str] = Field(default=None, description="The ID of the fact that should have been updated.")
    retrieval_query: Optional[str] = Field(default=None, description="The query to send to the retriever to test its performance.")

class Conversation(BaseModel):
    conversation_id: str
    scenario: str
    # This now uses our new, stricter Turn model.
    turns: List[Turn] = Field(..., description="The sequence of turns, each with a role and content.")
    ground_truth_events: List[GroundTruthEvent]

class Dataset(BaseModel):
    conversations: List[Conversation]
    