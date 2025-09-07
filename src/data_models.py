# src/data_models.py

from pydantic import BaseModel, Field
from typing import Optional
import datetime
import uuid

class Memory(BaseModel):
    fact_id: str = Field(default_factory=lambda: f"fact_{uuid.uuid4()}")
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    extracted_from: str
    previous_value: Optional[str] = None