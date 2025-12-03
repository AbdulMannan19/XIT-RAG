from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AdminStats(BaseModel):

    collection_name: str
    total_chunks: int
    last_updated: Optional[datetime] = None
    embedding_model: str
    vector_size: int
