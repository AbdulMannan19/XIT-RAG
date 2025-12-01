"""Chat request model."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request schema."""

    query: str = Field(..., description="User query", min_length=1, max_length=2000)
    filters: Optional[dict[str, Any]] = Field(
        None, description="Optional filters for retrieval"
    )
    json: bool = Field(False, description="Return JSON response format")
