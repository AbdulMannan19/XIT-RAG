"""Reindex request model."""

from pydantic import BaseModel, Field


class ReindexRequest(BaseModel):
    """Reindex request schema."""

    force: bool = Field(False, description="Force reindex even if collection exists")
