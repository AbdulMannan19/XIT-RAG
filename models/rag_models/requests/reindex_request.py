from pydantic import BaseModel, Field


class ReindexRequest(BaseModel):

    force: bool = Field(False, description="Force reindex even if collection exists")
