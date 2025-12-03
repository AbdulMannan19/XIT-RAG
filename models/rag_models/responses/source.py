from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class Source(BaseModel):

    url: HttpUrl
    title: str
    section: Optional[str] = None
    snippet: str
    char_start: int
    char_end: int
    score: float = Field(..., ge=0.0, le=1.0)
