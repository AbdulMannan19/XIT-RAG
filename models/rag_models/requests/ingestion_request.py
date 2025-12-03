from typing import Optional

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    seed_url: str = Field(default="https://www.irs.gov")
    max_pages: int = Field(default=100, ge=1, le=10000)
    concurrency: int = Field(default=2, ge=1, le=10)
    allow_prefix: Optional[list[str]] = Field(default=None)
    only_html: bool = Field(default=False)
    only_pdf: bool = Field(default=False)
    forms: Optional[list[str]] = Field(default=None)
    include_seed: bool = Field(default=True)
    follow_links: bool = Field(default=True)
    url_file: Optional[str] = Field(default=None)
