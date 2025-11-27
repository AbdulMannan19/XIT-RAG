"""Security utilities for API authentication."""

from fastapi import Header, HTTPException, status
from typing import Annotated

from services.rag_services.core_services.config import settings


async def verify_api_key(x_api_key: Annotated[str, Header()]) -> str:
    """Verify API key from header."""
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key


def should_add_disclaimer(query: str) -> bool:
    """Check if query requires legal disclaimer."""
    query_lower = query.lower()
    from services.rag_services.core_services.constants import ADVICE_KEYWORDS

    return any(kw in query_lower for kw in ADVICE_KEYWORDS)


