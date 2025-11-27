"""Application constants."""

# Guardrail messages
NO_KB_MSG = "I don't have verifiable information in the knowledge base for that query."

# Legal disclaimer keywords
ADVICE_KEYWORDS = [
    "should i file",
    "what should i claim",
    "advice",
    "deduct",
    "penalty strategy",
    "how should i",
    "what do you recommend",
    "should i",
    "can i claim",
    "what can i deduct",
]

# Content types
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_FAQ = "faq"
CONTENT_TYPE_FORM = "form"

# Chunking defaults
DEFAULT_CHUNK_MIN = 800
DEFAULT_CHUNK_MAX = 1600
DEFAULT_OVERLAP_RATIO = 0.25  # 25% overlap

# Vector DB defaults
HNSW_M = 64
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 128

# Language
DEFAULT_LANGUAGE = "en"


