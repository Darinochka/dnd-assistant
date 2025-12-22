from pydantic import BaseModel
from typing import Optional


class RetrieverArgs(BaseModel):
    embedding_model: str
    rerank_model: Optional[str]
    normalize_embeddings: bool
    ensemble: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "dnd_documents"
    qdrant_api_key: Optional[str] = None
