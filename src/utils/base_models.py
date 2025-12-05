from pydantic import BaseModel
from typing import Optional


class RetrieverArgs(BaseModel):
    embedding_model: str
    rerank_model: Optional[str]
    normalize_embeddings: bool
    ensemble: bool = False
