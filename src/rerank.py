
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512)


# Re-scores a shortlist from first-stage retrieval. The retrieval scores are not in input.
# The model reads the raw text pairs, and the old score is dropped. 
# The bi-encoder embeds query and chunk separately and compares two vectors (so chunk vectors are precomputed at ingest)
# A cross-encoder, on the other hand, feeds "query [SEP] chunk" through the transformer together, letting every query token
# - attend to every chunk token. More accurate, but nothing can be precomputed: one forward pass per
# - pair, which is why only k=20 candidates are reranked and not the whole collection.
def rerank(query: str, candidates: List[Tuple[Document, float]], top_n: int) -> List[Tuple[Document, float]]:
    """Scores are unbounded logits (from -10 to 10), not similarities"""
    if not candidates:
        return []

    # One score per candidate, in candidate order
    # query, chunk pairs
    scores = get_reranker().predict([(query, doc.page_content) for doc, _score in candidates])
    # Sort by reranked score, descending order, cut to top n
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:top_n]
    return [(doc, float(score)) for (doc, _old), score in ranked]