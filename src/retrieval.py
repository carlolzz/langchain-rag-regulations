
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple, Dict
from langchain_core.documents import Document
import re
from functools import lru_cache
from rank_bm25 import BM25Okapi

from src.config import DEFAULT_THRESHOLDS


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
Retrieved = List[Tuple[Document, float]]
# BM25 tokenizer. Two alternatives, tried left to right on every position:
#   \d+(?:[.,/]\d+)* --> a number, keeping . , / that sit between digits -> "380/2001", "3,00", "2.10"
#   \w+  --------------> otherwise a run of letters/digits/underscore (accents included)
# Punctuation and whitespace match neither, so they act as separators and are dropped.
# The number branch comes first on purpose: with \w+ alone, "380/2001" would split into the common
# tokens "380" and "2001" and BM25 would lose the rare statute reference the hypothesis rests on.
# Known effects: "art." -> "art" (so "Art." and "art" are the same token), "D.Lgs." -> "d", "lgs",
# and "dell'edificio" -> "dell", "edificio".
TOKEN_RE = re.compile(r"\d+(?:[.,/]\d+)*|\w+")
RRF_K: int = 60
HYBRID_CANDIDATES: int = 20
MODES: List[str] = ["dense", "bm25", "hybrid", "rerank"]


@lru_cache(maxsize=None)
def get_db(chroma_path: str, collection_name: str, emb_model_name: str) -> Chroma:

    persist_path = (PROJECT_ROOT / chroma_path).resolve()

    db = Chroma(
        persist_directory=persist_path,
        embedding_function=OpenAIEmbeddings(model=emb_model_name),
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if db._collection.count() == 0:
        raise ValueError(
            f"Collection {collection_name!r} is empty or missing in {persist_path}."
            "Run `uv run python -m src.ingest` first."
        )

    return db


def tokenize(text:str) -> List[str]:
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


@lru_cache(maxsize=None)
def get_bm25(chroma_path:str, collection_name: str, emb_model_name: str) -> Tuple[BM25Okapi, List[Document]]:

    # Get the database, retrieve the documents inside it
    # get_db -> Chroma -> dict
    data: dict = get_db(chroma_path, collection_name, emb_model_name).get(include=["documents", "metadatas"])
    docs = [Document(page_content=text, metadata=meta or {}) for text, meta in zip(data["documents"], data["metadatas"])]

    # BM250kapi creates a BM25 index
    # From a [list[list[str]]] with one token list per chunk, calculates the statistics
    # chunk -> list of tokens
    return BM25Okapi([tokenize(doc.page_content) for doc in docs]), docs


def dense_search(chroma_path:str, collection_name: str, query: str, emb_model_name: str, k:int) -> Retrieved:

    # Standard default search
    # Search the db for similar results. The relevance score is 1 - cosine distance,
    # k = number of top documents to retrieve
    return get_db(chroma_path, collection_name, emb_model_name).similarity_search_with_relevance_scores(query, k=k)


def bm25_search(chroma_path:str, collection_name: str, query: str, emb_model_name: str, k:int) -> Retrieved:

    # Calculate the scores
    # index is a bm25 object that holds word statistics
    # - 1. IDF: rarity weight per token, from how many chunks contain it; token appears less -> higher score
    # - 2. How many times does each token appear in one particular chunk, for each chunk
    index, docs = get_bm25(chroma_path, collection_name, emb_model_name)
    # get_scores returns an array with one BM25 score per chunk
    # Filter out scores less than 0
    scores = index.get_scores(tokenize(query))
    # Returns a sorted list of chunk positions by the bm25 scores for each doc in descending order, take the top k
    top = sorted(range(len(docs)), key=lambda idx: scores[idx], reverse=True)[:k]
    return [(docs[idx], float(scores[idx])) for idx in top if scores[idx] > 0.0]


def rrf_fuse(rankings: List[Retrieved], weights: List[float] | None = None, rff_k: int = RRF_K) -> Retrieved:
    """score(d) = sum over rankings of weight / (rrf_k + rank of d). Scores ~0.03, not cosine."""

    weights = weights or [1.0] * len(rankings)
    fused: Dict[str, float] = {}
    docs_by_key: Dict[str, Document] = {}

    # One weight per ranking
    # A chunk that appears in both result lists (dense, bm25) gets (w/(60+rank_dense) + w/(60+rank_bm25))
    for ranking, weight in zip(rankings, weights):
        for rank, (doc, _score) in enumerate(ranking, start=1):
            # key by chunk_id or content
            key = doc.metadata.get("chunk_id") or doc.page_content
            # score + weight / (60 + rank which is idx)
            fused[key] = fused.get(key, 0.0) + (weight / (rff_k + rank))
            docs_by_key[key] = doc

    order = sorted(fused, key=fused.get, reverse=True)
    # dict: key -> (document, rff_score)
    return [(docs_by_key[key], fused[key]) for key in order]


def retrieve(
        chroma_path: str, 
        query: str, 
        collection_name: str, 
        emb_model_name: str, 
        k: int = 3, 
        mode:str = "dense",
        apply_threshold: bool = True
    ) -> List[Tuple[Document, float]]:
    """
    The first part of RAG, the retrieval.
    Take the user query, a chroma database path and collection name, an 
    embedding model, and retrieve the most similar documents to the user query.\n
    'Mode' indicates the retrieval technique: dense, bm25, hybrid, or rerank    .
    """

    threshold = DEFAULT_THRESHOLDS.get(mode) if apply_threshold else None

    if mode == "dense":
        results = dense_search(
            chroma_path=chroma_path, 
            collection_name=collection_name,
            query=query, 
            emb_model_name=emb_model_name, 
            k=k
        )
    elif mode == "bm25":
        results = bm25_search(
            chroma_path=chroma_path, 
            query=query, 
            collection_name=collection_name, 
            emb_model_name=emb_model_name, 
            k=k
        )
    elif mode == "hybrid":
        n = max(k, HYBRID_CANDIDATES)
        results: Retrieved = rrf_fuse([
            dense_search(chroma_path, collection_name, query, emb_model_name, n),
            bm25_search(chroma_path, collection_name, query, emb_model_name, n)
        ])[:k]
    elif mode == "rerank":
        from src.rerank import rerank
        # Dense, then rerank with a cross-encoder
        candidates = retrieve(
            chroma_path, 
            query, 
            collection_name, 
            emb_model_name, 
            k=HYBRID_CANDIDATES, 
            mode="hybrid",
            apply_threshold=False, 
        )
        results = rerank(query=query, candidates=candidates, top_n=k)
    else:
        raise ValueError(f"Unknown mode {mode!r}. Options: {MODES}")

    # Tuples of retrieved documents and relative scores, filtering happens here
    return [(doc, score) for doc, score in results if threshold is None or score >= threshold]


# Which source documents the retrieved chunks came from, deduplicated
def get_sources(relevant_docs: Retrieved) -> set:

    raw_sources = [doc.metadata.get("source", None) for doc, _score in relevant_docs]
    # None as first arg of filter removes None, "", or False
    return set(filter(None, raw_sources))