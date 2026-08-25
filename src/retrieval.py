
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def retrieve(chroma_path: str, query: str, k: int = 3, threshold=0.3, collection_name: str = "base_1000_0") -> List[Tuple[Document, float]]:

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_path = PROJECT_ROOT / chroma_path
    persist_path = persist_path.resolve()

    db = Chroma(
        persist_directory=persist_path,
        embedding_function=embedding_model,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # Search the db for similar results. The relevance score is 1 - cosine distance,
    # Only chunks scoring >= threshold are kept.
    relevant_docs = db.similarity_search_with_relevance_scores(
        query,
        k=3,
        score_threshold=threshold,
    )

    if not relevant_docs:
        return []
    return relevant_docs


# Which source documents the retrieved chunks came from, deduplicated
def get_sources(relevant_docs: List[Tuple[Document, float]]) -> set:

    raw_sources = [doc.metadata.get("source", None) for doc, _score in relevant_docs]
    # None as first arg of filter removes None, "", or False
    return set(filter(None, raw_sources))
