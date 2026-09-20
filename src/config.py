
# One config file to handle paths, file extension, embedding models, and chunking strategies

CHROMA_PATH: str = "db/chroma_db"
DATA_RAW: str = "data/raw"

# File extension, determined here, compared to the one the function detects automatically
F_EXT: str = "pdf"
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_ABBR: dict[str, str] = {
    "text-embedding-3-small": "e3s"
}
LLM_MODEL: str = "gpt-4o"
# TODO: Anthropic model as the judge 
RAGAS_LLM: str = "gpt-4o-mini"
SEP_WIDTH: int = 100

# Default thresholds for the retrieval similarity
DEFAULT_THRESHOLDS: dict[str, float | None] = {"dense": 0.3, "bm25": None, "hybrid": None, "rerank": None}
MODES: list[str] = ["dense", "bm25", "hybrid", "rerank"]
CHUNKING_STRATEGY: str = "base"
CHUNKING_CFG: dict = {
    "base":         {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 0},
    "overlap":      {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 150},
    "article":      {"splitter": "article"},
    "semantic_p95": {"splitter": "semantic", "breakpoint_threshold_amount": 95},
}


def get_collection_name(chunking_strategy: str, ext: str = "pdf", embedding_model: str = EMBEDDING_MODEL) -> str:
    """Constructs the collection name based on the given configuration.\n
    Examples: base_1000_0_pdf_e3s, overlap_1000_150_pdf_e3s, article_2000_0_pdf_e3s, semantic_p95_pdf_e3s."""

    # cfg = {"splitter": ...}
    cfg: dict = CHUNKING_CFG[chunking_strategy]

    # Only size parameters go in the name, and only when the splitter has them
    # * unpacks the list to individual elements
    size_params = [str(cfg[key]) for key in ("chunk_size", "chunk_overlap") if key in cfg]

    return '_'.join([chunking_strategy, *size_params, ext, EMBEDDING_ABBR[embedding_model]])