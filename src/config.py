
# One config file to handle paths, file extension, embedding models, and chunking strategies

CHROMA_PATH: str = "db/chroma_db"
DATA_RAW: str = "data/raw"

F_EXT: str = "pdf"
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_ABBR: dict = {"text-embedding-3-small": "e3s"}
LLM_MODEL: str = "gpt-4o"

CHUNKING_STRATEGY: str = "base"
CHUNKING_CFG: dict = {
    "base":         {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 0},
    "overlap":      {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 150},
    "article":      {"splitter": "article"},
    "semantic_p95": {"splitter": "semantic", "breakpoint_threshold_amount": 95},
}


def get_collection_name(chunking_strategy: str, ext: str = "pdf", embedding_model: str = EMBEDDING_MODEL) -> str:

    # cfg = {"splitter": ...}
    cfg: dict = CHUNKING_CFG[chunking_strategy]

    cfg_kwargs = "_".join(map(str, list(cfg[chunking_strategy].values())))
    # Example: "base_recursive_1000_0_pdf_e3s"
    filename: str = f"{cfg_kwargs}_{ext}_{EMBEDDING_ABBR[embedding_model]}"

    return filename
