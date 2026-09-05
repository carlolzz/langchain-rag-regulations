
# One config file to handle paths, file extension, embedding models, and chunking strategies

CHROMA_PATH = "db/chroma_db"
DATA_RAW = "data/raw"

F_EXT = "pdf"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_ABBR = {"text-embedding-3-small": "e3s"}
LLM_MODEL = "gpt-4o"

CHUNKING_STRATEGY = "base"
CHUNKING_CFG = {
    "base":         {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 0},
    "overlap":      {"splitter": "recursive", "chunk_size": 1000, "chunk_overlap": 150},
    "article":      {"splitter": "article"},
    "semantic_p95": {"splitter": "semantic", "breakpoint_threshold_amount": 95},
}


def get_collection_name(chunking_strategy: str, ext: str = "pdf", embedding_model: str = EMBEDDING_MODEL) -> str:

    # cfg = {"splitter": ...}
    cfg = CHUNKING_CFG[chunking_strategy]
    # e.g. overlap_1000_150_pdf_rcts_e3s
    return f"{chunking_strategy}_{cfg['chunk_size']}_{cfg['chunk_overlap']}_{ext}_{EMBEDDING_ABBR[embedding_model]}"
