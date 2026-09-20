
# uv run python eval/make_tables.py --mode-collection base_1000_0_pdf_e3s --chunking-mode dense

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from langchain_core.documents import Document
from typing import List, Dict, Tuple

import chromadb

from src.config import CHROMA_PATH, MODES
from eval.paths import EVAL_RESULTS_DIR, PROJECT_ROOT
from eval.run_eval import rows_to_markdown


def load_results() -> dict:
    """
    {(collection, mode): payload} for every results file.\n
    Example: {base_1000_..., dense}: json result file.
    """
    results: dict = {}
    for path in sorted(EVAL_RESULTS_DIR.glob("*__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The payload contains the collection, mode, k and statistics
        results[(payload["collection"], payload["mode"])] = payload
    return results


def collection_stats(collection_name: str) -> tuple:
    """(chunk count, mean chunk count per doc) read straight from Chroma, no embedding model needed."""
    client = chromadb.PersistentClient(path=str(PROJECT_ROOT / CHROMA_PATH))
    try:
        docs: List[Document] = client.get_collection(collection_name).get(include=["documents"])["documents"]
    except Exception:
        return None, None
    return len(docs), round(mean(len(d) for d in docs))


def fmt(payload: dict | None, key: str) -> str:
    return "not built" if payload is None else f"{payload['scores'][key]:.3f}"


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-collection", default="base_1000_0_pdf_e3s")
    parser.add_argument("--chunking-mode", default="dense", help="The mode frozen at the end of Day D.")
    return parser.parse_args()


def main() -> int:
    
    args = get_args()

    results: dict = load_results()
    if not results:
        print(f"No results in {EVAL_RESULTS_DIR}.")
        return 1

    # {(collection, mode) -> payload}
    n_in_corpus = {p["n_in_corpus"] for p in results.values()}

    print(
        f"num in corpus = {sorted(n_in_corpus)} in_corpus questions" 
        + (" <- mixed" if len(n_in_corpus) > 1 else "")
    )

    print(f"\n**Table 1 - retrieval mode** (collection `{args.mode_collection}`)\n")

    # List of tuples to write to the markdown files
    rows: List[Tuple] = []
    for mode in MODES:
        # fmt(p, key) -> p(key)
        p = results.get((args.mode_collection, mode))
        rows.append((mode, fmt(p, "hit@1"), fmt(p, "hit@3"), fmt(p, "hit@5"), fmt(p, "hit@20"), fmt(p, "mrr")))
    print(rows_to_markdown(headers=["Mode", "hit@1", "hit@3", "hit@5", "hit@20", "MRR"], rows=rows))

    print(f"\n**Table 2 - chunking** (mode `{args.chunking_mode}`)\n")
    rows = []
    # collection, mode = strings; p the associated json dict
    for collection in sorted({coll for coll, mode in results if mode == args.chunking_mode}):
        p = results[(collection, args.chunking_mode)]
        chunks, mean_chars = collection_stats(collection)
        rows.append((f"`{collection}`", fmt(p, "hit@1"), fmt(p, "hit@3"), fmt(p, "hit@5"), fmt(p, "mrr"), chunks, mean_chars))
    print(rows_to_markdown(headers=["Collection", "hit@1", "hit@3", "hit@5", "MRR", "chunks", "mean chars"], rows=rows))
    print("\nAdd unbuilt rows by hand, with the reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
