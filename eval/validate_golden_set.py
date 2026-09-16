
"""
Validate data/eval/golden_set.jsonl before it is ever used to score a retriever.

A golden set with an unverifiable quote is worse than no golden set.

Quotes are checked against the same text the retriever sees.
Run from the project root:  python eval/validate_golden_set.py

Exit code is 1 if any error was found
"""

import io
import json
import re
import sys
from collections import defaultdict, Counter
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHUNKING_STRATEGY, F_EXT
from src.ingest import load_documents, clean_documents, strip_page_furniture, split_documents

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"

REQUIRED_FIELDS = ("id", "question", "expected_answer", "source", "gold_quotes", "type", "difficulty")
VALID_TYPES = {"in_corpus", "out_of_corpus"}
VALID_DIFFICULTIES = {"lexical", "paraphrased", "tabular"}

# A quote longer than this is asking for a chunk boundary to fall inside it.
MAX_QUOTE_WORDS = 20


# The single definition of "the same text", applied to both sides of every comparison.
# U+2019 -> U+0027 because these PDFs mix the two within one sentence:
# "s'intende" is ASCII but "dell'edificio" is typographic.
def normalize(text: str) -> str:
    """Normalizes text be handling U+2019 and U+0027 unicode characters."""
    text = text.replace("’", "'").replace("­", "")
    return re.sub(r"\s+", " ", text).strip().lower()


# Returns the corpus at two granularities, because they catch different bugs.
# Returns -> [pages_by_source : { source -> [(page number, normalized page text)] }, { source -> [normalized chunk text]} ]
def build_corpus() -> Tuple[Dict[str, List[Tuple[int, str]]], Dict[str, List[str]]]:
    """Run the real ingestion pipeline, minus the embedding call."""

    print("Rebuilding the corpus through the ingestion pipeline (no embeddings)...")

    # ingest.py is chatty; its progress output is noise here.
    with redirect_stdout(io.StringIO()):
        docs = load_documents(ext=F_EXT)
        docs = strip_page_furniture(clean_documents(docs))
        chunks = split_documents(docs, CHUNKING_STRATEGY, F_EXT)

    # { source -> [(page number, normalized page text)] }
    # { source -> [(0, "..."), (1, "...")] }
    pages_by_source: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for doc in docs:
        pages_by_source[doc.metadata["source"]].append(
            (doc.metadata.get("page"), normalize(doc.page_content))
        )

    # { source -> [normalized chunk text] }
    chunks_by_source: Dict[str, List[str]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk.metadata["source"]].append(normalize(chunk.page_content))

    print(f"{len(docs)} pages and {len(chunks)} chunks across {len(pages_by_source)} documents.\n")
    return pages_by_source, chunks_by_source


def load_entries(path: Path) -> List[dict]:
    """Loads golden set entries."""

    if not path.exists():
        raise FileNotFoundError(f"Error, golden set not found at {path}.")

    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Error, line {lineno} of {path.name} is not valid JSON: {exc}") from exc

    return entries


# Shape of the record, nothing here touches the corpus
def check_schema(entry: dict, seen_ids: set) -> Tuple[List[str], List[str]]:

    errors, warnings = [], []

    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        errors.append(f"missing field(s): {', '.join(missing)}")
        return errors, warnings

    if entry["id"] in seen_ids:
        errors.append(f"duplicate id '{entry['id']}'")
    seen_ids.add(entry["id"])

    if entry["type"] not in VALID_TYPES:
        errors.append(f"type '{entry['type']}' not in {sorted(VALID_TYPES)}")

    if entry["difficulty"] not in VALID_DIFFICULTIES:
        errors.append(f"difficulty '{entry['difficulty']}' not in {sorted(VALID_DIFFICULTIES)}")

    if not isinstance(entry["gold_quotes"], list):
        errors.append("gold_quotes must be a list, even when it holds one quote")

    # out_of_corpus questions test refusal
    if entry["type"] == "out_of_corpus" and entry.get("gold_quotes"):
        errors.append("out_of_corpus entry must have empty gold_quotes")

    if entry["type"] == "in_corpus" and not entry.get("gold_quotes"):
        errors.append("in_corpus entry has no gold_quotes, so it has no hit test")

    return errors, warnings


# Where the quotes meet the corpus.
def check_quotes(entry: dict, pages_by_source, chunks_by_source) -> Tuple[List[str], List[str]]:

    errors, warnings = [], []
    source = entry["source"]

    if entry["type"] == "out_of_corpus":
        return errors, warnings

    if source not in pages_by_source:
        errors.append(f"source '{source}' is not in the ingested corpus")
        return errors, warnings

    for quote in entry["gold_quotes"]:
        needle: str = normalize(quote)
        preview: str = quote[:55] + ("..." if len(quote) > 55 else "")

        n_words: int = len(needle.split())
        if n_words > MAX_QUOTE_WORDS:
            warnings.append(f"quote is {n_words} words, keep to ~10-15: '{preview}'")

        # Is it in its own document 
        hit_pages: List = [page for page, text in pages_by_source[source] if needle in text]
        if not hit_pages:
            errors.append(f"quote not found in {Path(source).name}: '{preview}'")
            continue

        # Does it survive chunking intact
        # needle is  normalized gold_quote
        if not any(needle in chunk for chunk in chunks_by_source[source]):
            errors.append(f"quote straddles a chunk boundary, unretrievable: '{preview}'")

        # Check if another source other than the one in the golden set entry has any golden quotes that match
        others = [
            Path(s).name for s in pages_by_source if s != source and any(needle in text for _pg, text in pages_by_source[s])
        ]
        if others:
            errors.append(f"quote is ambiguous, also in {', '.join(others)}: '{preview}'")

        # Convention: PyPDFLoader's 0-indexed metadata["page"]
        if "page" in entry and entry["page"] not in hit_pages:
            warnings.append(f"page {entry['page']} declared but quote is on page(s) {hit_pages}!")

    return errors, warnings


def main() -> int:

    pages_by_source, chunks_by_source = build_corpus()
    entries: List[dict] = load_entries(GOLDEN_SET_PATH)

    if not entries:
        print(f"{GOLDEN_SET_PATH.name} is empty. Nothing to validate.")
        return 0

    seen_ids: set = set()
    n_errors = n_warnings = 0

    for entry in entries:
        errors, warnings = check_schema(entry, seen_ids)
        # Only worth touching the corpus if the record is well formed.
        if not errors:
            quote_errors, quote_warnings = check_quotes(entry, pages_by_source, chunks_by_source)
            errors += quote_errors
            warnings += quote_warnings

        entry_id = entry.get("id", "<no id>")
        if not errors and not warnings:
            print(f"  OK    {entry_id}")
        for message in errors:
            print(f"  ERROR {entry_id}: {message}")
        for message in warnings:
            print(f"  WARNING  {entry_id}: {message}")

        n_errors += len(errors)
        n_warnings += len(warnings)

    n_in = sum(1 for e in entries if e.get("type") == "in_corpus")
    print(f"\n{len(entries)} entries ({n_in} in_corpus, {len(entries) - n_in} out_of_corpus): "
          f"{n_errors} error(s), {n_warnings} warning(s).")

    by_difficulty = Counter(e.get("difficulty") for e in entries if e.get("type") == "in_corpus")
    print(f"difficulty mix: {dict(by_difficulty)}")

    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
