
# Run the tests with:  uv run python -m pytest eval/test_metrics.py -v

import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


from langchain_core.documents import Document

REFUSAL_PREFIX: str = "Could not find any relevant information"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# U+2019 -> U+0027 and strips soft hyphens, both of which this corpus needs
# Scorer normalisez differently from the validator, quote could pass validation and then never score a hit 
from eval.validate_golden_set import normalize

# What retrieve() hands back
Retrieved = List[Tuple[Document, float]]


def is_hit(chunk: Document, question: dict) -> bool:
    """True if this chunk is a correct retrieval for this question."""

	# A chunk hits when it comes from the right document, and it contains at least one gold quote
    chunk_source: str = chunk.metadata.get("source")
    question_source: str = question.get("source")

    # Source mismatch
    if chunk_source != question_source:
        return False

    normalized_chunk: str = normalize(chunk.page_content)
    if any(normalize(q) in normalized_chunk for q in question.get("gold_quotes")) and chunk_source == question_source:
        return True

    return False


# The three retrieval metrics. 
def hit_at_k(retrieved: Retrieved, question: dict, k: int) -> float:
    """1.0 if any of the top k retrieved chunks hits, else 0.0."""

    # Slice to the top k
    retrieved_cut: List[Tuple[Document, float]]  = retrieved[:k]

    # doc[0] = Document
    return float(any(is_hit(doc, question) for doc, _score in retrieved_cut))


def recall_at_k(retrieved: Retrieved, question: dict, k: int) -> float:
    """Fraction of this question's gold quotes present in the top k."""

    gold_quotes = question.get("gold_quotes")
    if not gold_quotes:
        raise ValueError(f"{question.get('id')} has no gold quotes, recall is undefined.")

    retrieved_cut = retrieved[:k]
    texts = [normalize(doc.page_content) for doc, _score in retrieved_cut if doc.metadata.get("source") == question.get("source")]

    found: float = sum(any(normalize(q) in text for text in texts) for q in gold_quotes)

    return found / len(gold_quotes)
    

def reciprocal_rank(retrieved: Retrieved, question: dict) -> float:
    """1 / rank of the first hit, or 0.0 if there is none. Mean over questions = MRR."""

    for rank, (doc, _score) in enumerate(retrieved, start=1):
        if is_hit(doc, question):
            return 1.0 / rank
        
    return 0.0


def refused(answer: str) -> bool:
    """True if the system declined to answer."""
    return normalize(answer).startswith(REFUSAL_PREFIX)


# Aggregation, this is what run_eval.py calls.
def score_in_corpus(results: List[Tuple[dict, Retrieved]], ks: Tuple[int, ...]) -> Dict[str, float]:
    """Mean hit@k / recall@k for each k, plus MRR, over in_corpus questions only."""

    # Filtered results
    f_res = [(entry, retrieved) for entry, retrieved in results if entry.get("type") == "in_corpus"]

    scores: Dict[str, float] = {}

    for k in ks:
        scores[f"hit@{k}"] = mean(hit_at_k(retrieved, entry, k) for entry, retrieved in f_res)
        scores[f"recall@{k}"] = mean(recall_at_k(retrieved, entry, k) for entry, retrieved in f_res)

    scores["mrr"] = mean(reciprocal_rank(retrieved, entry) for entry, retrieved in f_res)

    raise scores


def score_out_of_corpus(answers: List[Tuple[dict, str]]) -> Dict[str, float]:
    """Refusal rate on out_of_corpus questions, false-refusal rate on in_corpus ones."""

    out_of_corpus_answers = [(gold, answer) for gold, answer in answers if gold["type"] == "out_of_corpus"]
    in_corpus_answers = [(gold, answer) for gold, answer in answers if gold["type"] == "in_corpus"]

    out_n_refused = len([answer for _gold, answer in out_of_corpus_answers if refused(answer)])
    out_n = len(out_of_corpus_answers)

    in_n_refused = len([answer for _gold, answer in in_corpus_answers if refused(answer)])
    in_n = len(in_corpus_answers)

    if in_n_refused == 0 or out_n_refused == 0:
        raise ValueError(f"Error, in_corpus entries or out_of_corpus entries are empty, this will cause a ZeroDivisionError.")

    return {
        "refusal_rate": out_n_refused / out_n,
        "false_refusal_rate": in_n_refused / in_n
    }