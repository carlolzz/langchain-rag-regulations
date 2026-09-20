
# uv add --dev pytest
# uv run python -m pytest eval/test_metrics.py -v

import sys
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from eval.metrics import (
    hit_at_k, is_hit, recall_at_k, reciprocal_rank, 
    refused, score_in_corpus, score_out_of_corpus
)

FIRENZE = "data/raw/firenze_regolamento_edilizio.pdf"
TORINO = "data/raw/torino_regolamento_edilizio.pdf"

QUOTE = "la lunghezza del segmento minimo congiungente la parete più avanzata del fabbricato"
Q1 = "un opportuno isolamento da: - umidità, di qualsivoglia origine e natura"
Q2 = "adeguata protezione dall'escursione termica e dalle fonti di rumore"


def make_doc(text: str, source: str = FIRENZE, score: float = 0.5):
    """One (Document, score) pair, the shape retrieve() returns.
    Returning the pair forces the tests to exercise the same unpacking the real code does.
    """
    
    return (Document(page_content=text, metadata={"source": source}), score)


def make_question(quotes, source: str = FIRENZE, qtype: str = "in_corpus") -> dict:
    """A minimal golden-set entry. Only the fields the metrics actually read."""

    return {"id": "qTEST", "source": source, "gold_quotes": quotes, "type": qtype}


# is_hit - the foundation
def test_hit_requires_matching_source():
    """The right text in the wrong document is a miss."""

    question = make_question([QUOTE])
    text = f"Art. 27 - Si intende {QUOTE} e il confine antistante"

    wrong_doc, _ = make_doc(text, source=TORINO)
    right_doc, _ = make_doc(text, source=FIRENZE)

    assert is_hit(wrong_doc, question) is False
    assert is_hit(right_doc, question) is True


def test_normalize_survives_pdf_newlines():
    """A quote split across a newline still matches."""

    question = make_question([QUOTE])
    chunk, _ = make_doc("la lunghezza del segmento minimo\n    congiungente la parete \n più avanzata del fabbricato")

    # The PDF carries U+2019, the golden set was typed with U+0027. The two sides must differ:
    # - with the same apostrophe on both, this passes as a plain substring check and normalize() could be deleted without the test noticing.
    apostrophe_question = make_question(["l'esecuzione di opere di conformazione, sulla base di idonea ordinanza"])
    apostrophe_chunk, _ = make_doc(
        "Il Comune ordina l’esecuzione di opere di conformazione, sulla base di idonea "
        "ordinanza e entro i relativi termini"
    )

    assert is_hit(chunk, question)
    assert is_hit(apostrophe_chunk, apostrophe_question)


# Ranking metrics
def test_reciprocal_rank_is_half_at_rank_two():
    """A hit in second place scores exactly 0.5.
    Pins the off-by-one. If enumerate() starts at 0 this returns 1.0 and the
    error is invisible in aggregate - MRR just looks flattering."""

    question = make_question([QUOTE])
    retrieved = [make_doc("niente di utile"), make_doc(QUOTE), make_doc("altro testo")]

    assert reciprocal_rank(retrieved, question) == 0.5


def test_miss_returns_zero():
    """No hit anywhere means 0.0, not None and not an exception."""

    question = make_question([QUOTE])
    retrieved = [make_doc("niente di utile"), make_doc("altro testo")]

    rr = reciprocal_rank(retrieved, question)
    hit = hit_at_k(retrieved, question, k=2)

    assert rr == 0.0 and isinstance(rr, float)
    assert hit == 0.0 and isinstance(hit, float)


def test_hit_at_k_respects_k():
    """A hit at rank 5 is invisible at k=3."""

    question = make_question([QUOTE])
    retrieved = [make_doc(f"riempitivo{i}") for i in range(4)] + [make_doc(QUOTE)]

    assert hit_at_k(retrieved, question, k=3) == 0.0
    assert hit_at_k(retrieved, question, k=5) == 1.0


# Recall - the multi-quote case
def test_recall_is_partial_when_one_of_two_quotes_found():
    """Two gold quotes, one retrieved: recall = 0.5, hit = 1.0."""

    question = make_question([Q1, Q2])
    retrieved = [make_doc(f"Le costruzioni devono avere {Q1}"), make_doc("niente di utile")]

    assert recall_at_k(retrieved, question, k=2) == 0.5
    assert hit_at_k(retrieved, question, k=2) == 1.0


def test_recall_ignores_quotes_from_the_wrong_source():
    """Both quotes present, but in the wrong document: recall = 0.0."""

    question = make_question([Q1, Q2])
    retrieved = [make_doc(Q1, source=TORINO), make_doc(Q1, source=TORINO)]

    assert recall_at_k(retrieved, question, k=2) == 0.0
