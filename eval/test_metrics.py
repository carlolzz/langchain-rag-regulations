
# Tests for the ruler itself.
#
# Half an hour here is the best-spent half hour in the evaluation work, and most
# portfolio projects skip it. The argument: a broken metric fails SILENTLY and
# looks exactly like a broken retriever. Without these you could spend the whole
# ablation tuning chunk sizes to fix an off-by-one in reciprocal_rank.
#
# Everything is hand-built - no Chroma, no PDFs, no API. That is deliberate: a
# test that needs the corpus is slow, and one that needs the network is flaky.
#
#   uv add --dev pytest
#   uv run python -m pytest eval/test_metrics.py -v

import sys
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import hit_at_k, is_hit, recall_at_k, reciprocal_rank

FIRENZE = "data/raw/firenze_regolamento_edilizio.pdf"
TORINO = "data/raw/torino_regolamento_edilizio.pdf"


def make_doc(text: str, source: str = FIRENZE, score: float = 0.5):
    """One (Document, score) pair, the shape retrieve() returns.

    Returning the PAIR rather than the Document is the point: it forces the
    tests to exercise the same unpacking the real code does.
    """
    return (Document(page_content=text, metadata={"source": source}), score)


def make_question(quotes, source: str = FIRENZE, qtype: str = "in_corpus") -> dict:
    """A minimal golden-set entry. Only the fields the metrics actually read."""
    return {"id": "qTEST", "source": source, "gold_quotes": quotes, "type": qtype}


# ---------------------------------------------------------------------------
# is_hit - the foundation
# ---------------------------------------------------------------------------
def test_hit_requires_matching_source():
    """The right text in the WRONG document is a miss.

    This is the single most important test in the file, because it encodes the
    property that makes this corpus interesting. Six parallel regolamenti share
    boilerplate and adopt the national definizioni uniformi verbatim, so the
    same sentence genuinely appears in several documents. A metric that ignored
    source would score those as hits and quietly inflate every number.

    TODO 1: build one chunk whose text contains the quote but whose source is
    TORINO, and assert is_hit(...) is False against a FIRENZE question. Then
    assert the same text from FIRENZE IS a hit, so the test proves the source
    check is what's failing it - not the text.
    """
    raise NotImplementedError("TODO 1")


def test_normalize_survives_pdf_newlines():
    """A quote split across a newline still matches.

    PyPDFLoader scatters newlines mid-sentence, so the extracted text almost
    never matches what you copied out of a PDF viewer. normalize() collapses
    whitespace runs; this test pins that behaviour so a future "tidy-up" of
    normalize() cannot silently break every quote at once.

    TODO 2: put a "\\n" (and some stray spaces) in the MIDDLE of the chunk text
    where the quote sits, and assert it still hits.

    Worth adding while you are here: a case with the typographic apostrophe
    U+2019 in the chunk and the ASCII one in the quote. The Firenze PDF mixes
    both inside one sentence, so this is a real input, not a hypothetical.
    """
    raise NotImplementedError("TODO 2")


# ---------------------------------------------------------------------------
# ranking metrics
# ---------------------------------------------------------------------------
def test_reciprocal_rank_is_half_at_rank_two():
    """A hit in second place scores exactly 0.5.

    Pins the off-by-one. If enumerate() starts at 0 this returns 1.0 and the
    error is invisible in aggregate - MRR just looks flattering.

    TODO 3: build a 3-chunk result where only the SECOND contains the quote,
    and assert reciprocal_rank(...) == 0.5. Assert the exact float; this is
    integer division of 1 by a small int, so there is no floating-point
    fuzziness to be tolerant of.
    """
    raise NotImplementedError("TODO 3")


def test_miss_returns_zero():
    """No hit anywhere means 0.0, not None and not an exception.

    TODO 4: assert reciprocal_rank == 0.0 AND hit_at_k == 0.0 for a result set
    containing none of the gold quotes. The types matter - these get averaged.
    """
    raise NotImplementedError("TODO 4")


def test_hit_at_k_respects_k():
    """A hit at rank 5 is invisible at k=3.

    This is what separates hit@k from reciprocal_rank, and it is the property
    the whole k-sweep on Day 3 depends on. If this passes trivially, the sweep
    will produce a suspiciously flat line - which is exactly the fictitious
    "retrieval plateaus after k=3" finding the old retrieve() bug would have
    produced.

    TODO 5: one result set, hit at position 5. Assert hit_at_k(..., k=3) == 0.0
    and hit_at_k(..., k=5) == 1.0.
    """
    raise NotImplementedError("TODO 5")


# ---------------------------------------------------------------------------
# recall - the multi-quote case
# ---------------------------------------------------------------------------
def test_recall_is_partial_when_one_of_two_quotes_found():
    """Two gold quotes, one retrieved: recall = 0.5, hit = 1.0.

    The case q001 exists to exercise, and the only place where recall@k and
    hit@k disagree. If this test passes but recall_at_k is secretly implemented
    as hit_at_k, you would not notice on a single-quote set - which is most of
    your golden set.

    TODO 6: assert BOTH in the same test - recall_at_k == 0.5 and hit_at_k == 1.0
    on identical inputs. Asserting them together is what makes the distinction
    visible; asserting recall alone would let a hit@k impostor through.
    """
    raise NotImplementedError("TODO 6")


def test_recall_ignores_quotes_from_the_wrong_source():
    """Both quotes present, but in the wrong document: recall = 0.0.

    Same trap as test_hit_requires_matching_source, one level up. It is easy to
    write recall_at_k so the source check applies to the chunk loop but not to
    each quote, and this catches that.

    TODO 7: build chunks from TORINO containing both FIRENZE quotes; assert 0.0.
    """
    raise NotImplementedError("TODO 7")
