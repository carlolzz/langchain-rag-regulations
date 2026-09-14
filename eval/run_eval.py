
# The harness. Runs the golden set through the real retriever, scores it with
# eval/metrics.py, prints a table and saves the raw numbers.
#
# Still no LLM: this is the free, deterministic, seconds-long sweep you run
# dozens of times while tuning. Generation metrics are a separate run on Day 8.
#
#   uv run python eval/run_eval.py
#   uv run python eval/run_eval.py --mode dense --collection base_1000_0_pdf_e3s

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHROMA_PATH, CHUNKING_STRATEGY, EMBEDDING_MODEL, F_EXT, get_collection_name
from src.retrieval import retrieve
from eval.metrics import Retrieved, score_in_corpus

GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# k=20 is in the sweep because Day 5 needs hit@20 as the ceiling a reranker
# could exploit. Collecting it now is free; re-running the sweep later is not.
KS = (1, 3, 5, 10, 20)


def load_golden_set(path: Path) -> List[dict]:
    """Read the JSONL. Reuse the validator's loader rather than re-parsing here.

    TODO 1: import load_entries from eval.validate_golden_set and call it. One
    parser, one set of error messages. If you find yourself wanting a second
    JSONL reader, that is a sign the first one should have been shared.
    """
    raise NotImplementedError("TODO 1: load the golden set")


def retrieve_all(questions: List[dict], collection_name: str, max_k: int) -> List[Tuple[dict, Retrieved]]:
    """Retrieve once per question, at the largest k in the sweep.

    THE CRITICAL LINE, and the one thing to get right today:

        threshold=0.0

    retrieve() defaults to threshold=0.3 and filters BEFORE returning, so with
    the default you would be measuring ranking and thresholding at the same time
    and could not tell which one moved a number. Measure raw ranking first;
    choose the threshold afterwards, from the score distributions (see 3.2 in
    PLAN.md). Passing 0.0 explicitly is not optional.

    DESIGN QUESTION, decide it deliberately:
      The obvious implementation is a nested loop - for each k, re-retrieve.
      That issues len(questions) * len(KS) queries and re-embeds the same query
      five times. But retrieval at k=20 already CONTAINS the results at k=3:
      the top 3 of a top-20 ranking are the top 3. So you can retrieve once at
      max(KS) and let the metrics slice.
      Is that always true? It is for dense similarity search. Write down why,
      because it stops being true the moment RRF fusion enters on Day 4 - fusing
      two top-20 lists is not guaranteed to agree with fusing two top-3 lists.
      That is a real trap and worth a comment in the code.

    TODO 2:
      a. Retrieve once per question at max_k with threshold=0.0.
      b. Return a list of (question, retrieved) pairs so the scoring functions
         get both halves. Keep the entry itself, not just its id - the metrics
         need source, gold_quotes and type.
      c. Print progress. 25 questions is fast, but silence during an API call
         is indistinguishable from a hang.

    EFFICIENCY NOTE: retrieve() constructs OpenAIEmbeddings and opens Chroma on
    EVERY call. Across 25 questions x several modes that is a lot of reopens.
    Hoist the store into a module-level cache keyed by (persist_dir, collection)
    when it actually bites - not before.
    """
    raise NotImplementedError("TODO 2: run the retrieval sweep")


def print_table(scores: dict, collection_name: str, mode: str) -> None:
    """One row per k, columns hit@k / recall@k, plus MRR once.

    TODO 3: print something you would paste straight into the README. f-string
    width specifiers (f"{value:>8.3f}") are enough; do not add a dependency for
    this.

    Print the QUESTION COUNT in the header. Every number here has a noise floor
    set by it - at 25 questions hit@k moves in 4-point steps - and a table that
    does not carry its own n invites you to over-read a 2-point difference six
    weeks from now.
    """
    raise NotImplementedError("TODO 3: print the results table")


def save_results(scores: dict, collection_name: str, mode: str) -> Path:
    """Write eval/results/<collection>__<mode>.json.

    One file per run, so the Day 9 ablation is loading JSON rather than
    re-running anything. That matters more than it sounds: it means the final
    table is reproducible without spending money or waiting.

    TODO 4:
      a. mkdir the results dir with parents=True, exist_ok=True.
      b. Save the scores AND the run's provenance: collection, mode, k values,
         threshold, embedding model, question count, and the git commit if you
         can get it cheaply. A results file that does not say what produced it
         is a number you will not trust in a week.
    """
    raise NotImplementedError("TODO 4: save the results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the golden set against a collection.")
    # TODO 5: add --collection (default get_collection_name(CHUNKING_STRATEGY, F_EXT,
    # EMBEDDING_MODEL)) and --mode (default "dense", choices dense/bm25/hybrid).
    #
    # Add --mode NOW, on Day 3, even though only "dense" works. Day 4 adds the
    # others, and a flag that already exists is a flag whose results filenames
    # are already correct. The alternative is renaming every results file later.
    #
    # Then add mode="dense" to retrieve()'s signature in src/retrieval.py for the
    # same reason - the signature has already changed once under this harness and
    # that is exactly what broke ask.py.
    args = parser.parse_args()

    questions = load_golden_set(GOLDEN_SET_PATH)

    # TODO 6: wire it together - retrieve_all, score_in_corpus, print_table,
    # save_results. Keep main() thin: it should read as a summary of the steps,
    # with no logic of its own worth testing.

    # TODO 7 (the thing to do BEFORE trusting any of this):
    #   Dump the score of the correct chunk for every in_corpus question, and the
    #   score of the TOP chunk for every out_of_corpus question. Print both sets
    #   sorted. That is how the threshold gets chosen from data instead of taste.
    #
    #   If the two distributions overlap heavily, that is a FINDING, not a
    #   failure: it means score thresholding alone cannot drive refusal on this
    #   corpus and the LLM-side instruction is doing the real work. Say so in the
    #   README - it is more interesting than a clean separation and it shows you
    #   looked.

    raise NotImplementedError("TODO 6: wire up main()")


if __name__ == "__main__":
    sys.exit(main())
