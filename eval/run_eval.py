
# The harness. Runs the golden set through the real retriever, scores it with
# - eval/metrics.py, prints a table and saves the raw numbers.
#
# Still no LLM: this is the free, deterministic, seconds-long sweep that is ran dozens of times while tuning. 
#
# uv run python -m eval.run_eval
# uv run python -m eval.run_eval --mode dense --collection base_1000_0_pdf_e3s

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import CHROMA_PATH, CHUNKING_STRATEGY, EMBEDDING_MODEL, F_EXT, get_collection_name
from src.retrieval import retrieve
from eval.metrics import Retrieved, score_in_corpus, hit_at_k, is_hit
from eval.paths import EVAL_RESULTS_DIR, GOLDEN_SET_PATH, PROJECT_ROOT, RESULTS_DIR

K_HEADERS = ["k", "hit@k", "recall@k"]
TIER_HEADERS = ["difficulty", "n", "hit@1", "hit@3", "hit@5", "MRR"]

KS = (1, 3, 5, 10, 20)


def load_golden_set(path: Path) -> List[dict]:
    """Read the JSONL. Reuse the validator's loader rather than re-parsing here."""

    entries: List[dict] = []
    with open(path, mode='r', encoding='utf-8') as e:
        for j_line in e.readlines():
            entries.append(json.loads(j_line))

    return entries


def retrieve_all(questions: List[dict], collection_name: str, max_k: int, mode:str, apply_threshold: bool = False) -> List[Tuple[dict, Retrieved]]:
    """Retrieve once per question, at the largest k in the sweep."""

    result: List[dict, Tuple[dict, Retrieved]] = []

    for idx, golden_entry in enumerate(questions, start=1):
        print(f"\rRetrieving {idx}/{len(questions)}...", end="", flush=True)
        relevant_docs = retrieve(
            CHROMA_PATH, 
            golden_entry["question"],
            collection_name,
            EMBEDDING_MODEL,
            # Run once at max k,
            max_k,
            # No threshold, we're doing evaluation on the golden set
            apply_threshold=apply_threshold,
            mode=mode,
        )

        result.append((golden_entry, relevant_docs))

    print("\n")
    return result


def rows_to_markdown(headers: list[str], rows: list[tuple]) -> str:

    # Tables in markdown are built by defining headers first
    # Then, the separators | --- | --- | --- | ...
    # Finally the fields for each row.
    alignment = [":---"] + [":---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    for row in rows:
        cells = ["" if value is None else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Sorts the in_corpus questions into their difficulty tiers and scores each tier separately, so a
# gain on tabular questions shows up instead of being averaged away against the lexical ones.
def score_by_difficulty(results: List[Tuple[dict, Retrieved]], ks: Tuple[int, ...]) -> Dict[str, dict]:
    """score_in_corpus, once per difficulty tier."""

    # Dictionary of { tier: list of results }, keyed on difficulty, not on type
    # { "lexical": [(entry, retrieved), ...], "paraphrased": [...] }
    groups: Dict[str, list] = defaultdict(list)
    for entry, retrieved in results:
        if entry["type"] == "in_corpus":
            # entry: golden entry
            # retrieved: tuple of (document, score)
            groups[entry["difficulty"]].append((entry, retrieved))

    # Constructs a dictionary with a key for each difficulty and their respective scores
    # { lexical: {"n": 15, "hit@1": 0.6, ...}, ... }
    return {
        tier: {
            "n": len(group), **score_in_corpus(group, ks)
        } 
        for tier, group in sorted(groups.items())
    }


# Turns the two score dicts into table bodies: one row per k, and one row per difficulty tier.
# Built in one place so the printed tables and the saved file can never drift apart.
def build_rows(scores: dict, by_difficulty: Dict[str, dict]) -> Tuple[List[tuple], List[tuple]]:
    """(k_rows, tier_rows). Each row is a tuple of already-formatted cells, in header order."""

    # One tuple per k: ("@1", "0.474", "0.447") - matches K_HEADERS
    k_rows = [(f"@{k}", f"{scores[f'hit@{k}']:.3f}", f"{scores[f'recall@{k}']:.3f}") for k in KS]

    # One tuple per tier: ("lexical", 15, "0.600", "0.667", "0.667", "0.648") - matches TIER_HEADERS
    tier_rows = [
        (tier, score["n"], f"{score['hit@1']:.3f}", f"{score['hit@3']:.3f}", f"{score['hit@5']:.3f}", f"{score['mrr']:.3f}")
        for tier, score in by_difficulty.items()
    ]
    return k_rows, tier_rows


# Writes both tables to eval/results/<collection>__<mode>.md, stamped with the time and the commit,
# so a table can be pasted into the README later without re-running the sweep to remember it.
def save_markdown(scores: dict, by_difficulty: Dict[str, dict], collection_name: str, mode:str, n_in_corpus: int) -> Path:

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"{collection_name}__{mode}.md"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    k_rows, tier_rows = build_rows(scores, by_difficulty)

    sections = [
        "# Evaluation results",
        "",
        f"`{collection_name}`, mode `{mode}`, n = {n_in_corpus} in_corpus "
        f"(one question = {100 / n_in_corpus:.1f} points)",
        "",
        f"Generated by `eval.run_eval` on {generated}, commit `{git_commit()}`.",
        "",
        rows_to_markdown(K_HEADERS, k_rows),
        "",
        f"**MRR: {scores['mrr']:.3f}**",
        "",
        "## By difficulty",
        "",
        rows_to_markdown(TIER_HEADERS, tier_rows),
    ]

    out_path.write_text("\n".join(sections) + "\n", encoding='utf-8') 
    return out_path


# Prints the headline table (one row per k) and the difficulty table, with the in-corpus count in
# - the heading, every number below it has a noise floor of 100/n points, so the n travels with them.
def print_table(scores: dict, by_difficulty: Dict[str, dict], collection_name: str, mode: str, n_in_corpus: int) -> None:
    """Markdown, pasteable into the README, carrying its own n."""

    print(f"\n### `{collection_name}`, mode `{mode}`, n = {n_in_corpus} in_corpus "
          f"(one question = {100 / n_in_corpus:.1f} points)\n")

    k_rows, tier_rows = build_rows(scores, by_difficulty)
    print(rows_to_markdown(K_HEADERS, k_rows))
    print(f"\n**MRR: {scores['mrr']:.3f}**\n")
    print(rows_to_markdown(TIER_HEADERS, tier_rows))


# Lists the questions the retriever missed, each with the comuni its top 3 chunks came from.
# The fastest way to see the failure six parallel regolamenti invite: right article, wrong comune.
def print_failures(results: List[Tuple[dict, Retrieved]], k: int = 5) -> None:
    """Every in_corpus miss at k, with the comuni it retrieved instead."""

    print(f"\nMisses at k = {k}:")
    misses: int = 0
    for entry, retrieved in results:
        # if out of corpus or it hits, continue, it's not a miss
        if entry["type"] != "in_corpus" or hit_at_k(retrieved, entry, k) == 1.0:
            continue
        # anything still here is an in_corpus question with no hit in the top k
        misses += 1
        # "firenze_regolamento_edilizio.pdf" -> "firenze"
        top = [Path(doc.metadata["source"]).stem.split("_")[0] for doc, _score in retrieved[:3]]
        print(f"  {entry['id']} [{entry['difficulty']}] top-3 from {top}: {entry['question'][:80]}")
    if not misses:
        print("  none")


# Gathers the two score populations the threshold has to separate: what a CORRECT chunk scores, and
# - what the best chunk scores on a question the corpus cannot answer.
def score_distributions(results: List[Tuple[dict, Retrieved]]) -> Tuple[List[tuple], List[tuple]]:
    """Score of the first correct chunk per in_corpus question; top score per out_of_corpus question."""

    in_scores, out_scores = [], []
    for entry, retrieved in results:
        if entry["type"] == "in_corpus":
            correct = next((score for doc, score in retrieved if is_hit(doc, entry)), None)
            if correct is not None:
                in_scores.append((round(correct, 3), entry["id"]))
        elif retrieved:
            out_scores.append((round(retrieved[0][1], 3), entry["id"]))

    return sorted(in_scores), sorted(out_scores, reverse=True)


# Prints both score lists sorted and says whether one number can sit between them. If they overlap,
# - no threshold can drive refusal here and the LLM's prompt instruction is doing that work instead.
def print_threshold_report(in_scores: List[tuple], out_scores: List[tuple]) -> None:
    """The threshold is chosen by reading this, not by taste."""

    print("\nCorrect-chunk scores, in_corpus (lowest first):")
    print("  " + ", ".join(f"{score} {question_id}" for score, question_id in in_scores))
    print("Top-chunk scores, out_of_corpus (highest first):")
    print("  " + ", ".join(f"{score} {question_id}" for score, question_id in out_scores))

    if in_scores and out_scores:
        lowest_in, highest_out = in_scores[0][0], out_scores[0][0]
        if highest_out < lowest_in:
            print(f"Separable: any threshold in ({highest_out}, {lowest_in}) keeps every hit and refuses every out_of_corpus.")
        else:
            print(f"Overlap: highest out_of_corpus {highest_out} >= lowest in_corpus hit {lowest_in}. "
                  "Thresholding alone cannot drive refusal - a finding for the README.")


# Stamps every results file with the commit that produced it, flagged +dirty when the working tree
# - has uncommitted changes, so it can be traced back to the code behind it.
def git_commit() -> str:
    """Short hash, with +dirty when the tree has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_results(payload: dict, collection_name: str, mode: str) -> Path:
    """Write eval/results/<collection>__<mode>.json."""

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Two underscores: make_tables.py globs "*__*.json", and collection names contain single ones
    out_path = EVAL_RESULTS_DIR / f"{collection_name}__{mode}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    return out_path


def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Score the golden set against a collection.")

    parser.add_argument(
        "--collection",
        default=get_collection_name(CHUNKING_STRATEGY, F_EXT, EMBEDDING_MODEL),
        required=False,
        help="Run `uv run python -m src.run_eval --collection <collection_name> --mode <mode>."
    )
    
    parser.add_argument(
        "--mode",
        default="dense",
        choices=["dense", "bm25", "hybrid", "rerank"],
        required=False,
        help="Run `uv run python -m src.run_eval --collection <collection_name> --mode <mode>."
    )

    return parser.parse_args()


def main() -> int:

    # Parse arguments, load golden set entries
    args = get_args()
    questions: List[dict] = load_golden_set(GOLDEN_SET_PATH)
    n_in_corpus = sum(1 for e in questions if e["type"] == "in_corpus")

    collection_name = args.collection if args.collection else None
    mode = args.mode if args.mode else None

    # Retrieve the list with golden set entries and relevant documents
    results = retrieve_all(
        questions=questions,
        collection_name=collection_name,
        max_k=max(KS),
        mode=mode,
        apply_threshold=False,
    )

    scores_in_corpus: Dict[str, float] = score_in_corpus(
        results, 
        KS,
    )

    by_difficulty = score_by_difficulty(results, KS)

    print_table(scores_in_corpus, by_difficulty, collection_name, mode, n_in_corpus)
    print_failures(results)
    in_scores, out_scores = score_distributions(results)
    print_threshold_report(in_scores, out_scores)

    payload = {
        "collection": args.collection,
        "mode": args.mode,
        "ks": list(KS),
        "threshold": None,
        "embedding_model": EMBEDDING_MODEL,
        "n_questions": len(questions),
        "n_in_corpus": n_in_corpus,
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scores": scores_in_corpus,
        "by_difficulty": by_difficulty,
        "score_distributions": {"in_corpus": in_scores, "out_of_corpus": out_scores},
    }

    md_path = save_markdown(scores_in_corpus, by_difficulty, collection_name, mode, n_in_corpus)
    out_path = save_results(payload, collection_name, mode)
    print(f"\nSaved {out_path.relative_to(PROJECT_ROOT)} and {md_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
