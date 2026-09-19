
# Run the llm generation part of RAG against the golden set questions

import json
import sys
from pathlib import Path
from typing import List, Dict

from src.config import CHROMA_PATH, CHUNKING_STRATEGY, EMBEDDING_MODEL, F_EXT, LLM_MODEL, get_collection_name
from src.generate import generate_answer
from src.retrieval import retrieve
from eval.metrics import score_out_of_corpus
from eval.run_eval import GOLDEN_SET_PATH, RESULTS_DIR, git_commit
from eval.validate_golden_set import load_entries


# Frozen configuration
MODE: str = "dense"
K: int = 3
THRESHOLD: float = 0.3
NO_CONTEXT_MSG = "Could not find any relevant information in the corpus."
ANSWERS_PATH = RESULTS_DIR / "generated_answers.json"


def generate_all(questions: List[Dict], collection_name: str) -> List:
    """Load the golden set json file, iterate through all entries and generate answers with the LLM with the relevant context."""

    rows: List = []
    for idx, entry in enumerate(questions, start=1):
        print(f"\rGenerating {idx}/{len(questions)}...", end="", flush=True)
        retrieved = retrieve(
            chroma_path=CHROMA_PATH, 
            query=entry["question"], 
            collection_name=collection_name, 
            emb_model_name=EMBEDDING_MODEL,
            k=K,
            mode=MODE,
            apply_threshold=True,
        )

        answer = generate_answer(relevant_docs=retrieved, user_query=entry["question"], model_name=LLM_MODEL) if retrieved else NO_CONTEXT_MSG

        rows.append({
            "id": entry["id"],
            "type": entry["type"],
            "question" : entry["question"],
            "reference": entry["expected_answer"],
            "contexts": [doc.page_content for doc, _score in retrieved],
            "answer": answer,

        })

        print("\n")
        return rows


def main() -> int:

    questions = load_entries(GOLDEN_SET_PATH)

    # Need a way to get the golden entries by id for later
    entry_by_id = {entry["id"] for entry in questions}
   
    collection = get_collection_name(CHUNKING_STRATEGY, F_EXT, EMBEDDING_MODEL)

    if ANSWERS_PATH.exists():
        rows = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))["rows"]
    else:
        rows = generate_all(questions=questions, collection_name=collection)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "collection": collection,
            "mode" : MODE,
            "k": K,
            "threshold": THRESHOLD,
            "llm": LLM_MODEL,
            "git_commit": git_commit()
        }
        ANSWERS_PATH.write_text(
            json.dumps({"config": config, "rows": rows}, indent=2, ensure_ascii=False
        ), encoding="utf-8")

        # Wants golden set entries and respective answers
        # Display the refusal rate on out of corpus entries 
        print(score_out_of_corpus([(entry_by_id["id"], rows["answer"]) for row in rows]))
        return 0


    if __name__ == "__main__":
        sys.exit(main())    
    