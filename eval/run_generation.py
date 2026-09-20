
# Run the llm generation part of RAG against the golden set questions

import json
import sys
from pathlib import Path
from typing import List, Dict

from src.config import CHROMA_PATH, CHUNKING_STRATEGY, EMBEDDING_MODEL, F_EXT, LLM_MODEL, get_collection_name
from src.generate import generate_answer
from src.retrieval import retrieve
from eval.metrics import score_out_of_corpus
from eval.paths import GOLDEN_SET_PATH
from eval.run_eval import git_commit
from eval.validate_golden_set import load_entries
from eval.paths import fmt_path, ANSWERS_PATH


# Frozen configuration
# dense, bm25, hybrid, rerank
MODE: str = "dense"
K: int = 3
THRESHOLD: float = 0.3
NO_CONTEXT_MSG = "Could not find any relevant information in the corpus."


def generate_all(questions: List[Dict], mode: str, collection_name: str) -> List:
    """
    Load the golden set json file, iterate through all entries, and generate answers with the LLM with the relevant context.
    Retrieval and augmented generation; this will be fed to Ragas.\n 
    The result are the generated answer to the golden set questions and the relevant documents retrieved from a certain collection and using a certain mode.\n
    Example: retrieving documents using bm25 from the collection 'collection base_1000_...' which was built using a certain chunking strategy and embedding model. 
    """

    rows: List = []

    for idx, entry in enumerate(questions, start=1):

        print(f"\rGenerating {idx}/{len(questions)}...", end="", flush=True)
        retrieved = retrieve(
            chroma_path=CHROMA_PATH, 
            query=entry["question"], 
            collection_name=collection_name, 
            emb_model_name=EMBEDDING_MODEL,
            k=K,
            mode=mode,
            apply_threshold=True,
        )

        answer = generate_answer(
            relevant_docs=retrieved, 
            user_query=entry["question"], 
            model_name=LLM_MODEL
        ) if retrieved else NO_CONTEXT_MSG

        # {[q001, in_corpus, question, reference, contexts, answer], ...}
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

    questions: List[Dict] = load_entries(GOLDEN_SET_PATH)

    # Need a way to get the golden entries by id for later
    # q001 -> golden entry
    entry_by_id: Dict[str, Dict] = {entry["id"]: entry for entry in questions}
   
    collection: str = get_collection_name(CHUNKING_STRATEGY, F_EXT, EMBEDDING_MODEL)

    # Rename the path so we have 1 result for each collection-mode pair
    # "{base_1000_...}_{dense}_generated_answers.json"
    answers_path: Path = fmt_path(ANSWERS_PATH, collection=collection, mode=MODE)

    # If results exists already, load the rows
    if answers_path.exists():
        rows = json.loads(answers_path.read_text(encoding="utf-8"))["rows"]
    else:
        rows = generate_all(questions=questions, mode=MODE, collection_name=collection)
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        # What collection was used (to retrieve the relevant docs from)
        # What mode was used to retrieve such data (dense, bm25...)
        # What k was used for how many relevant docs to retrieve
        # What LLM model was used to generate the answers
        config = {
            "collection": collection,
            "mode": MODE,
            "k": K,
            "threshold": THRESHOLD,
            "llm": LLM_MODEL,
            "git_commit": git_commit()
        }
        answers_path.write_text(
            json.dumps({"config": config, "rows": rows}, indent=2, ensure_ascii=False
        ), encoding="utf-8")

    # Wants golden set entries and respective answers
    # Display the refusal rate on out of corpus entries 
    print(score_out_of_corpus([(entry_by_id[row["id"]], row["answer"]) for row in rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main())    
    