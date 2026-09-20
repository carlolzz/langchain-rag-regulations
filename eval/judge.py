
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path
import argparse
import json

from eval.paths import ANSWERS_PATH, RAGAS_PATH, fmt_path, SEP_WIDTH
from src.config import CHUNKING_STRATEGY, EMBEDDING_MODEL, F_EXT, MODES, RAGAS_LLM, get_collection_name

# judge.py doesn't import src.generate or src.retrieval
load_dotenv()


def load_rag_data(path: Path, key: str):
    return json.loads(path.read_text(encoding='utf-8'))[key]


# ragas needs langchain-community<0.4, langchain-experimental needs >=0.4.2, so the judge runs in its own environment:
# uv run --isolated --no-project --with "ragas==0.4.3" --with "langchain-community<0.4" --with langchain-openai --with pandas --with python-dotenv python -m eval.judge
def llm_as_a_judge(collection: str, mode: str):

    # Load result path, load associated config and answer rows
    # "{base_1000_...}_{dense}_generated_answers.json"
    p: Path = fmt_path(ANSWERS_PATH, collection=collection, mode=mode)
    config: Dict = load_rag_data(p, "config")
    rows = load_rag_data(p, "rows")
    
    samples: List[Dict] = [
        {
            "user_input": row["question"],
            "retrieved_contexts": row["contexts"],
            "response": row["answer"],
            "reference": row["reference"],
        }
        for row in rows if row["type"] == "in_corpus" and row["contexts"]
    ]

    print("=" * SEP_WIDTH)
    print(f"\nJudging {len(samples)} of {len(rows)} rows.")
    print("Evaluating with Ragas...")
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextRecall()],
        llm=LangchainLLMWrapper(ChatOpenAI(model=RAGAS_LLM, temperature=0)),
        embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    )

    # "{base_1000_...}_{dense}_{gpt_4o_mini}_generation_ragas.json"
    judge_model_name: str = RAGAS_LLM.replace("-", "_")
    # The collection and mode come from the answers config, not the arguments, so the name always matches the data judged
    out_path: Path = fmt_path(RAGAS_PATH, collection=config["collection"], mode=config["mode"], judge=judge_model_name)
    # Only the parent is a directory, RAGAS_PATH itself is the file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pandas().to_json(out_path, orient="records", force_ascii=False, indent=2)
    print(f"Saved {out_path}")


def main() -> None:
    
    parser = argparse.ArgumentParser(description="Score generated answers with Ragas")
    parser.add_argument("--collection", default=get_collection_name(CHUNKING_STRATEGY, F_EXT, EMBEDDING_MODEL))
    parser.add_argument("--mode", default="dense", choices=MODES)
    args = parser.parse_args()

    llm_as_a_judge(collection=args.collection, mode=args.mode)


if __name__ == "__main__":
    main()