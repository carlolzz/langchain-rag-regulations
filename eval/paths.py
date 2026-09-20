
from pathlib import Path

# Every path is defined here so judge.py can run in its own environment
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
EVAL_RESULTS_DIR = RESULTS_DIR / "eval_results"
RAGAS_PATH = RESULTS_DIR / "ragas_results" / "{collection}_{mode}_{judge}_generation_ragas.json"
ANSWERS_PATH = RESULTS_DIR / "generation_results" / "{collection}_{mode}_generated_answers.json"
SEP_WIDTH = 100


def fmt_path(path: Path, **kwargs):
    """Rename a path with format string. **kwargs return an unpacked dict."""
    renamed_path = path.with_name(
        path.name.format(**kwargs)
    )
    return renamed_path