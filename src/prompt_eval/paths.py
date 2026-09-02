from pathlib import Path

DATA_DIR = Path("data")
DEFAULT_DATASET_FILE = DATA_DIR / "dataset.json"

OUTPUT_DIR = Path("output")
DEFAULT_SOLUTIONS_FILE = OUTPUT_DIR / "outputs.json"

EVAL_RESULTS_DIR = Path("eval_results")
DETERMINISTIC_RESULTS_FILE = EVAL_RESULTS_DIR / "deterministic_grader_results.json"
MODEL_RESULTS_FILE = EVAL_RESULTS_DIR / "model_grader_results.json"
COMBINED_RESULTS_FILE = EVAL_RESULTS_DIR / "combined_results.json"

BASELINE_DIR = Path("baseline")
DEFAULT_BASELINE_FILE = BASELINE_DIR / "baseline.json"

COMPARISON_RESULTS_DIR = Path("comparison_results")

_PACKAGED_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def prompts_dir() -> Path:
    """Return the directory prompt templates are loaded from."""
    local = Path("prompts")
    if local.is_dir():
        return local
    return _PACKAGED_PROMPTS_DIR
