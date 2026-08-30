from pathlib import Path

# Paths
DATA_DIR = Path("data")
DEFAULT_DATASET_FILE = DATA_DIR / "dataset.json"

BASELINE_DIR = Path("baseline")
DEFAULT_BASELINE_FILE = BASELINE_DIR / "baseline.json"

OUTPUT_DIR = Path("output")
DEFAULT_OUTPUTS_FILE = OUTPUT_DIR / "outputs.json"

COMPARISON_RESULTS_DIR = Path("comparison_results")
DEFAULT_COMPARISON_FILE = COMPARISON_RESULTS_DIR / "comparison.json"

EVAL_RESULTS_DIR = Path("eval_results")
DEFAULT_DETERMINISTIC_RESULTS_FILE = (
    EVAL_RESULTS_DIR / "deterministic_grader_results.json"
)
DEFAULT_COMBINED_RESULTS_FILE = EVAL_RESULTS_DIR / "combined_results.json"
DEFAULT_MODEL_GRADER_RESULTS_FILE = EVAL_RESULTS_DIR / "model_grader_results.json"
DETERMINISTIC_RESULTS_FILE = EVAL_RESULTS_DIR / "deterministic_grader_results.json"
MODEL_RESULTS_FILE = EVAL_RESULTS_DIR / "model_grader_results.json"
COMBINED_RESULTS_FILE = EVAL_RESULTS_DIR / "combined_results.json"

DEFAULT_TEST_CASES = 3
MIN_TEST_CASES = 1
