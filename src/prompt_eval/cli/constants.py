from pathlib import Path

DEFAULT_DATASET_PATH = Path("data/dataset.json")
DEFAULT_BASELINE_PATH = Path("baseline/baseline.json")
DEFAULT_COMBINED_RESULTS_PATH = Path("eval_results/combined_results.json")
DEFAULT_DETERMINISTIC_RESULTS_PATH = Path(
    "eval_results/deterministic_grader_results.json"
)
DEFAULT_MODEL_GRADER_RESULTS_PATH = Path("eval_results/model_grader_results.json")
DEFAULT_COMPARISON_PATH = Path("comparison_results/comparison.json")
DEFAULT_TEST_CASES = 3
MIN_TEST_CASES = 1
