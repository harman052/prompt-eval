# `prompt-eval`

Prompt Evaluation CLI-Tool

GitHub: https://github.com/harman052/prompt-eval

**Usage**:

```console
$ prompt-eval [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `init-dataset`: Create a test dataset at data/dataset.json via the LLM.
* `generate`: Generate a solution per test case using the LLM.
* `evaluate`: Grade existing solutions using one or more graders.
* `set-baseline`: Set the current combined results as the new comparison baseline.
* `compare`: Diff baseline scores against the current run.

**Exit codes** (shared by every command):

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Runtime error (bad configuration, API failure, corrupt artifact) |
| `2` | Usage error (a required input file is missing) |
| `3` | Quality gate failed (`--fail-under` or `--regression-threshold`) |

## `prompt-eval init-dataset`

Create a test dataset at `data/dataset.json` via the LLM. Refuses to overwrite
an existing dataset unless `--regenerate` is passed, because regenerating
invalidates the committed baseline.

**Usage**:

```console
$ prompt-eval init-dataset [OPTIONS]
```

**Options**:

* `--num-cases <int range>`: Number of test cases to generate. Minimum: 1  [default: 3; x&gt;=1]
* `--regenerate`: Overwrite an existing dataset instead of refusing to run.
* `--help`: Show this message and exit.

## `prompt-eval generate`

Generate a solution per test case using the LLM. Requests run concurrently, up
to `MAX_CONCURRENCY` at a time. Writes `output/outputs.json`.

**Usage**:

```console
$ prompt-eval generate [OPTIONS]
```

**Options**:

* `--dataset <path>`: Path the test dataset is loaded from.  [default: data/dataset.json]
* `--help`: Show this message and exit.

## `prompt-eval evaluate`

Grade existing solutions using one or more graders. Writes
`eval_results/deterministic_grader_results.json`,
`eval_results/model_grader_results.json` and, for `--grader both`,
`eval_results/combined_results.json`.

**Usage**:

```console
$ prompt-eval evaluate [OPTIONS]
```

**Options**:

* `--dataset <path>`: Path the test dataset is loaded from.  [default: data/dataset.json]
* `--grader <deterministic|llm-judge|both>`: Which grader(s) to run: syntax only, judge only, or both.  [default: both]
* `--fail-under <float range>`: Exit non-zero if the average score across all test cases falls below this value. If unset, no gate is applied.  [x&gt;=0.0]
* `--verbose`: Include the LLM-Judge rationale in the output.
* `--help`: Show this message and exit.

Results are written before the gate is applied, so a failing CI run still
uploads its artifacts.

## `prompt-eval set-baseline`

Set the current combined results as the new comparison baseline. The results
file is validated before it is promoted, so a corrupt artifact cannot become the
bar that every future run is judged against.

**Usage**:

```console
$ prompt-eval set-baseline [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `prompt-eval compare`

Diff baseline scores against the current run. Test cases are matched by
`test_case_id`, not by position, so reordering the dataset cannot manufacture a
regression. Test cases present on only one side are reported and skipped.

**Usage**:

```console
$ prompt-eval compare [OPTIONS]
```

**Options**:

* `--regression-threshold <float range>`: Exit non-zero if any test case's score drops by more than this amount versus the baseline. If unset, regressions are reported but never fail the run.  [x&gt;=0.0]
* `--help`: Show this message and exit.
