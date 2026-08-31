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

* `init-dataset`: Creates a test dataset at...
* `generate`: Generate solution per test case using a LLM
* `evaluate`: Grade existing outputs using one or more...
* `set-baseline`: Explicitly sets current results as a new...
* `compare`: Diffs baseline vs.

## `prompt-eval init-dataset`

Creates a test dataset at data/dataset.json via the LLM

**Usage**:

```console
$ prompt-eval init-dataset [OPTIONS]
```

**Options**:

* `--regenerate`: Forces dataset regeneration even if data/dataset.json already exists. Use --num-cases to specify the number of test cases to generate.
* `--num-cases <int range>`: Number of test cases to generate. Use it with --regenerate flag. Minimum value: 1  [default: 3; x&gt;=1]
* `--help`: Show this message and exit.

## `prompt-eval generate`

Generate solution per test case using a LLM

**Usage**:

```console
$ prompt-eval generate [OPTIONS]
```

**Options**:

* `--dataset <path>`: Path where the test dataset is loaded from.  [default: data/dataset.json]
* `--help`: Show this message and exit.

## `prompt-eval evaluate`

Grade existing outputs using one or more graders.

**Usage**:

```console
$ prompt-eval evaluate [OPTIONS]
```

**Options**:

* `--dataset <path>`: Path where the test dataset is loaded from.  [default: data/dataset.json]
* `--grader <deterministic|llm-judge|both>`: Specify the grader to run  [default: both]
* `--fail-under <float>`: Exit with a non-zero status if the average score across all test cases falls below this value. If unset, no gate is applied.
* `--verbose`: Display detailed LLM-Judge reasoning.
* `--help`: Show this message and exit.

## `prompt-eval set-baseline`

Explicitly sets current results as a new baseline for comparisons

**Usage**:

```console
$ prompt-eval set-baseline [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `prompt-eval compare`

Diffs baseline vs. current

**Usage**:

```console
$ prompt-eval compare [OPTIONS]
```

**Options**:

* `--regression-threshold <float>`: Exit with a non-zero status if any test case&#x27;s score drops by more than this amount compared to the baseline. If unset, regressions are still reported but never cause a non-zero exit.
* `--help`: Show this message and exit.
