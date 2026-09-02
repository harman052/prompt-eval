# Prompt Evaluation Pipeline

A lightweight, CI-ready pipeline for evaluating LLM-generated solutions with a
deterministic grader and an LLM-as-judge grader, and for gating pull requests on
the result.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

## Why

Prompt changes are code changes, but they have no compiler and no unit tests. This
tool gives them the same safety net: a versioned dataset, a reproducible score,
a committed baseline, and a non-zero exit code when quality drops.

## How it works

```
init-dataset ──▶ data/dataset.json          test cases, generated once
     │
generate ──────▶ output/outputs.json        one solution per test case
     │
evaluate ──────▶ eval_results/*.json        deterministic + LLM-judge scores
     │
set-baseline ──▶ baseline/baseline.json     the scores to beat
     │
compare ───────▶ comparison_results/*.json  per-test-case deltas vs. baseline
```

**Two graders, one 0-10 scale.**

| Grader        | What it measures                                        | Cost                         |
| ------------- | ------------------------------------------------------- | ---------------------------- |
| Deterministic | Does the solution parse as valid Python / JSON / regex? | free, offline, reproducible  |
| LLM-as-judge  | Does it satisfy the test case's stated criteria?        | one model call per test case |

The final score is their unweighted mean: syntax validity is binary and cheap,
so it anchors a judgement that is graded but subjective.

## Install

```bash
uv sync
cp .env.example .env      # then add your ANTHROPIC_API_KEY
```

## Usage

```bash
prompt-eval init-dataset --num-cases 5      # generate a dataset (once)
prompt-eval generate                        # solve every test case
prompt-eval evaluate --verbose              # grade, with the judge's rationale
prompt-eval set-baseline                    # accept these scores as the bar
prompt-eval evaluate --fail-under 7.0       # gate on absolute quality
prompt-eval compare --regression-threshold 1.0   # gate on relative quality
```

Run `prompt-eval <command> --help` for every option, or see [DOCS.md](DOCS.md).

## Exit codes

CI distinguishes "the tool broke" from "the prompt got worse":

| Code | Meaning                                                          |
| ---- | ---------------------------------------------------------------- |
| `0`  | Success                                                          |
| `1`  | Runtime error (bad config, API failure, corrupt artifact)        |
| `2`  | Usage error (a required input file is missing)                   |
| `3`  | Quality gate failed (`--fail-under` or `--regression-threshold`) |

## Configuration

All settings come from the environment or `.env` (see [.env.example](.env.example)).
Only `ANTHROPIC_API_KEY` is required. `MAX_CONCURRENCY` controls how many model
requests run in parallel; lower it if you hit rate limits.

## Development

```bash
uv run pytest                 # 155 tests, no network access required
uv run pytest --cov=prompt_eval --cov-report=term-missing
uv run ruff check src tests
uv run ruff format src tests
uv run mypy                   # strict mode
```

## Architecture

```
src/prompt_eval/
├── cli/              Typer commands - argument parsing and exit codes only
│   ├── app.py            command registration
│   ├── options.py        shared option types + the error boundary
│   └── commands/         one module per command
├── pipeline.py       orchestration: generate, grade, combine, compare
├── graders/          one scoring strategy per module
├── llm.py            the only place that knows about Anthropic
├── models.py         pydantic models; constrained scores, computed aggregates
├── reporting.py      all Rich rendering
├── storage.py        atomic JSON artifact I/O
├── config.py         pydantic-settings
├── errors.py         one exception hierarchy the CLI maps to exit codes
└── versioning.py     stamps the git prompt revision onto every report
```

The layering rule: `pipeline.py` and everything below it never imports Typer or
Rich, so every stage is testable without a CLI runner and reusable from another
front end.
