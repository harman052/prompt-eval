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

* `run`

## `prompt-eval run`

**Usage**:

```console
$ prompt-eval run [OPTIONS]
```

**Options**:

* `--grader <deterministic|llm-judge|both>`: Runs only the deterministic grader, only the LLM judge, or both side by side. Useful to a reviewer specifically because it lets them see the two grading strategies independently and compare them  [default: both]
* `--help`: Show this message and exit.
