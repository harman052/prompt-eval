# Architecture and Design Decisions

This document explains why the code looks the way it does: the problem each
part solves, the decision made, and the trade-offs involved.

---

## 1. Layering: the dependency rule

```
cli/            Typer + Rich. Parses arguments, maps errors to exit codes.
  ↓
pipeline.py     Orchestration. Plain functions in, pydantic models out.
  ↓
graders/  llm.py  storage.py  prompts.py  concurrency.py
  ↓
models.py  config.py  errors.py  constants.py  paths.py
```

**The rule:** nothing at or below `pipeline.py` imports `typer` or `rich`.

**Why it matters:** scoring, file I/O, and CLI presentation are kept in
separate layers, so `pipeline.py` is fully testable without a `CliRunner`, and
the same pipeline can be driven from an HTTP handler or a notebook without
touching a line. Adding a web API later just means importing `pipeline.py`:
the CLI is one of several possible front ends, not the application itself.

---

## 2. Async and concurrency

**Problem:** grading a dataset means many round trips to the LLM API, each
taking a couple of seconds. This is I/O-bound work, which is exactly what
asyncio is for (threads would also work, but async keeps a single-threaded
mental model and matches the SDK's `AsyncAnthropic`).

**Decision:** `AsyncAnthropic` plus a single reusable helper:

```python
async def map_concurrently(items, worker, *, limit) -> list[ResultT]
```

Three properties, each deliberate:

1. **Order-preserving.** Results come back in input order regardless of
   completion order. Reports are therefore deterministic and diffable: a
   `git diff` of two runs shows score changes, not row shuffling.
2. **Bounded.** A `Semaphore` caps in-flight requests at `MAX_CONCURRENCY`.
   Unbounded concurrency over a large dataset would fire hundreds of
   simultaneous requests and earn a `429` immediately. Concurrency is a
   configured resource, not a side effect of dataset size.
3. **Fail-fast.** `asyncio.TaskGroup` cancels siblings on the first failure.
   A run that quietly finishes with, say, 48 of 50 cases graded would report
   an average over the wrong denominator, and this tool's whole job is to
   gate CI on that number. A silently narrowed denominator is worse than a
   loud failure, so the pipeline fails fast with an error naming what broke.

**One subtlety:** `TaskGroup` raises `ExceptionGroup`. Leaking that would force
every caller (and the CLI error boundary) to write `except*`. `map_concurrently`
unwraps the first leaf exception, so callers keep catching ordinary exceptions.

**Testing async code** (`tests/test_concurrency.py`) asserts four things:

- results are returned in input order even when completion order differs;
- the observed peak concurrency equals the limit (instrument the worker);
- total wall-clock is well under the sequential time (proves it is actually
  concurrent, not merely `async`-decorated);
- a failure propagates as the original exception and siblings are cancelled.

---

## 3. Error handling

One exception hierarchy in `errors.py`, rooted at `PromptEvalError`
(`ConfigurationError`, `ArtifactError`, `DatasetError`, `PromptError`,
`LLMError`, `GradingError`). Every failure the pipeline can produce is
expressed as a `PromptEvalError` subclass, which keeps the CLI boundary
trivial: it catches one base class and maps it to an exit code, instead of
handling each possible exception type individually.

Three rules:

1. **Translate at the boundary.** `llm.py` is the only module that knows
   `anthropic` exists. It maps SDK exceptions to `LLMError` most-specific-first:
   a `401` and a `503` are both "the API call failed," but only one is worth
   retrying, and the user needs to be told which. Every message names the fix
   ("Check ANTHROPIC_API_KEY", "Lower MAX_CONCURRENCY").
2. **Never catch to continue.** Errors propagate to one place: the
   `@handle_errors` decorator in `cli/options.py`. Commands are straight-line
   happy paths; the boundary owns formatting and the exit code. Reusing one
   `Annotated` alias per option also keeps help text identical across
   commands, so `--dataset` cannot document itself differently in two places.
3. **Never catch bare** `Exception`**.** A boundary that only catches
   `PromptEvalError` cannot accidentally relabel a bug in our own code as a
   user-facing error.

**Retries.** `max_retries` is set on the SDK client, which implements
exponential backoff for `408/409/429/5xx` and connection errors, rather than
hand-rolling a backoff loop.

---

## 4. Pydantic modelling

**Generics for shared shape.** One envelope, several aliases:

```python
class Report[ResultT: IdentifiedResult](BaseModel):
    metadata: RunMetadata
    results: list[ResultT]

CombinedReport = Report[CombinedResult]   # + others
```

The generic bound is `IdentifiedResult` (a model with a `test_case_id`), not
`BaseModel`. That is what makes `Report.by_test_case_id` type-safe instead of a
`getattr` with an ignore comment: carrying an id is what makes a row a row.
A useful rule of thumb: if an operation only transforms, accesses, validates,
or derives information from the model's own data, it belongs on the model.
`by_test_case_id` fits that rule well.

**Inheritance for genuinely shared shape.** `TestCase(TestCaseSpec)` adds only
the id. `TestCaseRef` holds the three identifying columns every result row
echoes.

**Constrained types make bad states unrepresentable.**

```python
Score = Annotated[float, Field(ge=MIN_SCORE, le=MAX_SCORE)]
```

An out-of-range score cannot be constructed or loaded from disk. A judge that
returns `50` fails validation at the boundary instead of poisoning an average.

**Computed fields instead of stored aggregates.** `final_score` and `delta` are
`@computed_field` properties. They are serialised into the JSON (so consumers
still see them) but derived on read, so a persisted artifact can never disagree
with itself and no caller can forget to recompute one.

One gotcha worth documenting in the code itself: a computed field is written
but not accepted as input, so `extra="forbid"` breaks the save/load round trip
on exactly those models. `CombinedResult` and `ComparisonResult` therefore use
`extra="ignore"` with a comment explaining why, while models authored from
scratch keep `extra="forbid"` to catch typos.

**Frozen, strict, and self-documenting.** `frozen=True` (results are facts, not
mutable state), `extra="forbid"` by default, and `SolutionFormat`/`GraderChoice`
as `StrEnum` so an unknown format fails validation instead of raising `KeyError`
deep inside a validator lookup. `Field(description=...)` on `ModelGrade` is not
decoration: those descriptions become the JSON schema the judge is constrained
by, so documenting the 0-10 scale there is prompt engineering in the type
system.

---

## 5. Do we need this class?

The heuristic applied throughout the codebase: a class earns its keep when it
owns state with a lifecycle, or when polymorphism is needed. Grouping related
functions is what a module is for.

`LLMClient` is a class because it owns real state: an HTTP connection pool,
timeouts, retry config, and an `aclose` lifecycle. Grading logic, by contrast,
is plain functions (`grade_with_judge(llm, test_case, solution)`,
`grade_syntax(...)`) that take a client and data in and return a result: there's
no state to own, so a class would just be a function wearing a costume.

---

## 6. File layout

Data, output, and result directories are resolved relative to the current
working directory, since they hold user data and are expected to live next to
the repository being evaluated. Prompt templates are versioned assets, so they
are looked up in the working directory first and fall back to the directory
that ships with the installed package. This lets a project override a prompt
without forking the tool, while still working out of the box after a plain
`pip install`.

Configuration (`config.py`) is kept separate from constants (`constants.py`):
`config.py` holds environment-sensitive settings such as API keys, resolved
once via a cached, lazy `get_settings()` so that a missing key surfaces as a
clean CLI error rather than an import-time traceback. `constants.py` holds
path defaults and other values that don't depend on the environment.

---

## 7. Testing strategy

155 tests, 98% coverage, no network access, ~2 seconds.

**Fakes over mocks.** The LLM boundary is one class with two methods, so
`FakeLLMClient` is ~30 lines and reads like documentation. It also records every
prompt it receives, which is how prompt rendering gets tested: a `Mock` would
assert on call arguments and prove nothing about the rendered text.

**Two autouse fixtures enforce two rules for every test.** One `chdir`s into
`tmp_path`, since the pipeline resolves artifacts relative to the CWD and
tests must not write into the developer's tree or read each other's output.
One sets deterministic env vars, so a local `.env` cannot change a test
result. Together: no test may touch the network, and no test may touch the
developer's working tree.

**What is actually asserted**, beyond happy paths:

- **Exit codes**, explicitly: they are the contract with CI (0 success, 1
  error, 2 usage, 3 quality gate failed).
- **Boundary conditions:** `--fail-under 0`, a score exactly at the threshold,
  a delta exactly at the regression threshold, an empty score list.
- **Negative space:** no artifact is written when a stage fails; the
  deterministic-only path makes zero model calls; a corrupt results file
  cannot become a baseline.
- **Enum exhaustiveness:** `set(SYNTAX_VALIDATORS) == set(SolutionFormat)`, so
  adding a format without a validator fails the suite instead of production.
- **Schema round-tripping**, which is what catches the computed-field trap
  described in §4.

`filterwarnings = ["error"]` is on, so an unawaited coroutine or other async
warning fails the suite instead of passing silently.

---

## 8. Scalability

| Dimension  | How it scales                                                                                                                                       |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Test cases | Bounded concurrency, so 10 or 10,000 cases use the same number of connections. Wall-clock grows linearly at `n/MAX_CONCURRENCY`.                    |
| Graders    | Adding one means: a row model, a `score_*` function, an enum member. The `Report[T]` envelope, storage, joins and rendering are already generic.    |
| Front ends | `pipeline.py` has no CLI dependency.                                                                                                                |
| Providers  | `llm.py` is the only Anthropic-aware module, and it exposes two verbs (`complete`, `parse`).                                                        |
| Cost       | The free grader runs first; `--grader deterministic` is a zero-cost smoke test; CI skips the paid job unless `prompts/`, `data/` or `src/` changed. |

**Honest limits (v1).** Reports are single JSON files read fully into memory,
which is fine for thousands of cases but wrong for millions. Scores are
unweighted means. A failed run restarts from scratch rather than resuming.
`set-baseline` keeps one baseline, not a history. All are deliberate v1 scope,
not oversights.

---

## 9. CI

- **CI is two jobs.** Lint/type/test is fast and free and runs on everything;
  the paid evaluation job `needs:` it and is path-filtered. Never spend API
  credits to discover a formatting error.
