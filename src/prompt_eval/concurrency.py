"""Bounded concurrent execution helper."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence


async def map_concurrently[ItemT, ResultT](
    items: Sequence[ItemT],
    worker: Callable[[ItemT], Awaitable[ResultT]],
    *,
    limit: int,
) -> list[ResultT]:
    """Apply ``worker`` to every item, at most ``limit`` at a time.

    Results are returned in input order regardless of completion order, which
    keeps reports deterministic and diffable.

    Failures are *not* swallowed: the first exception propagates and the
    remaining tasks are cancelled. A partially graded run would produce an
    average score that silently understates coverage, which is worse than a
    loud failure for a tool whose whole job is gating CI.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if not items:
        return []

    semaphore = asyncio.Semaphore(limit)

    async def run(item: ItemT) -> ResultT:
        async with semaphore:
            return await worker(item)

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run(item)) for item in items]
    except ExceptionGroup as group_error:
        # A TaskGroup reports failures as an ExceptionGroup. Callers (and the
        # CLI error boundary) only ever act on the first real cause, so unwrap
        # it rather than leaking group semantics through the whole call stack.
        raise _first_leaf(group_error) from group_error

    return [task.result() for task in tasks]


def _first_leaf(error: BaseException) -> BaseException:
    """Return the first non-group exception inside a (possibly nested) group."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error
