"""map_concurrently: ordering, bounding, and failure semantics."""

from __future__ import annotations

import asyncio

import pytest

from prompt_eval.concurrency import map_concurrently


async def test_returns_results_in_input_order() -> None:
    """Completion order must not leak into the report."""

    async def worker(delay: float) -> float:
        await asyncio.sleep(delay)
        return delay

    items = [0.03, 0.0, 0.02, 0.01]
    assert await map_concurrently(items, worker, limit=4) == items


async def test_respects_the_concurrency_limit() -> None:
    in_flight = 0
    peak = 0

    async def worker(_: int) -> None:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1

    await map_concurrently(list(range(10)), worker, limit=3)
    assert peak == 3


async def test_runs_concurrently_rather_than_sequentially() -> None:
    async def worker(_: int) -> None:
        await asyncio.sleep(0.05)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await map_concurrently(list(range(8)), worker, limit=8)
    assert loop.time() - started < 0.25  # 8 x 50ms sequentially would be 400ms


async def test_empty_input_makes_no_calls() -> None:
    async def worker(_: int) -> int:
        raise AssertionError("worker must not be called")

    assert await map_concurrently([], worker, limit=2) == []


async def test_propagates_the_original_exception_not_a_group() -> None:
    """Callers catch domain errors, so the ExceptionGroup must be unwrapped."""

    async def worker(item: int) -> int:
        if item == 2:
            raise ValueError("boom")
        return item

    with pytest.raises(ValueError, match="boom"):
        await map_concurrently([1, 2, 3], worker, limit=1)


async def test_cancels_siblings_on_failure() -> None:
    completed: list[int] = []

    async def worker(item: int) -> None:
        if item == 0:
            raise ValueError("fail fast")
        await asyncio.sleep(0.5)
        completed.append(item)

    with pytest.raises(ValueError):
        await map_concurrently(list(range(6)), worker, limit=6)
    assert completed == []


@pytest.mark.parametrize("limit", [0, -1])
async def test_rejects_a_non_positive_limit(limit: int) -> None:
    async def worker(item: int) -> int:
        return item

    with pytest.raises(ValueError, match="limit must be"):
        await map_concurrently([1], worker, limit=limit)
