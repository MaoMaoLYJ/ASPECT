"""Work-conserving request admission grouped by LoRA adapter route."""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass
class _PendingRequest(Generic[T]):
    invoke: Callable[[Hashable], Awaitable[T]]
    future: asyncio.Future[T]


class RouteHomogeneousBatcher(Generic[T]):
    """Admit work-conserving FIFO chunks grouped by route.

    Calls remain independent requests. Admission is FIFO within each route and
    rotates between routes after each homogeneous chunk. A slow request never
    blocks free global slots from admitting the next route.
    """

    def __init__(
        self,
        *,
        max_batch_size: int,
        max_requests_per_turn: int | None = None,
        coalesce_ms: float = 2.0,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms cannot be negative")
        self.max_batch_size = int(max_batch_size)
        self.max_requests_per_turn = int(
            max_requests_per_turn or self.max_batch_size
        )
        if self.max_requests_per_turn < self.max_batch_size:
            raise ValueError(
                "max_requests_per_turn cannot be smaller than max_batch_size"
            )
        self.coalesce_seconds = float(coalesce_ms) / 1000.0
        self._queues: OrderedDict[Hashable, deque[_PendingRequest[T]]] = OrderedDict()
        self._dispatcher: asyncio.Task[None] | None = None
        self._new_request = asyncio.Event()
        self.dispatched_batches = 0
        self.dispatched_requests = 0

    async def submit(
        self,
        route_key: Hashable,
        invoke: Callable[[Hashable], Awaitable[T]],
    ) -> T:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        queue = self._queues.setdefault(route_key, deque())
        queue.append(_PendingRequest(invoke=invoke, future=future))
        self._new_request.set()
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch())
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _dispatch(self) -> None:
        try:
            if self.coalesce_seconds:
                await asyncio.sleep(self.coalesce_seconds)
            active: dict[
                asyncio.Task[T], tuple[Hashable, _PendingRequest[T]]
            ] = {}
            active_by_route: Counter[Hashable] = Counter()
            while self._queues or active:
                self._new_request.clear()
                while self._queues and len(active) < self.max_requests_per_turn:
                    made_progress = False
                    routes_to_visit = len(self._queues)
                    for _ in range(routes_to_visit):
                        if len(active) >= self.max_requests_per_turn:
                            break
                        route_key = next(iter(self._queues))
                        queue = self._queues[route_key]
                        route_capacity = (
                            self.max_batch_size - active_by_route[route_key]
                        )
                        chunk_capacity = min(
                            route_capacity,
                            self.max_requests_per_turn - len(active),
                        )
                        started = 0
                        while queue and started < chunk_capacity:
                            request = queue.popleft()
                            if request.future.cancelled():
                                continue
                            task = asyncio.create_task(request.invoke(route_key))
                            active[task] = (route_key, request)
                            active_by_route[route_key] += 1
                            started += 1
                            self.dispatched_requests += 1
                        if started:
                            made_progress = True
                            self.dispatched_batches += 1

                        if queue:
                            self._queues.move_to_end(route_key)
                        else:
                            del self._queues[route_key]
                    if not made_progress:
                        break

                if not active:
                    continue

                # A later workflow stage can arrive while every active request
                # is still decoding. Wake on arrival, not only on completion.
                arrival = asyncio.create_task(self._new_request.wait())
                try:
                    done, _ = await asyncio.wait(
                        [*active, arrival],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    arrival.cancel()
                    await asyncio.gather(arrival, return_exceptions=True)
                done.discard(arrival)
                for task in done:
                    route_key, request = active.pop(task)
                    active_by_route[route_key] -= 1
                    if active_by_route[route_key] == 0:
                        del active_by_route[route_key]
                    if request.future.cancelled():
                        continue
                    try:
                        request.future.set_result(task.result())
                    except BaseException as exc:
                        request.future.set_exception(exc)
        finally:
            # A submit can race with the dispatcher's final empty-queue check.
            # Hand any such request to a fresh dispatcher before this task exits.
            self._dispatcher = None
            if self._queues:
                self._dispatcher = asyncio.create_task(self._dispatch())
