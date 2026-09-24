"""Per-engine full-weight routing: drain requests before changing any weights."""

import asyncio
from collections import OrderedDict, deque


class FullParameterBatcher:
    def __init__(self, activate, max_batch_size=128):
        if max_batch_size <= 0:
            raise ValueError('Positive batch size required')
        self.activate = activate
        self.max_batch_size = max_batch_size
        self.queues = OrderedDict()
        self.dispatcher = None

    async def submit(self, route, invoke):
        future = asyncio.get_running_loop().create_future()
        self.queues.setdefault(route, deque()).append((invoke, future))
        if self.dispatcher is None:
            self.dispatcher = asyncio.create_task(self.dispatch())
        return await future

    async def dispatch(self):
        try:
            await asyncio.sleep(0.002)
            while self.queues:
                route, pending = self.queues.popitem(last=False)
                chunk = []
                while pending and len(chunk) < self.max_batch_size:
                    invoke, future = pending.popleft()
                    if not future.cancelled():
                        chunk.append((invoke, future))
                if pending:
                    self.queues[route] = pending
                if not chunk:
                    continue
                try:
                    await self.activate(route)
                    results = await asyncio.gather(*(invoke(route) for invoke, _ in chunk), return_exceptions=True)
                except Exception as exc:
                    results = [exc] * len(chunk)
                for (_, future), result in zip(chunk, results, strict=True):
                    if not future.done():
                        if isinstance(result, BaseException):
                            future.set_exception(result)
                        else:
                            future.set_result(result)
        finally:
            self.dispatcher = None
            if self.queues:
                self.dispatcher = asyncio.create_task(self.dispatch())
