"""Bound concurrent model work and reject an already-full waiting room."""

import asyncio
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from time import monotonic


class CapacityExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class CapacitySnapshot:
    active: int
    waiting: int
    concurrency: int
    queue_size: int


class RequestGate:
    def __init__(self, concurrency: int, queue_size: int) -> None:
        if concurrency < 1 or queue_size < 0:
            raise ValueError("invalid request capacity")
        self._concurrency = concurrency
        self._queue_size = queue_size
        self._semaphore = asyncio.Semaphore(concurrency)
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiting = 0

    async def snapshot(self) -> CapacitySnapshot:
        async with self._lock:
            return CapacitySnapshot(
                self._active,
                self._waiting,
                self._concurrency,
                self._queue_size,
            )

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self._active + self._waiting >= self._concurrency + self._queue_size:
                raise CapacityExceeded("request queue is full")
            self._waiting += 1
        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
                self._active += 1
            yield
        finally:
            async with self._lock:
                if acquired:
                    self._active -= 1
                else:
                    self._waiting -= 1
            if acquired:
                self._semaphore.release()


class ClientRateLimiter:
    """In-process sliding-window limiter for local/single-worker operation."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self._limit = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, client_id: str) -> bool:
        now = monotonic()
        cutoff = now - 60
        async with self._lock:
            events = self._requests[client_id]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self._limit:
                return False
            events.append(now)
            return True
