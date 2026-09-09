"""Async-safe lazy service cache with deterministic shutdown."""

import asyncio
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

T = TypeVar("T")


class LazyServiceCache(Generic[T]):
    def __init__(self, factory: Callable[[Hashable], T], *, close_owned: bool = True) -> None:
        self._factory = factory
        self._close_owned = close_owned
        self._services: dict[Hashable, T] = {}
        self._tasks: dict[Hashable, asyncio.Task[T]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._closed_ids: set[int] = set()

    async def get(self, key: Hashable) -> T:
        async with self._lock:
            if self._closed:
                raise RuntimeError("service cache is closed")
            service = self._services.get(key)
            if service is not None:
                return service
            task = self._tasks.get(key)
            if task is None:
                task = asyncio.create_task(asyncio.to_thread(self._factory, key))
                self._tasks[key] = task
        try:
            service = await asyncio.shield(task)
        except BaseException:
            async with self._lock:
                if self._tasks.get(key) is task and task.done():
                    self._tasks.pop(key, None)
            raise
        async with self._lock:
            self._tasks.pop(key, None)
            if self._closed:
                await self._close(service)
                raise RuntimeError("service cache is closed")
            self._services[key] = service
            return service

    async def _close(self, service: T) -> None:
        if not self._close_owned or id(service) in self._closed_ids:
            return
        self._closed_ids.add(id(service))
        close = getattr(service, "aclose", None)
        if callable(close):
            await close()

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            tasks = tuple(self._tasks.values())
            services = tuple(self._services.values())
            self._tasks.clear(); self._services.clear()
        if tasks:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            services += tuple(item for item in outcomes if not isinstance(item, BaseException))
        for service in services:
            await self._close(service)
