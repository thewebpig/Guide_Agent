"""Small in-memory conversation store for the single-process local product."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Session:
    messages: deque[dict[str, str]] = field(default_factory=deque)
    current_location: str | None = None


class InMemorySessionStore:
    """Retain recent context without an inactivity expiry.

    Sessions survive for the life of this FastAPI process. History length is
    bounded independently of session duration to avoid unbounded memory and
    model context growth.
    """

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._max_messages = max_turns * 2
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()

    async def context(
        self, session_id: str, current_location: str | None = None
    ) -> tuple[tuple[dict[str, str], ...], str | None]:
        async with self._lock:
            session = self._sessions.setdefault(session_id, _Session())
            if current_location is not None:
                session.current_location = current_location
            return tuple(dict(item) for item in session.messages), session.current_location

    async def append(self, session_id: str, question: str, answer: str) -> None:
        async with self._lock:
            session = self._sessions.setdefault(session_id, _Session())
            session.messages.extend(
                (
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                )
            )
            while len(session.messages) > self._max_messages:
                session.messages.popleft()

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

