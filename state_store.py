"""Session history: FirestoreHistory (prod) / InMemoryHistory (local, tests).

A turn is the user's message, the final answer, and the citations validated on that answer. Traces are
not stored. Firestore layout: sessions/{session_id} (updated_at, turn_count) and
sessions/{session_id}/turns/{n:06d}; the zero-padded ID keeps document order equal to turn order.
"""

import asyncio
import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from pydantic import BaseModel, Field

from config import firestore_project_id, firestore_timeout_s, state_backend
from graph.state import Citation

logger = logging.getLogger("state_store")

T = TypeVar("T")

MAX_ANSWER_CHARS = 20_000  # far under Firestore's 1 MiB document limit
APPEND_ATTEMPTS = 3        # two requests of one session racing for the same turn number


class Turn(BaseModel):
    n: int
    question: str
    answer: str
    cited_ids: list[str] = []
    citations: list[Citation] = []  # the answer's valid citations, so a restored transcript can link them
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class History(Protocol):
    async def load(self, session_id: str, limit: int) -> list[Turn]:
        """The session's last `limit` turns, oldest first."""
        ...

    async def append(self, session_id: str, question: str, answer: str, citations: list[Citation]) -> Turn:
        """Store one turn as the session's next turn and return it."""
        ...


def _turn(n: int, question: str, answer: str, citations: list[Citation]) -> Turn:
    return Turn(n=n, question=question, answer=answer[:MAX_ANSWER_CHARS], cited_ids=[c.id for c in citations],
                citations=citations)


class InMemoryHistory:
    """History in this process's memory, for local runs and tests. Not durable: it is lost when the
    process exits, and Cloud Run instances don't share it."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[Turn]] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str, limit: int) -> list[Turn]:
        if limit <= 0:
            return []
        async with self._lock:
            return list(self._sessions.get(session_id, [])[-limit:])

    async def append(self, session_id: str, question: str, answer: str, citations: list[Citation]) -> Turn:
        async with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turn = _turn(len(turns) + 1, question, answer, citations)
            turns.append(turn)
            return turn


class FirestoreHistory:
    """History in Firestore (Native mode), shared by every Cloud Run instance and revision.

    The client is created on first use from Application Default Credentials (the runtime service
    account on Cloud Run, the gcloud login locally), so a credentials problem shows up as a failed
    history call, not a failed startup. Every call is bounded by FIRESTORE_TIMEOUT_S.
    """

    def __init__(self, client=None) -> None:
        self._client = client

    def _sessions(self):
        if self._client is None:
            from google.cloud import firestore  # imported here so memory-backend runs don't need it

            self._client = firestore.AsyncClient(project=firestore_project_id())
        return self._client.collection("sessions")

    async def _bounded(self, call: Awaitable[T]) -> T:
        return await asyncio.wait_for(call, firestore_timeout_s())

    async def _latest(self, turns, limit: int) -> list[Turn]:
        query = turns.order_by("n", direction="DESCENDING").limit(limit)
        return [Turn.model_validate(doc.to_dict()) async for doc in query.stream()]

    async def load(self, session_id: str, limit: int) -> list[Turn]:
        if limit <= 0:
            return []
        turns = self._sessions().document(session_id).collection("turns")
        return list(reversed(await self._bounded(self._latest(turns, limit))))

    async def append(self, session_id: str, question: str, answer: str, citations: list[Citation]) -> Turn:
        return await self._bounded(self._append(session_id, question, answer, citations))

    async def _append(self, session_id: str, question: str, answer: str, citations: list[Citation]) -> Turn:
        from google.api_core.exceptions import AlreadyExists

        session = self._sessions().document(session_id)
        turns = session.collection("turns")
        for _ in range(APPEND_ATTEMPTS):
            latest = await self._latest(turns, 1)
            turn = _turn(latest[0].n + 1 if latest else 1, question, answer, citations)
            try:
                # create() fails if the document exists, so a concurrent turn can't be overwritten.
                await turns.document(f"{turn.n:06d}").create(turn.model_dump(mode="json") | {"created_at": turn.created_at})
            except AlreadyExists:
                continue
            await session.set({"updated_at": turn.created_at, "turn_count": turn.n}, merge=True)
            return turn
        raise RuntimeError(f"could not claim a turn number after {APPEND_ATTEMPTS} attempts")


def make_history() -> History:
    """The history backend named by STATE_BACKEND."""
    if state_backend() == "firestore":
        logger.info("history backend: Firestore (project %s)", firestore_project_id() or "from ADC")
        return FirestoreHistory()
    logger.info("history backend: in-memory (not durable)")
    return InMemoryHistory()
