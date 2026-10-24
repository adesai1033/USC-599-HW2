"""Phase 7: session history (memory and Firestore backends), load_history / save_history, follow-ups.

FirestoreHistory runs against a small in-memory fake of the AsyncClient calls it uses.
"""

import asyncio
import itertools

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import AlreadyExists
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

import main
from graph.build import DEFAULT_CONFIG, build_graph
from graph.state import Citation
from state_store import FirestoreHistory, InMemoryHistory, make_history
from tests.sse import parse_sse, serve
from tests.test_graph_skeleton import FakeWorker, decision
from tests.test_robustness import BreakingStream

CITE = Citation(id="db:2406.09756", source="db", arxiv_id="2406.09756", title="MASt3R",
                url="https://arxiv.org/abs/2406.09756")
FIRST_ANSWER = "1. 2DGS [db:2403.17888]\n2. MASt3R [db:2406.09756]"


# ---------------------------------------------------------------- a fake Firestore AsyncClient

class FakeSnapshot:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


class FakeQuery:
    def __init__(self, store, path, field, descending):
        self.store, self.path, self.field, self.descending, self.n = store, path, field, descending, None

    def limit(self, n):
        self.n = n
        return self

    async def stream(self):
        if self.store.stall:
            await asyncio.sleep(10)
        docs = [d for key, d in self.store.docs.items() if key[:-1] == self.path]
        docs.sort(key=lambda d: d[self.field], reverse=self.descending)
        for doc in docs[: self.n]:
            yield FakeSnapshot(doc)


class FakeDocument:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def collection(self, name):
        return FakeCollection(self.store, (*self.path, name))

    async def create(self, data):
        if self.store.steal_next:  # another request claims this turn number first
            self.store.steal_next = False
            self.store.docs[self.path] = data | {"question": "concurrent"}
            raise AlreadyExists("exists")
        if self.path in self.store.docs:
            raise AlreadyExists("exists")
        self.store.docs[self.path] = data

    async def set(self, data, merge=False):
        self.store.docs[self.path] = (self.store.docs.get(self.path, {}) if merge else {}) | data


class FakeCollection:
    def __init__(self, store, path):
        self.store, self.path = store, path

    def document(self, doc_id):
        return FakeDocument(self.store, (*self.path, doc_id))

    def order_by(self, field, direction):
        return FakeQuery(self.store, self.path, field, direction == "DESCENDING")


class FakeFirestore:
    def __init__(self):
        self.docs: dict[tuple, dict] = {}
        self.stall = False
        self.steal_next = False

    def collection(self, name):
        return FakeCollection(self, (name,))


# ---------------------------------------------------------------- the stores

@pytest.fixture(params=["memory", "firestore"])
def store(request):
    return InMemoryHistory() if request.param == "memory" else FirestoreHistory(client=FakeFirestore())


@pytest.mark.asyncio
async def test_turns_load_oldest_first_limited_and_isolated_by_session(store):
    for i in range(1, 5):
        await store.append("s1", f"q{i}", f"a{i}", [CITE] if i == 2 else [])
    await store.append("s2", "other", "other answer", [])
    turns = await store.load("s1", 3)
    assert [(t.n, t.question) for t in turns] == [(2, "q2"), (3, "q3"), (4, "q4")]
    assert turns[0].cited_ids == ["db:2406.09756"] and turns[0].citations[0].title == "MASt3R"
    assert [t.question for t in await store.load("s2", 6)] == ["other"]
    assert await store.load("nobody", 6) == [] and await store.load("s1", 0) == []


@pytest.mark.asyncio
async def test_firestore_layout_and_counter():
    client = FakeFirestore()
    history = FirestoreHistory(client=client)
    await history.append("s1", "q1", "a1", [CITE])
    await history.append("s1", "q2", "a2", [])
    assert ("sessions", "s1", "turns", "000001") in client.docs and ("sessions", "s1", "turns", "000002") in client.docs
    assert client.docs[("sessions", "s1")]["turn_count"] == 2
    assert client.docs[("sessions", "s1", "turns", "000001")]["cited_ids"] == ["db:2406.09756"]


@pytest.mark.asyncio
async def test_firestore_append_never_overwrites_a_concurrent_turn():
    client = FakeFirestore()
    history = FirestoreHistory(client=client)
    await history.append("s1", "q1", "a1", [])
    client.steal_next = True
    turn = await history.append("s1", "q2", "a2", [])
    assert turn.n == 3 and client.docs[("sessions", "s1", "turns", "000002")]["question"] == "concurrent"


@pytest.mark.asyncio
async def test_firestore_calls_are_bounded(monkeypatch):
    monkeypatch.setenv("FIRESTORE_TIMEOUT_S", "0.01")
    client = FakeFirestore()
    client.stall = True
    with pytest.raises(TimeoutError):
        await FirestoreHistory(client=client).load("s1", 6)


def test_backend_is_chosen_by_state_backend(monkeypatch):
    monkeypatch.setenv("STATE_BACKEND", "memory")
    assert isinstance(make_history(), InMemoryHistory)
    monkeypatch.setenv("STATE_BACKEND", "firestore")
    assert isinstance(make_history(), FirestoreHistory)
    monkeypatch.setenv("STATE_BACKEND", "redis")
    with pytest.raises(RuntimeError, match="STATE_BACKEND"):
        make_history()


# ---------------------------------------------------------------- the graph

class RecordingSupervisor:
    """Plays `script` and records the context message of every supervisor call."""

    def __init__(self, script):
        self.script = iter(script)
        self.contexts: list[str] = []

    def with_structured_output(self, schema):
        async def decide(messages):
            self.contexts.append(messages[-1].text)
            return next(self.script)

        return RunnableLambda(decide)


def answers(*texts):
    return GenericFakeChatModel(messages=itertools.chain((AIMessage(t) for t in texts), itertools.repeat(AIMessage("..."))))


async def ask(graph, question, session_id="s1"):
    return await graph.ainvoke({"session_id": session_id, "messages": [HumanMessage(question)], "step": 0},
                               config=DEFAULT_CONFIG)


@pytest.mark.asyncio
async def test_follow_up_sees_the_earlier_turn_and_its_ids():
    history = InMemoryHistory()
    supervisor = RecordingSupervisor([decision("synthesize"), decision("research", "Summarize MASt3R (2406.09756)"),
                                      decision("synthesize")])
    graph = build_graph(workers={"research": FakeWorker()}, supervisor_llm=supervisor,
                        synthesize_llm=answers(FIRST_ANSWER, "MASt3R summary"), history=history)
    await ask(graph, "Most-cited 2024 papers")
    final = await ask(graph, "Summarize the second one")

    assert "Earlier in this conversation" not in supervisor.contexts[0]
    context = supervisor.contexts[1]
    assert "Turn 1. User: Most-cited 2024 papers" in context and "2. MASt3R" in context
    assert "IDs cited in that answer, in order: db:2403.17888, db:2406.09756" in context
    assert context.index("Earlier in this conversation") < context.index("Question: Summarize the second one")
    # history first, then this turn's question and answer
    assert [m.type for m in final["messages"]] == ["human", "ai", "human", "ai"]
    assert [m.text for m in final["messages"][:3]] == ["Most-cited 2024 papers", FIRST_ANSWER, "Summarize the second one"]
    assert [t.question for t in await history.load("s1", 6)] == ["Most-cited 2024 papers", "Summarize the second one"]


@pytest.mark.asyncio
async def test_history_turns_limits_what_is_loaded(monkeypatch):
    monkeypatch.setenv("HISTORY_TURNS", "1")
    history = InMemoryHistory()
    supervisor = RecordingSupervisor(itertools.repeat(decision("synthesize")))
    graph = build_graph(workers={}, supervisor_llm=supervisor, synthesize_llm=answers("a1", "a2", "a3"), history=history)
    for q in ("q1", "q2", "q3"):
        await ask(graph, q)
    assert "Turn 1. User: q2" in supervisor.contexts[-1] and "q1" not in supervisor.contexts[-1]


@pytest.mark.asyncio
async def test_sessions_do_not_share_history():
    supervisor = RecordingSupervisor(itertools.repeat(decision("synthesize")))
    graph = build_graph(workers={}, supervisor_llm=supervisor, synthesize_llm=answers("a1", "a2"), history=InMemoryHistory())
    await ask(graph, "secret question", session_id="alice")
    await ask(graph, "hello", session_id="bob")
    assert "secret question" not in supervisor.contexts[-1]


class BrokenHistory(InMemoryHistory):
    def __init__(self, load=False, append=False):
        super().__init__()
        self.fail_load, self.fail_append = load, append

    async def load(self, session_id, limit):
        if self.fail_load:
            raise ConnectionError("firestore unreachable")
        return await super().load(session_id, limit)

    async def append(self, *args):
        if self.fail_append:
            raise ConnectionError("firestore unreachable")
        return await super().append(*args)


@pytest.mark.asyncio
async def test_unreadable_history_still_answers_and_says_so():
    graph = build_graph(workers={}, supervisor_llm=RecordingSupervisor([decision("synthesize")]),
                        synthesize_llm=answers("answer"), history=BrokenHistory(load=True))
    final = await ask(graph, "What's a point cloud?")
    assert final["messages"][-1].text == "answer"
    assert [e.kind for e in final["errors"]] == ["history_unavailable"]


@pytest.mark.asyncio
async def test_failed_answers_are_not_saved():
    history = InMemoryHistory()
    graph = build_graph(workers={}, supervisor_llm=RecordingSupervisor([decision("synthesize")]),
                        synthesize_llm=BreakingStream(tokens=[]), history=history)
    await ask(graph, "What's a point cloud?")
    assert await history.load("s1", 6) == []


# ---------------------------------------------------------------- the endpoints

def post(client, message, session_id="s1"):
    response = client.post("/chat/stream", json={"session_id": session_id, "message": message})
    assert response.status_code == 200
    return parse_sse(response.text)


def test_endpoint_turns_share_history_and_can_be_restored(monkeypatch):
    supervisor = RecordingSupervisor(itertools.repeat(decision("synthesize")))
    serve(monkeypatch, lambda history: build_graph(workers={}, supervisor_llm=supervisor,
                                                   synthesize_llm=answers(FIRST_ANSWER, "second"), history=history))
    with TestClient(main.app) as client:
        post(client, "Most-cited 2024 papers")
        events = post(client, "Summarize the second one")
        assert "Turn 1. User: Most-cited 2024 papers" in supervisor.contexts[-1]
        assert [d["detail"] for n, d in events if n == "trace" and d["node"] == "load_history" and d["kind"] == "exit"] == ["1 earlier turn"]
        turns = client.get("/sessions/s1/turns").json()
        assert [(t["n"], t["question"]) for t in turns] == [(1, "Most-cited 2024 papers"), (2, "Summarize the second one")]
        assert turns[0]["answer"] == FIRST_ANSWER
        assert client.get("/sessions/someone-else/turns").json() == []
        assert client.get("/sessions/a.b/turns").status_code == 422


def test_unwritable_history_still_ends_with_done(monkeypatch):
    serve(monkeypatch, build_graph(workers={}, supervisor_llm=RecordingSupervisor([decision("synthesize")]),
                                   synthesize_llm=answers("answer"), history=BrokenHistory(append=True)))
    with TestClient(main.app) as client:
        events = post(client, "What's a point cloud?")
    assert events[-1][0] == "done" and "error" not in [n for n, _ in events]
    assert any(d["node"] == "save_history" and d["kind"] == "error" for n, d in events if n == "trace")


def test_turns_endpoint_reports_an_unavailable_store(monkeypatch):
    serve(monkeypatch, build_graph(workers={}, supervisor_llm=RecordingSupervisor([]), synthesize_llm=answers()))
    with TestClient(main.app) as client:
        client.app.state.history = BrokenHistory(load=True)
        response = client.get("/sessions/s1/turns")
    assert response.status_code == 503 and "unreachable" not in response.text
