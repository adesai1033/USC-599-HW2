# Agent orchestration flow

How one question moves through the system, from the browser to the streamed answer. The graph is
built in `graph/build.py`; the full design is in [`architecture.md`](architecture.md).

## End to end

```mermaid
flowchart TD
    U(["Browser: POST /chat/stream<br/>{session_id, message}"]) --> API["FastAPI (main.py)<br/>opens the SSE stream"]
    API --> LH["load_history<br/>last 6 turns from Firestore"]
    LH --> SUP{"supervisor<br/>Gemini Flash, structured output<br/>→ next, instruction, task_type, reason"}

    SUP -- "next = research" --> R["research worker"]
    SUP -- "next = data" --> D["data worker"]
    SUP -- "next = summary" --> S["summary worker"]
    SUP -- "next = synthesize<br/>(enough findings, no tools needed,<br/>step limit, or every worker failed)" --> SYN

    R -- "Finding or AgentError" --> SUP
    D -- "Finding or AgentError" --> SUP
    S -- "Finding or AgentError" --> SUP

    subgraph MCP ["MCP servers (MultiServerMCPClient)"]
        CORPUS[("corpus · stdio<br/>search_corpus, get_chunks<br/>Chroma, 4,741 chunks")]
        DB[("papers_db · stdio<br/>describe_schema, run_sql,<br/>get_paper_metadata · SQLite")]
        TAV[("tavily · streamable HTTP<br/>tavily_search, tavily_extract")]
    end

    R -. "search_corpus, get_paper_metadata, tavily_search" .-> MCP
    D -. "describe_schema, run_sql, get_paper_metadata" .-> MCP
    S -. "search_corpus, get_chunks, get_paper_metadata, tavily_extract" .-> MCP

    SYN["synthesize<br/>answer from findings only<br/>streams tokens, validates citations"] --> SH["save_history<br/>write the turn to Firestore"]
    SH --> DONE(["SSE: citations, then done"])
```

Every step also streams to the browser as it happens: a `routing` event for each model call, and
`trace` events for each node entry and exit, supervisor decision and tool call.

## Inside a worker

Each worker is its own compiled subgraph, with only the tools in its allowlist
(`agents/registry.py`) and its own prompt (`prompts/<agent>.md`):

```mermaid
flowchart LR
    IN(["instruction from the supervisor"]) --> RT["router.route()<br/>pick fast or strong tier"]
    RT --> A["worker_agent<br/>model call"]
    A -- "tool_calls" --> T["ToolNode<br/>allowlisted MCP tools"]
    T -- "tool results" --> A
    A -- "no more tool calls,<br/>or round cap reached" --> V{"citations valid?<br/>attribution right?"}
    V -- "no: ask once to fix" --> A
    V -- "yes, or still invalid<br/>after one fix → [unverified]" --> OUT(["Finding → supervisor"])
    G["guarded(): 60 s timeout,<br/>exception → AgentError"] -.- A
```

| Worker | Default tier | Escalates to Claude Sonnet when | Tools |
|---|---|---|---|
| `research` | Gemini Flash | the task is a compare or a lit review | `search_corpus`, `get_paper_metadata`, `tavily_search` |
| `data` | Gemini Flash | the first SQL attempt this turn errored | `describe_schema`, `run_sql`, `get_paper_metadata` |
| `summary` | Claude Sonnet | (drops to Gemini when one paper is in scope) | `search_corpus`, `get_chunks`, `get_paper_metadata`, `tavily_extract` |
| `synthesize` | Gemini Flash | 2+ findings, 3+ papers cited, or a large context | none |

On a provider error, any call retries once on the other provider (Gemini ↔ Claude).

## An example turn

"Most-cited 2024 papers and what each proposes" needs two workers in turn:

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant SUP as supervisor
    participant D as data
    participant S as summary
    participant M as MCP servers
    participant SYN as synthesize
    participant F as Firestore

    B->>SUP: question (after load_history reads earlier turns from F)
    SUP->>D: next=data: "find the most-cited 2024 papers"
    D->>M: describe_schema, then run_sql
    M-->>D: rows (arXiv IDs, citation counts)
    D-->>SUP: Finding with [db:…] citations
    SUP->>S: next=summary: "summarize what each of these IDs proposes"
    S->>M: search_corpus per paper, get_paper_metadata
    M-->>S: chunks
    S-->>SUP: Finding with chunk citations
    SUP->>SYN: next=synthesize
    SYN-->>B: token events, then citations
    SYN->>F: save_history writes the turn
    SYN-->>B: done (timings, token usage)
```

## When something fails

| Failure | What happens | The user sees |
|---|---|---|
| A worker throws or passes 60 s | `guarded()` records an `AgentError`; the supervisor retries once with a new instruction, tries another worker, or synthesizes | a partial answer, still ending in `done` |
| A provider errors (timeout, 429, 5xx, bad key) | the call retries once on the other provider | `fallback=True` on the routing badge |
| An MCP server is down | its tools return a clear error, and the agents' prompts say the capability is unavailable | "corpus search is unavailable" in the answer |
| The supervisor's call fails | go straight to `synthesize` with what exists | an answer from the findings so far |
| `synthesize` fails | a fixed notice ends the answer | `error`, then `done` |
| Too many steps | `MAX_SUPERVISOR_STEPS` (6) forces `synthesize` | an answer from the findings so far |
