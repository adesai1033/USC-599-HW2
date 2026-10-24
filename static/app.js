"use strict";

// One session per tab, kept across reloads. crypto.randomUUID exists only on HTTPS and localhost.
function newSessionId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

function loadSessionId() {
  try {
    const id = sessionStorage.getItem("session_id") || newSessionId();
    sessionStorage.setItem("session_id", id);
    return id;
  } catch {
    return newSessionId();
  }
}

const sessionId = loadSessionId();

const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendButton = document.getElementById("send");

let streaming = false;

// Failures guard() records for a worker (graph/guard.py); other worker errors don't stop the worker.
const WORKER_FAILURES = new Set(["worker_timeout", "transport", "provider_error", "worker_exception"]);
const WORKERS = new Set(["research", "data", "summary"]);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function addUserMessage(text) {
  const message = el("div", "message user");
  message.appendChild(el("div", "body", text));
  messagesEl.appendChild(message);
  message.scrollIntoView({ block: "end" });
}

function formatMs(ms) {
  return ms == null ? "—" : `${(ms / 1000).toFixed(2)} s`;
}

function formatTokens(n) {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

// ---------------------------------------------------------------- Markdown and citations

// Mirrors rag/citations.py: a bracket holds one or more IDs separated by commas or semicolons.
const CITE_ID = /^(?:\d{4}\.\d{4,5}#[a-z0-9]+(?:-[a-z0-9]+)*-\d{2,}|web:[0-9a-f]{8}|db:\d{4}\.\d{4,5}|unverified)$/;
const BRACKET = /\[([^\[\]\n]{1,400})\]/g;

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function citationLabel(id) {
  if (id.startsWith("web:")) return "web";
  if (id.startsWith("db:")) return `db · ${id.slice(3)}`;
  const [arxivId, chunk] = id.split("#");
  const section = chunk.replace(/-\d+$/, "");
  return `${arxivId} · ${section.length > 18 ? `${section.slice(0, 16)}…` : section}`;
}

function citationUrl(citation) {
  const url = citation.url || (citation.arxiv_id ? `https://arxiv.org/abs/${citation.arxiv_id}` : "");
  return /^https?:\/\//.test(url) ? url : "";
}

function citationTitle(citation) {
  const where = [citation.section && `§${citation.section}`, citation.page != null && `p.${citation.page}`].filter(Boolean);
  const source = { corpus: "paper corpus", web: "web", db: "paper database" }[citation.source];
  return [citation.title || citation.id, where.join(", "), source].filter(Boolean).join(" — ");
}

// `cites` is null while the answer streams; after the citations event it holds the validated report.
function chipHtml(id, cites) {
  if (id === "unverified") {
    return `<span class="cite unverified" title="This claim's source was never retrieved">unverified</span>`;
  }
  const label = escapeHtml(citationLabel(id));
  if (!cites) return `<span class="cite pending" title="${escapeHtml(id)}">${label}</span>`;
  const citation = cites.valid.get(id);
  if (!citation) {
    return `<span class="cite unverified" title="${escapeHtml(id)}: not returned by any tool this turn">${label}</span>`;
  }
  const url = citationUrl(citation);
  const attrs = `class="cite source-${escapeHtml(citation.source)}" title="${escapeHtml(citationTitle(citation))}"`;
  // A web chip names its site once the URL is known.
  const shown = citation.source === "web" && url ? escapeHtml(new URL(url).hostname.replace(/^www\./, "")) : label;
  return url
    ? `<a ${attrs} href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${shown}</a>`
    : `<span ${attrs}>${shown}</span>`;
}

function emphasis(html) {
  return html
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
}

function renderCitations(text, cites) {
  let html = "";
  let last = 0;
  for (const match of text.matchAll(BRACKET)) {
    const ids = match[1].split(/[,;]/).map((t) => t.trim());
    if (!ids.every((t) => CITE_ID.test(t))) continue;
    html += emphasis(escapeHtml(text.slice(last, match.index)));
    html += ids.map((id) => chipHtml(id, cites)).join("");
    last = match.index + match[0].length;
  }
  return html + emphasis(escapeHtml(text.slice(last)));
}

function renderInline(text, cites) {
  return text
    .split(/(`[^`\n]+`)/)
    .map((part) => (part.length > 2 && part.startsWith("`") && part.endsWith("`")
      ? `<code>${escapeHtml(part.slice(1, -1))}</code>`
      : renderCitations(part, cites)))
    .join("");
}

const LIST_ITEM = /^\s*(?:[-*+]|\d+[.)])\s+/;
const TABLE_ROW = /^\s*\|.*\|\s*$/;
const TABLE_RULE = /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$/;

function tableCells(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

function startsBlock(lines, i) {
  const line = lines[i];
  return /^```/.test(line) || /^#{1,6}\s/.test(line) || LIST_ITEM.test(line) || /^\s*(?:---|\*\*\*)\s*$/.test(line)
    || (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1]));
}

// A small Markdown subset: headings, paragraphs, lists, tables, code. Every piece of text is escaped
// before any markup is added, so model output can't inject HTML.
function renderMarkdown(source, cites) {
  const lines = source.split("\n");
  const html = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const code = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) code.push(lines[i]);
      i++;
      html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
    } else if (/^\s*$/.test(line)) {
      i++;
    } else if (/^\s*(?:---|\*\*\*)\s*$/.test(line)) {
      html.push("<hr>");
      i++;
    } else if (/^#{1,6}\s/.test(line)) {
      const [, hashes, text] = /^(#{1,6})\s+(.*)$/.exec(line);
      const level = Math.min(hashes.length + 2, 6);
      html.push(`<h${level}>${renderInline(text, cites)}</h${level}>`);
      i++;
    } else if (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
      const head = tableCells(line).map((c) => `<th>${renderInline(c, cites)}</th>`).join("");
      const rows = [];
      for (i += 2; i < lines.length && TABLE_ROW.test(lines[i]); i++) {
        rows.push(`<tr>${tableCells(lines[i]).map((c) => `<td>${renderInline(c, cites)}</td>`).join("")}</tr>`);
      }
      html.push(`<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`);
    } else if (LIST_ITEM.test(line)) {
      const ordered = /^\s*\d/.test(line);
      const items = [];
      while (i < lines.length && (LIST_ITEM.test(lines[i]) || (/^\s+\S/.test(lines[i]) && items.length))) {
        if (LIST_ITEM.test(lines[i])) items.push(lines[i].replace(LIST_ITEM, ""));
        else items[items.length - 1] += ` ${lines[i].trim()}`;
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      html.push(`<${tag}>${items.map((item) => `<li>${renderInline(item, cites)}</li>`).join("")}</${tag}>`);
    } else {
      const paragraph = [];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !(paragraph.length && startsBlock(lines, i))) {
        paragraph.push(lines[i]);
        i++;
      }
      html.push(`<p>${paragraph.map((p) => renderInline(p, cites)).join("<br>")}</p>`);
    }
  }
  return html.join("");
}

// ---------------------------------------------------------------- one assistant turn

class Turn {
  constructor() {
    this.root = el("div", "message assistant");

    // Routing badge: what ran (from the supervisor's decisions) and which model each call used.
    this.route = el("div", "route");
    this.path = el("div", "route-path");
    this.path.appendChild(el("span", "route-label", "Route"));
    this.models = el("div", "route-models");
    this.why = el("details", "route-why");
    this.why.appendChild(el("summary", null, "Why this route"));
    this.whyList = el("ul");
    this.why.appendChild(this.whyList);
    this.route.append(this.path, this.models, this.why);

    this.body = el("div", "body markdown");
    this.sources = el("div", "sources");
    this.sources.hidden = true;

    this.trace = el("details", "trace");
    this.traceSummary = el("summary", null, "Trace · 0 events");
    this.traceRows = el("ol", "trace-rows");
    this.trace.append(this.traceSummary, this.traceRows);

    this.meta = el("div", "meta");
    this.root.append(this.route, this.body, this.sources, this.trace, this.meta);
    messagesEl.appendChild(this.root);
    this.root.scrollIntoView({ block: "end" });

    this.text = "";
    this.cites = null;
    this.events = 0;
    this.pending = null;     // the supervisor's latest decision, until its worker enters
    this.override = null;    // a supervisor error that replaced its decision with synthesize
    this.chips = new Map();  // agent -> its latest path chip
    this.renderQueued = false;
  }

  render() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      // Follow the answer only while the reader is at the bottom, not while they read the trace above.
      const following = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 120;
      this.body.innerHTML = renderMarkdown(this.text, this.cites);
      if (following) this.root.scrollIntoView({ block: "end" });
    });
  }

  addWhy(label, text) {
    const item = el("li");
    item.append(el("strong", null, label), document.createTextNode(` ${text}`));
    this.whyList.appendChild(item);
  }

  addPathChip(node) {
    if (this.path.children.length > 1) this.path.appendChild(el("span", "route-arrow", "→"));
    const decision = this.pending && this.pending.next === node ? this.pending : null;
    const chip = el("span", `route-chip node-${node}`, node);
    if (decision && decision.task_type && node !== "synthesize") {
      chip.appendChild(el("span", "route-task", decision.task_type));
    }
    if (decision) chip.title = decision.reason;
    else if (this.override) chip.title = `supervisor override: ${this.override}`;
    this.path.appendChild(chip);
    this.chips.set(node, chip);
    this.pending = null;
    this.override = null;
  }

  onRouting(d) {
    const pill = el("span", `route-model tier-${d.tier}${d.fallback ? " fallback" : ""}`);
    pill.append(el("span", "route-site", d.call_site), document.createTextNode(` ${d.provider}`));
    if (d.fallback) pill.appendChild(el("span", "route-warn", " ⚠ fallback"));
    pill.title = `${d.tier} tier · ${d.model} · ${d.dimension}: ${d.reason}`;
    this.models.appendChild(pill);
    this.addWhy(`${d.call_site} → ${d.tier} (${d.provider} ${d.model}).`, d.reason);
    this.addTraceRow({ step: null, node: d.call_site, agent: d.agent, kind: "routing",
      detail: `${d.tier} ${d.provider} ${d.model} (${d.dimension}): ${d.reason}`, ms: null });
  }

  onTrace(t) {
    if (t.kind === "decision" && t.node === "supervisor" && t.decision) {
      this.pending = t.decision;
      const target = t.decision.next === "synthesize" ? "answer now" : t.decision.next;
      this.addWhy(`Supervisor → ${target} (${t.decision.task_type}).`, t.decision.reason);
    } else if (t.kind === "error" && t.node === "supervisor") {
      this.override = t.detail;
      this.addWhy("Supervisor override.", t.detail);
    } else if (t.kind === "enter" && (WORKERS.has(t.node) || t.node === "synthesize")) {
      this.addPathChip(t.node);
    } else if (t.kind === "error" && WORKER_FAILURES.has(t.detail) && this.chips.has(t.node)) {
      const chip = this.chips.get(t.node);
      chip.classList.add("failed");
      chip.title = `${chip.title ? `${chip.title} — ` : ""}failed: ${t.detail}`;
    }
    this.addTraceRow(t);
  }

  addTraceRow(t) {
    const row = el("li", `trace-row kind-${t.kind}`);
    const who = t.agent && t.agent !== t.node ? `${t.node}/${t.agent}` : t.node;
    row.append(
      el("span", "t-step", t.step == null ? "" : String(t.step)),
      el("span", "t-who", who),
      el("span", "t-kind", t.kind.replace("_", " ")),
      el("span", "t-detail", t.detail || ""),
      el("span", "t-ms", t.ms == null ? "" : `${t.ms} ms`),
    );
    this.traceRows.appendChild(row);
    this.events += 1;
    this.traceSummary.textContent = `Trace · ${this.events} events`;
  }

  onToken(text) {
    this.text += text;
    this.render();
  }

  onCitations(report) {
    this.cites = { valid: new Map(report.valid.map((c) => [c.id, c])), invalid: new Set(report.invalid) };
    this.render();
    this.sources.replaceChildren();
    if (!report.valid.length && !report.invalid.length) return;
    this.sources.hidden = false;
    this.sources.appendChild(el("div", "sources-title", "Sources"));
    // One line per paper (or web page), listing the chunks cited from it.
    const groups = new Map();
    for (const c of report.valid) {
      const key = c.source === "web" ? c.id : `${c.source}:${c.arxiv_id}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(c);
    }
    const list = el("ol");
    for (const cites of groups.values()) {
      const first = cites[0];
      const item = el("li");
      const url = citationUrl(first);
      const name = url ? el("a", null, first.title || first.id) : el("span", null, first.title || first.id);
      if (url) Object.assign(name, { href: url, target: "_blank", rel: "noopener noreferrer" });
      item.appendChild(name);
      if (first.arxiv_id) item.appendChild(el("span", "source-id", first.arxiv_id));
      if (first.source === "corpus") {
        const where = cites.map((c) => [c.section && `§${c.section}`, c.page != null && `p.${c.page}`].filter(Boolean).join(" "));
        item.appendChild(el("span", "source-where", where.filter(Boolean).join(" · ")));
      } else {
        item.appendChild(el("span", `source-tag source-${first.source}`, first.source === "db" ? "paper database" : "web"));
      }
      list.appendChild(item);
    }
    if (report.valid.length) this.sources.appendChild(list);
    if (report.invalid.length) {
      this.sources.appendChild(el("div", "sources-unverified", `Unverified (not retrieved this turn): ${report.invalid.join(", ")}`));
    }
  }

  onError(message) {
    this.root.insertBefore(el("div", "error", message), this.trace);
  }

  onDone({ timings, usage }) {
    const parts = [`TTFT ${formatMs(timings.ttft_ms)}`, `retrieval ${formatMs(timings.retrieval_ms)}`,
      `LLM ${formatMs(timings.llm_ms)}`, `total ${formatMs(timings.total_ms)}`];
    for (const [provider, u] of Object.entries(usage || {})) {
      parts.push(`${provider} ${formatTokens(u.input)} in / ${formatTokens(u.output)} out`);
    }
    this.meta.textContent = parts.join(" · ");
    this.traceSummary.textContent = `Trace · ${this.events} events · ${formatMs(timings.total_ms)}`;
  }

  handle({ event, data }) {
    if (event === "routing") this.onRouting(data);
    else if (event === "trace") this.onTrace(data);
    else if (event === "token") this.onToken(data.text);
    else if (event === "citations") this.onCitations(data);
    else if (event === "error") this.onError(data.message);
    else if (event === "done") this.onDone(data);
  }
}

// ---------------------------------------------------------------- the stream

// Parse one SSE frame (lines between blank lines). Comment lines start with ":".
function parseFrame(frame) {
  let event = "message";
  const data = [];
  for (const line of frame.split("\n")) {
    if (line === "" || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (data.length === 0) return null;
  return { event, data: JSON.parse(data.join("\n")) };
}

async function streamAnswer(message, turn) {
  const response = await fetch("/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (response.status === 422) {
    throw new Error("That message can't be sent. Keep it under 8,000 characters.");
  }
  if (!response.ok || !response.body) {
    throw new Error(`Request failed (HTTP ${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDone = false;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n");
    // Complete frames end with a blank line; keep the incomplete tail for the next chunk.
    const frames = buffer.split("\n\n");
    buffer = frames.pop();
    for (const frame of frames) {
      const parsed = parseFrame(frame);
      if (!parsed) continue;
      turn.handle(parsed);
      if (parsed.event === "done") sawDone = true;
    }
  }
  if (!sawDone) throw new Error("The stream ended unexpectedly.");
}

async function send() {
  const message = input.value.trim();
  if (!message || streaming) return;

  streaming = true;
  sendButton.disabled = true;
  input.value = "";
  addUserMessage(message);
  const turn = new Turn();

  try {
    await streamAnswer(message, turn);
  } catch (err) {
    turn.onError(err.message || "Something went wrong.");
  } finally {
    streaming = false;
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
