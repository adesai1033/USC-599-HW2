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

function addMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  el.appendChild(body);
  messagesEl.appendChild(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

function appendChild(parent, className, text) {
  const el = document.createElement("div");
  el.className = className;
  el.textContent = text;
  parent.appendChild(el);
  return el;
}

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

function formatMs(ms) {
  return ms == null ? "—" : `${(ms / 1000).toFixed(2)} s`;
}

function handleEvent({ event, data }, answerEl, bodyEl) {
  if (event === "token") {
    bodyEl.textContent += data.text;
    answerEl.scrollIntoView({ block: "end" });
  } else if (event === "error") {
    appendChild(answerEl, "error", data.message);
  } else if (event === "done") {
    const { ttft_ms, total_ms } = data.timings;
    appendChild(answerEl, "meta", `TTFT ${formatMs(ttft_ms)} · total ${formatMs(total_ms)}`);
  }
}

async function streamAnswer(message, answerEl) {
  const bodyEl = answerEl.querySelector(".body");
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
      handleEvent(parsed, answerEl, bodyEl);
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
  addMessage("user", message);
  const answerEl = addMessage("assistant", "");

  try {
    await streamAnswer(message, answerEl);
  } catch (err) {
    appendChild(answerEl, "error", err.message || "Something went wrong.");
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
