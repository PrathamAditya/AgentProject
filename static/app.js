"use strict";

const els = {
  threadList: document.getElementById("thread-list"),
  newThread: document.getElementById("new-thread"),
  currentThread: document.getElementById("current-thread"),
  messages: document.getElementById("messages"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  budget: document.getElementById("budget"),
  memoryCounts: document.getElementById("memory-counts"),
};

let currentThread = null;
let busy = false;

function esc(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function makeThreadId() {
  return "t" + Date.now().toString(36);
}

async function refreshThreads(keepActive) {
  const threads = await api("/api/threads");
  els.threadList.innerHTML = "";
  if (!threads.length) {
    els.threadList.innerHTML = '<div class="thread-item" style="color:var(--muted)">No conversations yet</div>';
  }
  for (const t of threads) {
    const div = document.createElement("div");
    div.className = "thread-item" + (t.thread_id === currentThread ? " active" : "");
    div.innerHTML =
      `<div class="tid">${esc(t.thread_id)}</div>` +
      `<div class="meta">${t.n} messages · ${esc((t.last_ts || "").slice(0, 19).replace("T", " "))}</div>`;
    div.addEventListener("click", () => openThread(t.thread_id));
    els.threadList.appendChild(div);
  }
  if (!keepActive) selectThreadBest(threads);
}

function selectThreadBest(threads) {
  if (currentThread && threads.some((t) => t.thread_id === currentThread)) return;
  if (threads.length) openThread(threads[0].thread_id, true);
}

async function refreshMemory() {
  try {
    const mem = await api("/api/memory");
    const c = mem.counts;
    const order = [
      ["Conversational", c.conversational],
      ["Knowledge base", c.knowledge_base],
      ["Workflow", c.workflow],
      ["Toolbox", c.toolbox],
      ["Entities", c.entity],
      ["Summaries", c.summary],
    ];
    els.memoryCounts.innerHTML = order
      .map(([name, n]) => `<div class="memory-count"><b>${n}</b>${esc(name)}</div>`)
      .join("");
  } catch (e) {
    els.memoryCounts.textContent = "memory unavailable";
  }
}

function addMessage(role, text, opts = {}) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (opts.thinking) div.classList.add("thinking");
  let html = `<div class="role">${role === "user" ? "You" : "Assistant"}</div>`;
  html += esc(text);
  if (opts.steps && opts.steps.length) {
    html += `<div class="steps">steps: ${esc(opts.steps.join(" · "))}</div>`;
  }
  if (opts.summary_id) {
    html += `<div class="summary-note">&#9881; conversation consolidated into summary ${esc(opts.summary_id)}</div>`;
  }
  div.innerHTML = html;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function setBudget(status) {
  els.budget.textContent = status ? `budget: ${status}` : "";
  els.budget.className = "budget " + (status || "");
}

async function loadThreadMessages(threadId) {
  els.messages.innerHTML = "";
  const msgs = await api(`/api/threads/${encodeURIComponent(threadId)}/messages`);
  for (const m of msgs) {
    addMessage(m.role === "assistant" ? "assistant" : "user", m.content || "");
  }
}

async function openThread(threadId, silent) {
  currentThread = threadId;
  els.currentThread.textContent = threadId;
  els.threadList.querySelectorAll(".thread-item").forEach((n) =>
    n.classList.toggle("active", n.querySelector(".tid").textContent === threadId)
  );
  try {
    await loadThreadMessages(threadId);
  } catch (e) {
    els.messages.innerHTML = "";
  }
  if (!silent) await refreshThreads(true);
}

async function send() {
  const text = els.input.value.trim();
  if (!text || busy || !currentThread) return;
  busy = true;
  els.send.disabled = true;
  els.input.value = "";
  addMessage("user", text);
  const thinking = addMessage("assistant", "Thinking…", { thinking: true });
  try {
    const res = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ thread_id: currentThread, message: text }),
    });
    thinking.remove();
    addMessage("assistant", res.final_answer, {
      steps: res.steps,
      summary_id: res.summary_id,
    });
    setBudget(res.budget_status);
    await refreshThreads(true);
    await refreshMemory();
  } catch (e) {
    thinking.remove();
    addMessage("assistant", "Error: " + e.message);
  } finally {
    busy = false;
    els.send.disabled = false;
    els.input.focus();
  }
}

// events
els.send.addEventListener("click", send);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
});
els.newThread.addEventListener("click", async () => {
  openThread(makeThreadId(), true);
  els.input.focus();
});

// boot
(async function init() {
  await refreshMemory();
  const threads = await api("/api/threads");
  if (threads.length) {
    selectThreadBest(threads);
  } else {
    const id = makeThreadId();
    currentThread = id;
    els.currentThread.textContent = id;
    addMessage("assistant", "This is a new conversation. Ask me anything about research papers, or try: \"Find the Kestrel paper about streaming memory consolidation.\"");
  }
})();
