const state = {
  currentConversationId: null,
  selectedTurnId: null,
  selectedModel: "rule-based",
  conversations: [],
  turns: [],
  ragDocs: [],
  debugRecords: [],
  selectedDebugRecordId: null,
  liveWorklog: [],
  pendingAssistantText: "",
};

const el = {
  healthStatus: document.getElementById("health-status"),
  historyList: document.getElementById("history-list"),
  refreshHistoryBtn: document.getElementById("refresh-history-btn"),
  newConversationBtn: document.getElementById("new-conversation-btn"),
  chatThread: document.getElementById("chat-thread"),
  chatForm: document.getElementById("chat-form"),
  queryInput: document.getElementById("query-input"),
  modelRuleBtn: document.getElementById("model-rule-btn"),
  modelLlmBtn: document.getElementById("model-llm-btn"),
  modelIndicator: document.getElementById("model-indicator"),
  streamToggle: document.getElementById("stream-toggle"),
  ragToggle: document.getElementById("rag-toggle"),
  sendBtn: document.getElementById("send-btn"),
  agentReport: document.getElementById("agent-report"),
  agentWorklog: document.getElementById("agent-worklog"),
  ragUploadForm: document.getElementById("rag-upload-form"),
  ragImportType: document.getElementById("rag-import-type"),
  ragFileInput: document.getElementById("rag-file-input"),
  ragUploadBtn: document.getElementById("rag-upload-btn"),
  ragUploadStatus: document.getElementById("rag-upload-status"),
  ragDocList: document.getElementById("rag-doc-list"),
  refreshDebugBtn: document.getElementById("refresh-debug-btn"),
  debuggerList: document.getElementById("debugger-list"),
  debuggerDetail: document.getElementById("debugger-detail"),
};

function setHealth(ok, text) {
  el.healthStatus.textContent = text;
  el.healthStatus.style.borderColor = ok ? "#0f8f69" : "#d86a49";
}

function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function findSelectedTurn() {
  return state.turns.find((t) => t.id === state.selectedTurnId) || null;
}

function formatTraceItems(items) {
  return (items || [])
    .map((item, idx) => {
      const step = item.step || idx + 1;
      if (item.type === "thought") {
        return `#${step} thought: ${item.content || ""}`;
      }
      if (item.type === "action") {
        return `#${step} action(${item.tool}): ${safeJson(item.arguments || {})}`;
      }
      if (item.type === "observation") {
        return `#${step} observation(${item.tool}): ${safeJson(item.result || {})}`;
      }
      if (item.type === "response") {
        return `#${step} response: ${item.content || ""}`;
      }
      return `#${step} ${safeJson(item)}`;
    })
    .join("\n\n");
}

function renderHistory() {
  el.historyList.innerHTML = "";
  if (!state.conversations.length) {
    const empty = document.createElement("div");
    empty.className = "msg system";
    empty.textContent = "暂无会话，点击“新建会话”或直接发送消息。";
    el.historyList.appendChild(empty);
    return;
  }

  state.conversations.forEach((item) => {
    const node = document.createElement("button");
    node.type = "button";
    node.className = `history-item ${state.currentConversationId === item.conversation_id ? "active" : ""}`;

    const q = document.createElement("div");
    q.className = "q";
    q.textContent = item.last_query || "(empty query)";

    const meta1 = document.createElement("div");
    meta1.className = "meta";
    const m1 = document.createElement("span");
    m1.textContent = item.model || "-";
    const m2 = document.createElement("span");
    m2.textContent = `${item.turn_count || 0} 轮`;
    meta1.appendChild(m1);
    meta1.appendChild(m2);

    const meta2 = document.createElement("div");
    meta2.className = "meta";
    const t = document.createElement("span");
    t.textContent = (item.updated_at || "").replace("T", " ").slice(0, 19);
    meta2.appendChild(t);

    const preview = document.createElement("div");
    preview.className = "preview";
    preview.textContent = item.last_preview || "";

    node.appendChild(q);
    node.appendChild(meta1);
    node.appendChild(meta2);
    node.appendChild(preview);
    node.addEventListener("click", () => loadConversation(item.conversation_id));
    el.historyList.appendChild(node);
  });
}

function renderRagDocs() {
  el.ragDocList.innerHTML = "";
  if (!state.ragDocs.length) {
    const empty = document.createElement("div");
    empty.className = "msg system";
    empty.textContent = "RAG 资料库为空。";
    el.ragDocList.appendChild(empty);
    return;
  }

  state.ragDocs.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "rag-doc-item";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = doc.file_name || "(unnamed)";
    const meta = document.createElement("div");
    meta.className = "meta";
    const ts = (doc.created_at || "").replace("T", " ").slice(0, 19);
    meta.textContent = `${doc.source_type || "-"} | chunks ${doc.chunk_count || 0} | ${ts}`;
    item.appendChild(name);
    item.appendChild(meta);
    el.ragDocList.appendChild(item);
  });
}

function renderDebuggerList() {
  el.debuggerList.innerHTML = "";
  if (!state.debugRecords.length) {
    const empty = document.createElement("div");
    empty.className = "msg system";
    empty.textContent = "暂无 Debugger 记录。";
    el.debuggerList.appendChild(empty);
    if (!el.debuggerDetail.textContent) {
      el.debuggerDetail.textContent = "点击某条记录查看请求/响应详情。";
    }
    return;
  }

  state.debugRecords.forEach((item) => {
    const node = document.createElement("div");
    node.className = `debug-item ${state.selectedDebugRecordId === item.record_id ? "active" : ""}`;
    const ts = String(item.timestamp || "").replace("T", " ").slice(0, 19);
    node.textContent = `${ts} | ${item.status_code} | ${item.duration_ms}ms`;
    node.addEventListener("click", () => {
      state.selectedDebugRecordId = item.record_id;
      renderDebuggerList();
      loadDebuggerDetail(item.record_id);
    });
    el.debuggerList.appendChild(node);
  });
}

function renderChatThread() {
  el.chatThread.innerHTML = "";

  if (!state.turns.length) {
    const welcome = document.createElement("div");
    welcome.className = "msg system";
    welcome.textContent = "开始一轮对话后，这里会按轮次显示完整上下文。";
    el.chatThread.appendChild(welcome);
    return;
  }

  state.turns.forEach((turn) => {
    const card = document.createElement("div");
    card.className = `turn-card ${state.selectedTurnId === turn.id ? "active" : ""}`;
    card.addEventListener("click", () => {
      state.selectedTurnId = turn.id;
      renderChatThread();
      renderAgentPanel();
    });

    const meta = document.createElement("div");
    meta.className = "turn-meta";
    const left = document.createElement("span");
    left.textContent = `Turn ${turn.turn_index || "-"}`;
    const right = document.createElement("span");
    right.textContent = (turn.created_at || "").replace("T", " ").slice(0, 19);
    meta.appendChild(left);
    meta.appendChild(right);
    card.appendChild(meta);

    const userMsg = document.createElement("div");
    userMsg.className = "msg user";
    userMsg.textContent = turn.query || "";
    card.appendChild(userMsg);

    if (turn.status !== "ok") {
      const err = document.createElement("div");
      err.className = "msg assistant";
      err.textContent = `请求失败: ${turn.error?.message || "unknown error"}`;
      card.appendChild(err);
    } else {
      const chatAnswer = turn.chat?.final_answer || "";
      const multiReport = turn.multi_agent?.report || "";

      if (chatAnswer) {
        const assistantMsg = document.createElement("div");
        assistantMsg.className = "msg assistant";
        assistantMsg.textContent = `Chat Agent 回答:\n${chatAnswer}`;
        card.appendChild(assistantMsg);
      }

      if (multiReport && multiReport !== chatAnswer) {
        const reportMsg = document.createElement("div");
        reportMsg.className = "msg assistant";
        reportMsg.textContent = `Multi-Agent 汇总:\n${multiReport}`;
        card.appendChild(reportMsg);
      }

      const warnings = Array.isArray(turn.warnings) ? turn.warnings : [];
      if (warnings.length) {
        const warningMsg = document.createElement("div");
        warningMsg.className = "msg system";
        warningMsg.textContent = `提示: ${warnings.map((w) => w.message || w.code || "").join(" | ")}`;
        card.appendChild(warningMsg);
      }

      const ragHits = turn.rag?.hits || [];
      if (ragHits.length) {
        const ragDetails = document.createElement("details");
        ragDetails.className = "msg system";
        const ragSummary = document.createElement("summary");
        ragSummary.textContent = `RAG 引用: ${ragHits.length} 条（点击展开）`;
        const ragPre = document.createElement("pre");
        ragPre.textContent = ragHits
          .map((x, i) => `#${i + 1} [${x.doc_name}] score=${x.score}\n${x.text || ""}`)
          .join("\n\n");
        ragDetails.appendChild(ragSummary);
        ragDetails.appendChild(ragPre);
        card.appendChild(ragDetails);
      }

      const reasoningSummary = turn.chat?.reasoning_summary || "";
      if (reasoningSummary) {
        const reasoningMsg = document.createElement("div");
        reasoningMsg.className = "msg system";
        reasoningMsg.textContent = `推理摘要: ${reasoningSummary}`;
        card.appendChild(reasoningMsg);
      }

      const reactTrace = turn.chat?.react_trace || [];
      if (reactTrace.length) {
        const reactDetails = document.createElement("details");
        reactDetails.className = "msg system";
        const reactSummary = document.createElement("summary");
        reactSummary.textContent = `ReAct 轨迹: ${reactTrace.length} 条（点击展开）`;
        const reactPre = document.createElement("pre");
        reactPre.textContent = formatTraceItems(reactTrace);
        reactDetails.appendChild(reactSummary);
        reactDetails.appendChild(reactPre);
        card.appendChild(reactDetails);
      }

      const trace = turn.chat?.tool_trace || [];
      if (trace.length) {
        const details = document.createElement("details");
        details.className = "msg system";
        const summary = document.createElement("summary");
        summary.textContent = `工具轨迹: ${trace.length} 条（点击展开）`;
        const pre = document.createElement("pre");
        pre.textContent = formatTraceItems(trace);
        details.appendChild(summary);
        details.appendChild(pre);
        card.appendChild(details);
      }

      const worklog = turn.multi_agent?.worklog || [];
      if (worklog.length) {
        const processDetails = document.createElement("details");
        processDetails.className = "msg system";
        const processSummary = document.createElement("summary");
        processSummary.textContent = `Multi-Agent 过程: ${worklog.length} 步（点击展开）`;
        const processPre = document.createElement("pre");
        processPre.textContent = worklog
          .map((x, i) => `#${i + 1} ${x.agent || "Agent"} | ${x.title || ""}\n${safeJson(x.detail || {})}`)
          .join("\n\n");
        processDetails.appendChild(processSummary);
        processDetails.appendChild(processPre);
        card.appendChild(processDetails);
      }
    }

    el.chatThread.appendChild(card);
  });

  if (state.pendingAssistantText) {
    const liveCard = document.createElement("div");
    liveCard.className = "turn-card";
    const liveMeta = document.createElement("div");
    liveMeta.className = "turn-meta";
    liveMeta.textContent = "流式生成中...";
    const liveMsg = document.createElement("div");
    liveMsg.className = "msg assistant";
    liveMsg.textContent = state.pendingAssistantText;
    liveCard.appendChild(liveMeta);
    liveCard.appendChild(liveMsg);
    el.chatThread.appendChild(liveCard);
  }

  el.chatThread.scrollTop = el.chatThread.scrollHeight;
}

function renderAgentPanel() {
  const selected = findSelectedTurn();

  if (!selected) {
    el.agentReport.textContent = "请选择一轮对话查看 Agent 工作信息。";
    el.agentReport.classList.add("empty");
    el.agentWorklog.innerHTML = "";
    if (state.liveWorklog.length) {
      state.liveWorklog.forEach((item) => appendWorklogCard(item));
    }
    return;
  }

  if (selected.status !== "ok") {
    el.agentReport.textContent = "当前轮次执行失败，暂无可展示的多 Agent 结果。";
    el.agentReport.classList.add("empty");
    el.agentWorklog.innerHTML = "";
    return;
  }

  const report = selected.multi_agent?.report || "";
  el.agentReport.textContent = report || "(no report)";
  el.agentReport.classList.remove("empty");

  const worklog = selected.multi_agent?.worklog || [];
  el.agentWorklog.innerHTML = "";
  if (!worklog.length) {
    const empty = document.createElement("div");
    empty.className = "msg system";
    empty.textContent = "未收到 agent worklog。";
    el.agentWorklog.appendChild(empty);
  } else {
    worklog.forEach((item) => appendWorklogCard(item));
  }

  if (state.liveWorklog.length) {
    state.liveWorklog.forEach((item) => appendWorklogCard(item));
  }

  const reflection = selected.multi_agent?.reflection;
  if (reflection) {
    appendWorklogCard({
      agent: "ReflectAgent",
      title: "反思总结",
      detail: reflection,
    });
  }

  const ragHits = selected.rag?.hits || [];
  if (ragHits.length) {
    appendWorklogCard({
      agent: "RAGRetriever",
      title: `检索命中 ${ragHits.length} 条`,
      detail: ragHits.map((x) => ({ doc: x.doc_name, score: x.score, text: x.text })),
    });
  }
}

function appendWorklogCard(item) {
  const card = document.createElement("div");
  card.className = "agent-card";

  const head = document.createElement("div");
  head.className = "head";
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = item.agent || item.type || "Agent";
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = item.title || item.stage || "";
  head.appendChild(name);
  head.appendChild(title);

  const pre = document.createElement("pre");
  pre.textContent = safeJson(item.detail || item);

  card.appendChild(head);
  card.appendChild(pre);
  el.agentWorklog.appendChild(card);
}

async function loadConversations() {
  const resp = await fetch("/api/conversations");
  const data = await resp.json();
  state.conversations = data.items || [];
  if (!state.currentConversationId && state.conversations.length) {
    state.currentConversationId = state.conversations[0].conversation_id;
  }
  renderHistory();
}

async function loadRagDocs() {
  const resp = await fetch("/api/rag/docs?limit=100");
  if (!resp.ok) {
    state.ragDocs = [];
    renderRagDocs();
    return;
  }
  const data = await resp.json();
  state.ragDocs = data.items || [];
  renderRagDocs();
}

async function loadDebuggerRecords() {
  const resp = await fetch("/api/debug/records?limit=50&offset=0");
  if (!resp.ok) {
    state.debugRecords = [];
    state.selectedDebugRecordId = null;
    el.debuggerDetail.textContent = "加载 Debugger 记录失败。";
    renderDebuggerList();
    return;
  }
  const data = await resp.json();
  state.debugRecords = data.records || [];
  if (!state.selectedDebugRecordId && state.debugRecords.length) {
    state.selectedDebugRecordId = state.debugRecords[0].record_id;
  }
  renderDebuggerList();
  if (state.selectedDebugRecordId) {
    await loadDebuggerDetail(state.selectedDebugRecordId);
  }
}

async function loadDebuggerDetail(recordId) {
  if (!recordId) {
    el.debuggerDetail.textContent = "点击某条记录查看请求/响应详情。";
    return;
  }
  const resp = await fetch(`/api/debug/records/${recordId}`);
  if (!resp.ok) {
    el.debuggerDetail.textContent = "记录不存在或读取失败。";
    return;
  }
  const data = await resp.json();
  const detail = {
    record_id: data.record_id,
    timestamp: data.timestamp,
    status_code: data.status_code,
    duration_ms: data.duration_ms,
    request_id: data.request_id,
    error: data.error,
    tool_calls: data.tool_calls,
    request_body: data.request_body,
    response_body_preview: String(data.response_body || "").slice(0, 1600),
  };
  el.debuggerDetail.textContent = safeJson(detail);
}

async function uploadRagFile(event) {
  event.preventDefault();
  const file = el.ragFileInput.files?.[0];
  if (!file) {
    el.ragUploadStatus.textContent = "请选择文件后再上传。";
    return;
  }

  el.ragUploadBtn.disabled = true;
  el.ragUploadStatus.textContent = "上传解析中...";

  try {
    const form = new FormData();
    form.append("import_type", el.ragImportType.value || "document");
    form.append("file", file);
    const resp = await fetch("/api/rag/upload", { method: "POST", body: form });
    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload?.message || payload?.error || "上传失败");
    }
    const doc = payload.document || {};
    el.ragUploadStatus.textContent = `已入库: ${doc.file_name || file.name} (chunks ${doc.chunk_count || 0})`;
    el.ragFileInput.value = "";
    await loadRagDocs();
  } catch (err) {
    el.ragUploadStatus.textContent = `上传失败: ${String(err.message || err)}`;
  } finally {
    el.ragUploadBtn.disabled = false;
  }
}

async function loadConversation(conversationId) {
  state.currentConversationId = conversationId;
  state.selectedTurnId = null;
  state.pendingAssistantText = "";
  state.liveWorklog = [];

  const resp = await fetch(`/api/conversations/${conversationId}`);
  if (!resp.ok) {
    state.turns = [];
  } else {
    const data = await resp.json();
    state.turns = data.turns || [];
    if (state.turns.length) {
      state.selectedTurnId = state.turns[state.turns.length - 1].id;
      const model = state.turns[state.turns.length - 1].model || "rule-based";
      setModel(model === "rule-based" ? "rule-based" : "deepseek-chat");
    }
  }

  renderHistory();
  renderChatThread();
  renderAgentPanel();
}

function resetConversationSelection() {
  state.currentConversationId = null;
  state.selectedTurnId = null;
  state.turns = [];
  state.pendingAssistantText = "";
  state.liveWorklog = [];
  renderHistory();
  renderChatThread();
  renderAgentPanel();
}

function setSending(sending) {
  el.sendBtn.disabled = sending;
  el.sendBtn.textContent = sending ? "发送中..." : "发送";
}

function setModel(model) {
  state.selectedModel = model;
  const isRule = model === "rule-based";
  el.modelRuleBtn.classList.toggle("active", isRule);
  el.modelLlmBtn.classList.toggle("active", !isRule);
  el.modelRuleBtn.setAttribute("aria-pressed", isRule ? "true" : "false");
  el.modelLlmBtn.setAttribute("aria-pressed", !isRule ? "true" : "false");
  const label = isRule ? "Rule-based" : "DeepSeek";
  el.modelIndicator.textContent = `当前模型: ${label}`;
}

function parseSseEvent(block) {
  const lines = block.split(/\r?\n/);
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  const data = dataLines.join("\n");

  if (!data) {
    return null;
  }

  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return { event, data: { raw: data } };
  }
}

function handleSseParsed(parsed, context) {
  if (!parsed) {
    return;
  }
  const { event: evt, data } = parsed;
  if (evt === "meta") {
    state.currentConversationId = data.conversation_id || state.currentConversationId;
    return;
  }
  if (evt === "rag_context") {
    if (Array.isArray(data?.hits) && data.hits.length) {
      state.liveWorklog.push({
        agent: "RAGRetriever",
        title: `检索命中 ${data.hits.length} 条`,
        detail: data.hits,
      });
      renderAgentPanel();
    }
    return;
  }
  if (evt === "agent_event") {
    state.liveWorklog.push(data);
    renderAgentPanel();
    return;
  }
  if (evt === "delta") {
    state.pendingAssistantText += data.text || "";
    renderChatThread();
    return;
  }
  if (evt === "record") {
    context.finalRecord = data;
    return;
  }
  if (evt === "error") {
    throw new Error(data?.error?.message || "流式执行失败");
  }
}

async function sendQuery(event) {
  event.preventDefault();
  const query = el.queryInput.value.trim();
  const model = state.selectedModel || "rule-based";
  const useStream = !!el.streamToggle.checked;
  const useRag = !!el.ragToggle.checked;
  if (!query) {
    return;
  }

  setSending(true);
  state.pendingAssistantText = "";
  state.liveWorklog = [];

  const optimisticTurn = {
    id: `temp-${Date.now()}`,
    conversation_id: state.currentConversationId,
    turn_index: (state.turns[state.turns.length - 1]?.turn_index || 0) + 1,
    created_at: new Date().toISOString(),
    query,
    status: "pending",
    chat: { final_answer: "" },
    multi_agent: { report: "", worklog: [] },
    rag: { enabled: useRag, hits: [] },
  };
  state.turns.push(optimisticTurn);
  state.selectedTurnId = optimisticTurn.id;
  renderChatThread();
  renderAgentPanel();

  try {
    if (!useStream) {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, model, use_rag: useRag, conversation_id: state.currentConversationId }),
      });
      const record = await resp.json();
      if (!resp.ok) {
        throw new Error(record?.error?.message || record?.message || "请求失败");
      }

      await loadConversations();
      await loadConversation(record.conversation_id);
      await loadDebuggerRecords();
      el.queryInput.value = "";
      return;
    }

    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ query, model, use_rag: useRag, conversation_id: state.currentConversationId }),
    });

    if (!resp.ok || !resp.body) {
      let payload = {};
      try {
        payload = await resp.json();
      } catch {
        payload = {};
      }
      throw new Error(payload?.error?.message || payload?.message || `流式请求失败(${resp.status})`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    const context = { finalRecord: null };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        const parsed = parseSseEvent(block);
        handleSseParsed(parsed, context);
      }
    }

    if (buffer.trim()) {
      handleSseParsed(parseSseEvent(buffer), context);
    }

    if (context.finalRecord) {
      state.pendingAssistantText = "";
      state.liveWorklog = [];
      await loadConversations();
      await loadConversation(context.finalRecord.conversation_id || state.currentConversationId);
      await loadDebuggerRecords();
    } else {
      // Fallback for browsers/networks that break streaming chunks.
      const fallbackResp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, model, use_rag: useRag, conversation_id: state.currentConversationId }),
      });
      const fallbackRecord = await fallbackResp.json();
      if (!fallbackResp.ok) {
        throw new Error(fallbackRecord?.error?.message || fallbackRecord?.message || "流式降级失败");
      }
      state.pendingAssistantText = "";
      state.liveWorklog = [];
      await loadConversations();
      await loadConversation(fallbackRecord.conversation_id || state.currentConversationId);
      await loadDebuggerRecords();
    }

    el.queryInput.value = "";
  } catch (err) {
    state.pendingAssistantText = "";
    state.turns = state.turns.filter((x) => !String(x.id).startsWith("temp-"));
    const failedTurn = {
      id: `err-${Date.now()}`,
      conversation_id: state.currentConversationId,
      turn_index: (state.turns[state.turns.length - 1]?.turn_index || 0) + 1,
      created_at: new Date().toISOString(),
      query,
      status: "error",
      error: { message: String(err.message || err) },
    };
    state.turns.push(failedTurn);
    state.selectedTurnId = failedTurn.id;
    renderChatThread();
    renderAgentPanel();
  } finally {
    setSending(false);
  }
}

async function init() {
  try {
    const resp = await fetch("/api/health");
    if (resp.ok) {
      const data = await resp.json();
      setHealth(true, `在线 | ${String(data.time || "")}`);
    } else {
      setHealth(false, "离线");
    }
  } catch {
    setHealth(false, "离线");
  }

  await loadConversations();
  await loadRagDocs();
  await loadDebuggerRecords();
  if (state.currentConversationId) {
    await loadConversation(state.currentConversationId);
  } else {
    renderChatThread();
    renderAgentPanel();
  }
}

el.refreshHistoryBtn.addEventListener("click", async () => {
  await loadConversations();
  if (state.currentConversationId) {
    await loadConversation(state.currentConversationId);
  }
});

el.newConversationBtn.addEventListener("click", () => {
  resetConversationSelection();
});
el.ragUploadForm.addEventListener("submit", uploadRagFile);
el.refreshDebugBtn.addEventListener("click", loadDebuggerRecords);

el.modelRuleBtn.addEventListener("click", () => setModel("rule-based"));
el.modelLlmBtn.addEventListener("click", () => setModel("deepseek-chat"));

el.chatForm.addEventListener("submit", sendQuery);

setModel("rule-based");
init();
