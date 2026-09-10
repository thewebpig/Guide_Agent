"use strict";

const SESSION_KEY = "hfut-guide-session";
const state = {
  config: null,
  pending: false,
  hasMessages: false,
  sessionId: localStorage.getItem(SESSION_KEY) || newSessionId(),
};
const $ = (id) => document.getElementById(id);

function newSessionId() { return crypto.randomUUID().replaceAll("-", ""); }
localStorage.setItem(SESSION_KEY, state.sessionId);

function make(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
function empty(target, title, body) {
  target.replaceChildren();
  const node = make("div", "empty");
  node.append(make("strong", "", title), make("p", "", body));
  target.append(node);
}
function resetMessages() {
  state.hasMessages = false;
  empty($("messages"), "你好，我是中心导览助手", "可以询问场馆信息、教师公开资料、房间位置和仿真路线。");
  empty($("tracePanel"), "暂无回答依据", "提问后可在这里查看工具调用、资料来源和耗时。");
}
function addMessage(role, content, error = false) {
  if (!state.hasMessages) { $("messages").replaceChildren(); state.hasMessages = true; }
  const node = make("article", `message ${role}${error ? " error" : ""}`);
  node.append(make("div", "message-label", role === "user" ? "你" : "导览助手"), make("div", "", content));
  $("messages").append(node);
  node.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function busy(value) {
  state.pending = value;
  ["sendButton", "question", "clearButton", "currentLocation"].forEach((id) => {
    $(id).disabled = value || !state.config;
  });
  $("sendButton").textContent = value ? "正在回答…" : "发送";
}
function showEvidence(data) {
  const panel = $("tracePanel");
  panel.replaceChildren();
  const meta = make("div", "meta");
  [["回答耗时", `${data.elapsed_ms} ms`], ["请求编号", data.request_id]].forEach(([key, value]) => {
    const item = make("div", "metric");
    item.append(make("span", "", key), make("b", "", value));
    meta.append(item);
  });
  panel.append(meta);
  if (data.sources.length) {
    panel.append(make("p", "eyebrow", "资料来源"));
    data.sources.forEach((source) => {
      const item = make("div", "source");
      item.append(make("b", "", source.source), make("div", "", `${source.chunk_id} · 检索分数 ${source.score.toFixed(2)}`));
      panel.append(item);
    });
  }
  if (data.traces.length) {
    panel.append(make("p", "eyebrow", "工具记录"));
    data.traces.forEach((trace) => {
      const item = make("details", "trace");
      const summary = make("summary", "", trace.tool_name);
      summary.append(make("span", "trace-status", trace.business_status || trace.call_status));
      item.append(summary, make("pre", "", JSON.stringify(trace.result, null, 2)));
      panel.append(item);
    });
  }
  if (!data.sources.length && !data.traces.length) panel.append(make("div", "empty", "本次回答没有可展示的工具记录。"));
}
async function send(question) {
  const cleaned = question.trim();
  if (state.pending || !cleaned) return;
  addMessage("user", cleaned);
  $("question").value = "";
  busy(true);
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: cleaned, session_id: state.sessionId, current_location: $("currentLocation").value }),
    });
    const data = await response.json();
    state.sessionId = data.session_id || state.sessionId;
    localStorage.setItem(SESSION_KEY, state.sessionId);
    addMessage("assistant", data.answer, !response.ok);
    showEvidence(data);
  } catch (_) {
    addMessage("assistant", "请求未能完成，请确认本地服务仍在运行。", true);
  } finally {
    busy(false);
    $("question").focus();
  }
}
async function startNewConversation() {
  if (state.pending) return;
  try { await fetch(`/sessions/${encodeURIComponent(state.sessionId)}`, { method: "DELETE" }); }
  finally {
    state.sessionId = newSessionId();
    localStorage.setItem(SESSION_KEY, state.sessionId);
    $("question").value = "";
    resetMessages();
  }
}
async function init() {
  try {
    const response = await fetch("/demo/config");
    if (!response.ok) throw Error();
    state.config = await response.json();
    $("sceneTitle").textContent = state.config.scene.name;
    $("routeNotice").textContent = state.config.scene.route_notice;
    const status = $("configStatus");
    status.textContent = state.config.model_configured ? "服务已就绪" : "等待部署人员配置模型密钥";
    status.className = `status ${state.config.model_configured ? "ready" : "warn"}`;
    const location = $("currentLocation");
    state.config.scene.locations.forEach((poi) => {
      const option = document.createElement("option");
      option.value = poi.id;
      option.textContent = `${poi.name}${poi.floor === null ? "" : ` · ${poi.floor}层`}`;
      option.selected = poi.id === state.config.scene.default_location;
      location.append(option);
    });
    state.config.scene.preset_questions.forEach((question) => {
      const button = make("button", "", question);
      button.type = "button";
      button.onclick = () => send(question);
      $("presets").append(button);
    });
    resetMessages();
    busy(false);
  } catch (_) {
    $("configStatus").textContent = "服务配置读取失败";
    $("configStatus").className = "status warn";
  }
}
$("chatForm").onsubmit = (event) => { event.preventDefault(); send($("question").value); };
$("clearButton").onclick = startNewConversation;
init();
