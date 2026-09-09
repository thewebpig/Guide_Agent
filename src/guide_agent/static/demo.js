"use strict";
const state = { config: null, pending: false },
  $ = (id) => document.getElementById(id);
function make(t, c, v) {
  const n = document.createElement(t);
  if (c) n.className = c;
  if (v !== undefined) n.textContent = v;
  return n;
}
function empty(t, a, b) {
  t.replaceChildren();
  const n = make("div", "empty");
  n.append(make("strong", "", a), make("p", "", b));
  t.append(n);
}
function messages() {
  empty(
    $("messages"),
    "从一个真实问题开始",
    "模型经由 LangChain 调用 MCP 工具；过程将在右侧展示。",
  );
}
function trace(a) {
  empty($("tracePanel"), a, "工具调用、引用和耗时只在服务端真实返回后显示。");
}
function msg(r, c, e = false) {
  const n = make("article", `message ${r}${e ? " error" : ""}`);
  n.append(
    make("div", "message-label", r === "user" ? "你的问题" : "导览助手"),
    make("div", "", c),
  );
  $("messages").append(n);
}
function busy(v) {
  state.pending = v;
  ["sendButton", "question", "clearButton"].forEach(
    (id) => ($(id).disabled = v || !state.config),
  );
  $("sendButton").textContent = v ? "处理中…" : "发送问题";
}
function show(d) {
  const p = $("tracePanel");
  p.replaceChildren();
  const m = make("div", "meta");
  [
    ["服务端总耗时", `${d.elapsed_ms} ms`],
    ["请求 ID", d.request_id],
    ["API 格式", d.api_format],
  ].forEach(([k, v]) => {
    const x = make("div", "metric");
    x.append(make("span", "", k), make("b", "", v));
    m.append(x);
  });
  p.append(m);
  if (d.sources.length) {
    p.append(make("p", "eyebrow", "可信来源"));
    d.sources.forEach((s) => {
      const x = make("div", "source");
      x.append(
        make("b", "", s.source),
        make(
          "div",
          "",
          `${s.chunk_id} · 相似度 ${s.score.toFixed(2)}（非概率）`,
        ),
      );
      p.append(x);
    });
  }
  if (d.traces.length) {
    p.append(make("p", "eyebrow", "MCP 工具调用"));
    d.traces.forEach((t) => {
      const x = make("details", "trace"),
        s = make("summary", "", t.tool_name);
      const callStatus = t.call_status ?? t.status;
      const businessStatus = t.business_status;
      s.append(make("span", "trace-status", `调用：${callStatus}`));
      if (businessStatus !== null && businessStatus !== undefined) {
        s.append(
          make(
            "span",
            `trace-status ${businessStatus === "ok" ? "ok" : "business-failed"}`,
            `业务：${businessStatus}`,
          ),
        );
      }
      x.append(
        s,
        make("p", "", "参数"),
        make("pre", "", JSON.stringify(t.arguments, null, 2)),
        make("p", "", "结果"),
        make("pre", "", JSON.stringify(t.result, null, 2)),
      );
      p.append(x);
    });
  } else
    p.append(
      make(
        "div",
        "empty",
        d.status === "ok" ? "本次回答没有调用工具。" : "未取得工具执行记录。",
      ),
    );
}
async function send(q) {
  if (state.pending || !q.trim()) return;
  $("messages").replaceChildren();
  msg("user", q.trim());
  trace("处理中");
  busy(true);
  try {
    const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q.trim() }),
      }),
      d = await r.json();
    msg("assistant", d.answer, !r.ok);
    show(d);
  } catch (_) {
    msg("assistant", "请求未能完成，请确认本地服务仍在运行。", true);
    trace("请求未完成");
  } finally {
    busy(false);
  }
}
async function init() {
  try {
    const r = await fetch("/demo/config");
    if (!r.ok) throw Error();
    state.config = await r.json();
    $("sceneTitle").textContent = state.config.scene.name;
    $("format").textContent = `OpenAI-compatible · ${state.config.api_format}`;
    const s = $("configStatus");
    s.textContent = state.config.model_configured
      ? "已配置 · 连接待验证"
      : "未配置模型 · 运行 start_demo.ps1 -ConfigureModel";
    s.className = `status ${state.config.model_configured ? "ready" : "warn"}`;
    state.config.scene.preset_questions.forEach((q) => {
      const b = make("button", "", q);
      b.type = "button";
      b.onclick = () => {
        $("question").value = q;
        $("question").focus();
      };
      $("presets").append(b);
    });
    messages();
    trace("等待请求");
    busy(false);
  } catch (_) {
    $("configStatus").textContent = "无法读取演示配置";
    $("configStatus").className = "status warn";
  }
}
$("chatForm").onsubmit = (e) => {
  e.preventDefault();
  send($("question").value);
};
$("clearButton").onclick = () => {
  $("question").value = "";
  messages();
  trace("等待请求");
};
init();
