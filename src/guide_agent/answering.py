"""把Agent运行结果收敛为可信、可序列化的用户回答。

模型可以决定调用哪个Tool，但引用、路线和地点字段必须从本次Tool
的结构化结果生成。这一层不会把模型编造的source或chunk_id传给用户。
"""

from dataclasses import dataclass

import asyncio
from typing import Protocol

from guide_agent.agent_contracts import AgentResult, ToolTrace


# This threshold is calibrated for the bundled synthetic demo scene.  It is
# intentionally not presented as a universal RAG quality threshold: replace it
# after evaluating a real venue's documents and representative questions.
DEFAULT_DEMO_MINIMUM_SCORE = 0.50


class AgentRunner(Protocol):
    def run(self, user_message: str) -> AgentResult: ...

    async def arun(self, user_message: str) -> AgentResult: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class AnswerSource:
    """从本次知识检索结果中提取的可追溯引用。"""

    source: str
    chunk_id: str
    score: float


@dataclass(frozen=True)
class TrustedAnswer:
    """HTTP层和CLI层共用的稳定回答结构。"""

    status: str
    answer: str
    tools: tuple[str, ...] = ()
    sources: tuple[AnswerSource, ...] = ()
    error: str | None = None


def _tool_names(
    traces: tuple[ToolTrace, ...],
) -> tuple[str, ...]:
    """按调用顺序保留Tool名，便于演示和调试。"""

    return tuple(
        trace.tool_name
        for trace in traces
    )


def _safe_agent_failure(
    result: AgentResult,
) -> TrustedAnswer:
    """把内部失败转为不泄露异常、密钥或Prompt的说明。"""

    messages = {
        "max_steps_exceeded": (
            "本次请求需要的工具步骤过多，"
            "已安全停止，请缩小问题范围。"
        ),
        "model_error": (
            "模型服务暂时不可用，请稍后再试。"
        ),
        "empty_response": (
            "模型未返回可用内容，请换一种问法。"
        ),
    }
    answer = messages.get(
        result.status,
        "本次请求未能完成，请稍后再试。",
    )

    return TrustedAnswer(
        status=result.status,
        answer=answer,
        tools=_tool_names(result.tool_traces),
        error=result.status,
    )


def _tool_payload(
    trace: ToolTrace,
) -> object:
    """取出ToolRegistry统一外层结果里的data。"""

    return trace.result.get("data")


def _answer_from_route(
    trace: ToolTrace,
    tools: tuple[str, ...],
) -> TrustedAnswer:
    payload = _tool_payload(trace)

    if not isinstance(payload, dict):
        return TrustedAnswer(
            status="tool_error",
            answer="路线工具未返回可用结果。",
            tools=tools,
            error="tool_error",
        )

    business_status = payload.get("status")

    if business_status == "route_data_unavailable":
        return TrustedAnswer(
            status="route_data_unavailable",
            answer="当前场景尚未录入核实过的通道和距离，无法计算路线。请向现场工作人员确认走法。",
            tools=tools,
        )

    if business_status == "unreachable":
        return TrustedAnswer(
            status="no_route",
            answer="这两个地点之间暂无可用路线。",
            tools=tools,
        )

    if business_status == "invalid_input":
        return TrustedAnswer(
            status="invalid_location",
            answer="起点或终点不存在，请先确认地点ID。",
            tools=tools,
            error="invalid_location",
        )

    navigation_failures = {
        "destination_restricted": "该区域属于内部科研与办公区域，不能默认引导访客进入。如有访问授权，请按照现场工作人员指引前往。",
        "location_unverified": "目前公开资料中没有可核实的具体办公室房间号，因此暂时无法规划路线。请以学院最新信息或现场指引为准。",
        "not_navigable": "该信息是知识资料，不是可导航的实体地点，因此无法规划路线。",
        "invalid_start_location": "起点不是经过确认的可导航地点，请重新选择当前位置。",
    }
    if business_status in navigation_failures:
        return TrustedAnswer(
            status=str(business_status),
            answer=navigation_failures[str(business_status)],
            tools=tools,
        )

    path = payload.get("path")
    distance = payload.get("distance")

    if (
        business_status != "ok"
        or not isinstance(path, list)
        or not all(isinstance(item, str) for item in path)
        or not isinstance(distance, (int, float))
        or isinstance(distance, bool)
    ):
        return TrustedAnswer(
            status="tool_error",
            answer="路线工具未返回可用结果。",
            tools=tools,
            error="tool_error",
        )

    path_names = payload.get("path_names")
    display_path = (
        path_names
        if isinstance(path_names, list)
        and len(path_names) == len(path)
        and all(isinstance(item, str) for item in path_names)
        else path
    )
    route_text = " → ".join(display_path)
    is_simulation = payload.get("distance_unit") == "simulation_weight"
    measurement = "仿真路径权重" if is_simulation else "总距离"
    disclaimer = "该路线用于系统演示，具体走法请以现场标识为准。" if is_simulation else ""
    return TrustedAnswer(
        status="ok",
        answer=(
            f"路线：{route_text}；"
            f"{measurement}：{float(distance):g}。"
            f"{disclaimer}"
        ),
        tools=tools,
    )


def _answer_from_poi(
    trace: ToolTrace,
    tools: tuple[str, ...],
) -> TrustedAnswer:
    payload = _tool_payload(trace)

    if not isinstance(payload, dict):
        return TrustedAnswer(
            status="tool_error",
            answer="地点工具未返回可用结果。",
            tools=tools,
            error="tool_error",
        )

    if payload.get("status") == "not_found":
        return TrustedAnswer(
            status="not_found",
            answer="没有找到这个地点，请确认地点ID。",
            tools=tools,
        )

    poi = payload.get("poi")

    if payload.get("status") != "ok" or not isinstance(poi, dict):
        return TrustedAnswer(
            status="tool_error",
            answer="地点工具未返回可用结果。",
            tools=tools,
            error="tool_error",
        )

    name = poi.get("name")
    description = poi.get("description")
    position = poi.get("position")

    if not isinstance(name, str) or not isinstance(description, str):
        return TrustedAnswer(
            status="tool_error",
            answer="地点工具未返回可用结果。",
            tools=tools,
            error="tool_error",
        )

    floor_text = ""
    if isinstance(position, dict):
        floor = position.get("floor")
        if isinstance(floor, int) and not isinstance(floor, bool):
            floor_text = f"，位于{floor}层"

    return TrustedAnswer(
        status="ok",
        answer=f"{name}{floor_text}。{description}",
        tools=tools,
    )


def _answer_from_knowledge(
    trace: ToolTrace,
    tools: tuple[str, ...],
    *,
    minimum_score: float,
) -> TrustedAnswer:
    payload = _tool_payload(trace)

    if not isinstance(payload, list):
        return TrustedAnswer(
            status="tool_error",
            answer="知识检索未返回可用结果。",
            tools=tools,
            error="tool_error",
        )

    evidence: list[tuple[str, AnswerSource]] = []
    seen_chunks: set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue

        source = item.get("source")
        chunk_id = item.get("chunk_id")
        text = item.get("text")
        score = item.get("score")

        if (
            not isinstance(source, str)
            or not isinstance(chunk_id, str)
            or not isinstance(text, str)
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or float(score) < minimum_score
            or chunk_id in seen_chunks
        ):
            continue

        seen_chunks.add(chunk_id)
        evidence.append(
            (
                text,
                AnswerSource(
                    source=source,
                    chunk_id=chunk_id,
                    score=float(score),
                ),
            )
        )

    if not evidence:
        return TrustedAnswer(
            status="no_answer",
            answer=(
                "当前资料中没有找到足够可靠的答案，"
                "请换一种问法或咨询现场工作人员。"
            ),
            tools=tools,
        )

    # 证据块会保留 Markdown 标题，帮助向量检索识别语境；用户回答不应把标题
    # 和同一块中的无关段落整段倾倒出去。这里不做生成式改写，只从最高分证据
    # 中抽取首个正文段落，因此每个字仍可回溯到本次检索证据。
    return TrustedAnswer(
        status="ok",
        answer=_concise_evidence_text(evidence[0][0]),
        tools=tools,
        sources=tuple(
            source
            for _, source in evidence
        ),
    )


def _concise_evidence_text(text: str) -> str:
    """Return the first substantive paragraph from a semantic evidence chunk.

    Markdown heading context belongs in the indexed evidence, but it is not a
    user-facing answer. Paragraph-aware chunking makes the first body paragraph
    a bounded, source-faithful extract rather than an arbitrary 500-character
    slice. Plain-text chunks remain unchanged.
    """

    body_lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        body_lines.append(line.strip())
    body = "\n".join(body_lines).strip()
    if not body:
        return text.strip()
    return body.split("\n\n", maxsplit=1)[0].strip()


def build_trusted_answer(
    result: AgentResult,
    *,
    minimum_score: float = DEFAULT_DEMO_MINIMUM_SCORE,
) -> TrustedAnswer:
    """根据Agent轨迹生成最终回答，不接受模型自报的引用。"""

    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError(
            "minimum_score must be between 0 and 1"
        )

    if result.status != "completed":
        return _safe_agent_failure(result)

    tools = _tool_names(result.tool_traces)

    # 只要某次Tool外层执行失败，就不用后续模型文本伪装成成功。
    for trace in result.tool_traces:
        if trace.status != "ok":
            return TrustedAnswer(
                status=trace.status,
                answer="请求的工具未能安全执行。",
                tools=tools,
                error=trace.status,
            )

    if not result.tool_traces:
        if not result.answer:
            return _safe_agent_failure(
                AgentResult(
                    status="empty_response",
                    answer=None,
                    tool_traces=(),
                    error="empty_response",
                    response_id=result.response_id,
                )
            )
        return TrustedAnswer(
            status="ok",
            answer=result.answer,
        )

    last_trace = result.tool_traces[-1]

    if last_trace.tool_name == "plan_route":
        return _answer_from_route(last_trace, tools)
    if last_trace.tool_name == "lookup_poi":
        return _answer_from_poi(last_trace, tools)
    if last_trace.tool_name == "search_knowledge":
        return _answer_from_knowledge(
            last_trace,
            tools,
            minimum_score=minimum_score,
        )

    return TrustedAnswer(
        status="unknown_tool",
        answer="请求包含未注册的工具，已安全停止。",
        tools=tools,
        error="unknown_tool",
    )


class ChatService:
    """组合Agent循环与可信回答策略。"""

    def __init__(
        self,
        agent: AgentRunner,
        *,
        minimum_score: float = DEFAULT_DEMO_MINIMUM_SCORE,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between 0 and 1"
            )
        self._agent = agent
        self._minimum_score = minimum_score

    @property
    def minimum_score(self) -> float:
        """The retrieval acceptance threshold used for this service."""

        return self._minimum_score

    def answer(self, question: str) -> TrustedAnswer:
        """执行一次完整请求并收敛为稳定回答。"""

        answer, _ = self.answer_with_trace(question)
        return answer

    def answer_with_trace(
        self,
        question: str,
    ) -> tuple[TrustedAnswer, AgentResult]:
        """执行一次请求，同时返回可信回答和本次可审计轨迹。

        演示层需要显示真实工具调用；此方法避免为取得轨迹而重复请求模型。
        """

        result = self._agent.run(question)
        return build_trusted_answer(
            result,
            minimum_score=self._minimum_score,
        ), result

    async def aanswer(self, question: str) -> TrustedAnswer:
        answer, _ = await self.aanswer_with_trace(question)
        return answer

    async def aanswer_with_trace(self, question: str) -> tuple[TrustedAnswer, AgentResult]:
        """Async equivalent which runs an old synchronous fake safely in a thread."""
        arun = getattr(self._agent, "arun", None)
        if callable(arun):
            result = await arun(question)
        else:
            result = await asyncio.to_thread(self._agent.run, question)
        return build_trusted_answer(result, minimum_score=self._minimum_score), result

    async def aanswer_with_context(
        self,
        question: str,
        *,
        history: tuple[dict[str, str], ...] = (),
        current_location: str | None = None,
    ) -> tuple[TrustedAnswer, AgentResult]:
        """Execute with server-owned conversation and location context."""
        arun = getattr(self._agent, "arun", None)
        if callable(arun):
            result = await arun(
                question,
                history=history,
                current_location=current_location,
            )
        else:
            result = await asyncio.to_thread(
                self._agent.run,
                question,
                history=history,
                current_location=current_location,
            )
        return build_trusted_answer(result, minimum_score=self._minimum_score), result

    async def aclose(self) -> None:
        close = getattr(self._agent, "aclose", None)
        if callable(close):
            await close()
