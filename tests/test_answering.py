from guide_agent.agent_contracts import AgentResult, ToolTrace
from guide_agent.answering import DEFAULT_DEMO_MINIMUM_SCORE, build_trusted_answer


def _knowledge_result(score: float) -> AgentResult:
    return AgentResult(
        status="completed",
        answer=None,
        tool_traces=(
            ToolTrace(
                tool_name="search_knowledge",
                arguments={"query": "体验中心常规开放时间是什么？", "top_k": 5},
                result={
                    "status": "ok",
                    "tool_name": "search_knowledge",
                    "data": [
                        {
                            "source": "visitor_guide.md",
                            "chunk_id": "opening-hours",
                            "text": "常规开放时间为周二至周日 9:30-17:30。",
                            "score": score,
                        }
                    ],
                    "error": None,
                },
                status="ok",
            ),
        ),
        error=None,
        response_id="fixture-response",
    )


def test_demo_threshold_accepts_calibrated_positive_query() -> None:
    """A relevant semantic evidence chunk is accepted at the demo threshold."""

    answer = build_trusted_answer(_knowledge_result(0.715108))

    assert DEFAULT_DEMO_MINIMUM_SCORE == 0.50
    assert answer.status == "ok"
    assert "9:30-17:30" in answer.answer


def test_demo_threshold_rejects_representative_no_answer_scores() -> None:
    """Scores below the calibrated threshold remain rejected."""

    # Measured after Markdown semantic chunking with BAAI/bge-small-zh-v1.5:
    # the full parking question scores 0.424282 and the overly-short query
    # "停车" scores 0.496596, both below the 0.50 demonstration threshold.
    for score in (0.424282, 0.496596):
        answer = build_trusted_answer(_knowledge_result(score))
        assert answer.status == "no_answer"


def test_trusted_answer_strips_heading_context_but_keeps_evidence_paragraph() -> None:
    result = _knowledge_result(0.8)
    trace = result.tool_traces[0]
    trace.result["data"][0]["text"] = (
        "# 星河科技体验中心访客指南\n## 开放与访问规则\n\n"
        "星河科技体验中心的常规开放时间为周二至周日9:30—17:30，16:30停止入场。"
    )

    answer = build_trusted_answer(result)

    assert answer.status == "ok"
    assert answer.answer == "星河科技体验中心的常规开放时间为周二至周日9:30—17:30，16:30停止入场。"
    assert "#" not in answer.answer


def test_route_intent_not_found_cannot_be_overridden_by_later_rag_hit() -> None:
    result = AgentResult(
        status="completed",
        answer="第二学术报告厅的介绍",
        tool_traces=(
            ToolTrace(
                tool_name="lookup_poi",
                arguments={"poi_id": "hall_4"},
                result={"status": "ok", "tool_name": "lookup_poi", "data": {"status": "not_found", "poi": None}, "error": None},
                status="ok",
            ),
            _knowledge_result(0.8).tool_traces[0],
        ),
        error=None,
        response_id="fixture-response",
    )

    answer = build_trusted_answer(result, question="带我去第四学术报告厅")

    assert answer.status == "not_found"
    assert "无法规划路线" in answer.answer
    assert "第二学术报告厅" not in answer.answer


def test_route_intent_prefers_unverified_location_over_person_profile() -> None:
    unverified = ToolTrace(
        tool_name="lookup_poi",
        arguments={"poi_id": "yang_shanlin_office"},
        result={
            "status": "ok",
            "tool_name": "lookup_poi",
            "data": {
                "status": "ok",
                "poi": {
                    "name": "杨善林院士办公地点（待确认）",
                    "description": "具体办公室没有公开。",
                    "position": {"floor": 14},
                    "entity_type": "place",
                    "navigation_status": "location_unverified",
                    "public_info": {},
                },
            },
            "error": None,
        },
        status="ok",
    )
    profile = ToolTrace(
        tool_name="lookup_poi",
        arguments={"poi_id": "yang_shanlin_profile"},
        result={
            "status": "ok",
            "tool_name": "lookup_poi",
            "data": {
                "status": "ok",
                "poi": {
                    "name": "杨善林院士",
                    "description": "教师公开资料。",
                    "position": {"floor": 14},
                    "entity_type": "person",
                    "navigation_status": "not_navigable",
                    "public_info": {"office_verification": "unverified"},
                },
            },
            "error": None,
        },
        status="ok",
    )
    result = AgentResult("completed", "ignored", (unverified, profile), None, "id")

    answer = build_trusted_answer(result, question="带我去杨善林院士办公室")

    assert answer.status == "location_unverified"
    assert "具体办公室房间号" in answer.answer
    assert "14层" not in answer.answer


def test_person_profile_never_exposes_modeling_coordinates_as_office() -> None:
    trace = ToolTrace(
        tool_name="lookup_poi",
        arguments={"poi_id": "yang_shanlin_profile"},
        result={
            "status": "ok",
            "tool_name": "lookup_poi",
            "data": {
                "status": "ok",
                "poi": {
                    "name": "杨善林院士",
                    "description": "教师公开资料。",
                    "position": {"floor": 14},
                    "entity_type": "person",
                    "navigation_status": "not_navigable",
                    "public_info": {"office_verification": "unverified"},
                },
            },
            "error": None,
        },
        status="ok",
    )

    answer = build_trusted_answer(
        AgentResult("completed", "ignored", (trace,), None, "id")
    )

    assert answer.status == "location_unverified"
    assert "14层" not in answer.answer
    assert "没有可核实的具体办公室房间号" in answer.answer
