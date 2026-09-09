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


def test_demo_threshold_accepts_calibrated_shortened_positive_query() -> None:
    """0.499150 is the observed score for MiniCPM's shortened positive query."""

    answer = build_trusted_answer(_knowledge_result(0.499150))

    assert DEFAULT_DEMO_MINIMUM_SCORE == 0.48
    assert answer.status == "ok"
    assert "9:30-17:30" in answer.answer


def test_demo_threshold_rejects_representative_no_answer_scores() -> None:
    """Observed parking/no-answer scores remain below the calibrated threshold."""

    for score in (0.408374, 0.384459, 0.328818):
        answer = build_trusted_answer(_knowledge_result(score))
        assert answer.status == "no_answer"
