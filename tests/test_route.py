"""路线规划测试：用最小样例验证算法对业务约定的边界处理。"""

from pathlib import Path

import pytest

from guide_agent.route import RouteInputError, plan_route
from guide_agent.scene import load_scene


DEMO_SCENE_PATH = (
    Path(__file__).parents[1]
    / "scenes"
    / "demo"
    / "scene.json"
)


def test_plan_route_returns_shortest_path() -> None:
    """正常场景：验证返回的是“最短总距离”的路径，而不是任意可达路径。

    这里用演示场景中的多个可行路线来固定一个行为：
    1）返回路径顺序要和最短路径一致（入口到 AI 实验室）；
    2）总距离要和已知最短值一致；
    3）可达时 reason 为空，表示不是业务性失败。
    """
    scene = load_scene(DEMO_SCENE_PATH)

    result = plan_route(scene, "entrance", "ai_lab")

    assert result.path == [
        "entrance",
        "robot",
        "exhibition",
        "ai_lab",
    ]
    assert result.distance == 14.0
    assert result.reason is None


def test_plan_route_returns_zero_for_same_start_and_end() -> None:
    """同起点终点：这是输入边界条件。

    业务约定是“站在同一地点，路线是本身，距离是 0”。
    这能防止算法进入图搜索时把这种情况误判为错误或不可达。
    """
    scene = load_scene(DEMO_SCENE_PATH)

    result = plan_route(scene, "entrance", "entrance")

    assert result.path == ["entrance"]
    assert result.distance == 0.0
    assert result.reason is None


def test_plan_route_rejects_unknown_start() -> None:
    """起点不存在时应抛输入错误（参数级错误）。"""
    scene = load_scene(DEMO_SCENE_PATH)

    with pytest.raises(
        RouteInputError,
        match="unknown start POI: ghost",
    ):
        plan_route(scene, "ghost", "entrance")


def test_plan_route_rejects_unknown_end() -> None:
    """终点不存在时应抛输入错误（参数级错误）。"""
    scene = load_scene(DEMO_SCENE_PATH)

    with pytest.raises(
        RouteInputError,
        match="unknown end POI: ghost",
    ):
        plan_route(scene, "entrance", "ghost")


def test_plan_route_returns_unreachable_result() -> None:
    """终点存在但无法从起点到达时，返回可达性失败对象。

    这里用 demo 场景里故意隔离的 maintenance 节点验证：
    这不是异常，而是业务上“无路径可达”的正常返回。
    返回约定是 path 空、distance=None、reason="unreachable"。
    """
    scene = load_scene(DEMO_SCENE_PATH)

    result = plan_route(scene, "entrance", "maintenance")

    assert result.path == []
    assert result.distance is None
    assert result.reason == "unreachable"
