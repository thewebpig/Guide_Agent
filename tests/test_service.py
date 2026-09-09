"""稳定导览业务接口的测试。

这里不重复验证POI仓库或Dijkstra内部步骤，而是固定GuideService对调用者承诺的
参数和返回字典：成功、未找到、不可达、非法输入和同起终点必须明确区分。
"""

from pathlib import Path

import pytest

from guide_agent.scene import load_scene
from guide_agent.service import GuideService


# 从测试文件自身位置推导场景路径，不依赖pytest从哪个工作目录启动。
# parents[1]是项目根目录guide_agent，因此本地和CI都能定位同一份demo数据。
DEMO_SCENE_PATH = (
    Path(__file__).parents[1]
    / "scenes"
    / "demo"
    / "scene.json"
)


@pytest.fixture
def guide_service() -> GuideService:
    """为每个测试加载真实demo场景并创建一个新的服务实例。

    fixture集中处理重复的准备工作；测试函数只关注输入和服务层输出。
    """

    scene = load_scene(DEMO_SCENE_PATH)
    return GuideService(scene)


def test_lookup_poi_returns_serializable_poi(
    guide_service: GuideService,
) -> None:
    """地点存在时返回ok，并将嵌套Pydantic模型转换成普通字典。"""

    result = guide_service.lookup_poi("robot")

    # 先固定服务层的状态字段，再检查poi负载的类型和关键业务字段。
    assert result["status"] == "ok"
    assert result["reason"] is None

    poi_data = result["poi"]
    # isinstance断言既验证序列化结果，也让后面的字典下标访问具备明确前提。
    assert isinstance(poi_data, dict)
    assert poi_data["id"] == "robot"
    assert poi_data["name"] == "接待机器人"
    assert poi_data["position"] == {
        "x": 3.0,
        "y": 0.0,
        "floor": 1,
    }


def test_lookup_poi_returns_not_found_result(
    guide_service: GuideService,
) -> None:
    """地点ID不存在时返回not_found，而不是抛异常或省略固定字段。"""

    result = guide_service.lookup_poi("missing")

    # 整体比较能固定未找到分支的完整契约，防止字段或状态被意外改动。
    assert result == {
        "status": "not_found",
        "poi": None,
        "reason": "unknown POI: missing",
    }


def test_plan_route_returns_serializable_route(
    guide_service: GuideService,
) -> None:
    """可达时返回底层Dijkstra结果的JSON兼容字典。"""

    result = guide_service.plan_route(
        "entrance",
        "ai_lab",
    )

    # 调用者只看到普通字典，不需要接触Scene或RouteResult对象。
    assert result == {
        "status": "ok",
        "path": [
            "entrance",
            "robot",
            "exhibition",
            "ai_lab",
        ],
        "distance": 14.0,
        "reason": None,
    }


def test_plan_route_returns_unreachable_result(
    guide_service: GuideService,
) -> None:
    """地点存在但无路径时返回unreachable业务结果，不伪造路线。"""

    result = guide_service.plan_route(
        "entrance",
        "maintenance",
    )

    # 不可达不是参数错误：path为空、distance为None，并保留明确reason。
    assert result == {
        "status": "unreachable",
        "path": [],
        "distance": None,
        "reason": "unreachable",
    }


# 同一套断言分别检查未知起点和未知终点。参数化会为列表中的每组参数
# 独立生成一条测试用例，既减少重复代码，也保证两个错误分支都被执行。
@pytest.mark.parametrize(
    (
        "start_id",
        "end_id",
        "expected_reason",
    ),
    [
        (
            "ghost",
            "entrance",
            "unknown start POI: ghost",
        ),
        (
            "entrance",
            "ghost",
            "unknown end POI: ghost",
        ),
    ],
)
def test_plan_route_returns_invalid_input_result(
    guide_service: GuideService,
    start_id: str,
    end_id: str,
    expected_reason: str,
) -> None:
    """未知起点或终点都转换成字段一致的invalid_input结果。"""

    result = guide_service.plan_route(
        start_id,
        end_id,
    )

    assert result == {
        "status": "invalid_input",
        "path": [],
        "distance": None,
        "reason": expected_reason,
    }


def test_plan_route_returns_zero_for_same_start_and_end(
    guide_service: GuideService,
) -> None:
    """同起点终点是距离为0的正常路线，不是不可达或非法输入。"""

    result = guide_service.plan_route(
        "entrance",
        "entrance",
    )

    assert result == {
        "status": "ok",
        "path": ["entrance"],
        "distance": 0.0,
        "reason": None,
    }
