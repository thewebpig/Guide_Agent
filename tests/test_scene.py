"""场景模型与加载逻辑的测试。

每个测试都围绕“输入/输出是否与约定行为一致”编写行为断言，
目的是把数据结构、异常路径和底层 cause 链固定下来。
"""

from json import JSONDecodeError, dumps
from pathlib import Path

import pytest
from pydantic import ValidationError

from guide_agent.config import load_json
from guide_agent.scene import (
    Scene,
    SceneLoadError,
    load_scene,
)


# 用pathlib从测试文件自身的位置组装合成测试夹具路径。
# 这样不依赖当前工作目录，CI与本地都能找到同一份资源文件。
DEMO_SCENE_PATH = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "demo_scene"
    / "scene.json"
)


def test_load_scene_reads_demo_scene() -> None:
    # 读入真实场景文件并校验基础属性。
    # isinstance防卫：先证明返回类型是Scene，后续字段断言才有意义。
    scene = load_scene(DEMO_SCENE_PATH)

    assert isinstance(scene, Scene)
    assert scene.scene_id == "demo_tech_center"
    # 演示场景约定包含 6 个 POI、5 条边；这里通过 len 作为“结构不变性”检查。
    assert len(scene.pois) == 6
    assert len(scene.route_graph.edges) == 5


def test_scene_rejects_duplicate_poi_ids() -> None:
    # 先把 JSON 文件转成 dict，便于在内存里改造输入，验证模型校验逻辑。
    raw_data = load_json(DEMO_SCENE_PATH)
    assert isinstance(raw_data, dict)

    # 取出 POI 列表并附加一个同样的 dict，故意制造重复 id（业务约束级错误）。
    pois = raw_data["pois"]
    assert isinstance(pois, list)

    duplicate = dict(pois[0])
    pois.append(duplicate)

    # 断言：重复 ID 触发 ValidationError，且错误信息包含具体冲突 id。
    with pytest.raises(
        ValidationError,
        match="duplicate POI id: entrance",
    ):
        Scene.model_validate(raw_data)


def test_scene_rejects_unknown_route_node() -> None:
    # 构造“边引用了不存在的节点”场景：把第一条边的 to_id 改成 ghost。
    raw_data = load_json(DEMO_SCENE_PATH)
    assert isinstance(raw_data, dict)

    route_graph = raw_data["route_graph"]
    assert isinstance(route_graph, dict)

    edges = route_graph["edges"]
    assert isinstance(edges, list)

    first_edge = edges[0]
    assert isinstance(first_edge, dict)
    first_edge["to_id"] = "ghost"

    # 模型级后置校验(validate_route_nodes)应捕获这类引用完整性问题并抛 ValidationError。
    with pytest.raises(
        ValidationError,
        match="unknown route node: ghost",
    ):
        Scene.model_validate(raw_data)


def test_scene_rejects_negative_route_distance() -> None:
    # 把第一条边距离改为负数，触发 Field(ge=0) 的字段约束。
    raw_data = load_json(DEMO_SCENE_PATH)
    assert isinstance(raw_data, dict)

    route_graph = raw_data["route_graph"]
    assert isinstance(route_graph, dict)

    edges = route_graph["edges"]
    assert isinstance(edges, list)

    first_edge = edges[0]
    assert isinstance(first_edge, dict)
    first_edge["distance"] = -1

    with pytest.raises(
        ValidationError,
        match="greater than or equal to 0",
    ):
        Scene.model_validate(raw_data)


def test_scene_rejects_unknown_fields_instead_of_silently_discarding_them() -> None:
    raw_data = load_json(DEMO_SCENE_PATH)
    assert isinstance(raw_data, dict)
    raw_data["pois"][0]["misspelled_navigation_status"] = "restricted"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Scene.model_validate(raw_data)


def test_load_scene_reports_missing_file(tmp_path: Path) -> None:
    # tmp_path 是 pytest 提供的自动清理临时目录，避免污染仓库。
    missing_path = tmp_path / "missing_scene.json"

    # load_scene 内部在路径不存在时会抛 SceneLoadError；
    # as error_info 让我们还能检查“异常原因链”里是否保留原始 FileNotFoundError。
    with pytest.raises(
        SceneLoadError,
        match="missing_scene.json",
    ) as error_info:
        load_scene(missing_path)

    assert isinstance(error_info.value.__cause__, FileNotFoundError)


def test_load_scene_reports_invalid_json(tmp_path: Path) -> None:
    # 写入一个非法 JSON（多一个尾逗号）到临时文件，模拟读取层失败。
    invalid_path = tmp_path / "invalid_scene.json"
    invalid_path.write_text(
        '{"scene_id": "demo",}',
        encoding="utf-8",
    )

    with pytest.raises(
        SceneLoadError,
        match="invalid_scene.json",
    ) as error_info:
        load_scene(invalid_path)

    # __cause__ 能精确分辨是 JSONDecodeError，不与其它错误类型混淆。
    assert isinstance(error_info.value.__cause__, JSONDecodeError)


def test_load_scene_reports_validation_error(tmp_path: Path) -> None:
    # 先从 demo 场景拿到合法基础数据，再有意复制一个 POI 制造“重复ID”。
    raw_data = load_json(DEMO_SCENE_PATH)
    assert isinstance(raw_data, dict)

    pois = raw_data["pois"]
    assert isinstance(pois, list)
    pois.append(dict(pois[0]))

    # 不直接修改正式场景文件，而是写一份临时文件用于验证加载流程。
    invalid_scene_path = tmp_path / "duplicate_scene.json"
    invalid_scene_path.write_text(
        dumps(raw_data, ensure_ascii=False),
        encoding="utf-8",
    )

    # load_scene 应把验证异常包装为 SceneLoadError，但 message 仍可保留关键业务文案。
    with pytest.raises(
        SceneLoadError,
        match="duplicate POI id: entrance",
    ) as error_info:
        load_scene(invalid_scene_path)

    # 断言包装后的消息里仍包含原文件名，也可向上游展示“是哪一个输入”失败。
    assert invalid_scene_path.name in str(error_info.value)
    # 检查异常链，确认底层确实是 ValidationError。
    assert isinstance(error_info.value.__cause__, ValidationError)
