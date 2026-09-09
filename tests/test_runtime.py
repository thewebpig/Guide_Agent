from guide_agent.runtime import DEFAULT_SCENE_PATH, _build_scene_context, resolve_scene_path
from guide_agent.scene import load_scene


def test_demo_is_the_only_default_scene() -> None:
    assert DEFAULT_SCENE_PATH.name == "scene.json"
    assert DEFAULT_SCENE_PATH.parent.name == "demo"
    assert resolve_scene_path() == DEFAULT_SCENE_PATH


def test_scene_context_keeps_poi_tool_arguments_to_exact_ids() -> None:
    context = _build_scene_context(load_scene(DEFAULT_SCENE_PATH))

    assert "POI 工具参数允许的纯 ID 清单：entrance, robot, exhibition, ai_lab" in context
    assert "poi_id，以及 plan_route 的 start_id、end_id" in context
    assert "必须且只能填写上面清单中的一个精确纯 ID" in context
    assert "例如 entrance、ai_lab" in context
    assert "绝不能填写名称、别名、冒号后的说明、方括号注释" in context


def test_scene_context_requires_complete_knowledge_query() -> None:
    context = _build_scene_context(load_scene(DEFAULT_SCENE_PATH))

    assert "query 必须保留用户完整原问题和场馆语境" in context
    assert "不能把“体验中心常规开放时间是什么？”缩成“常规开放时间”" in context
