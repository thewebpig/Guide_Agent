from guide_agent.runtime import DEFAULT_SCENE_PATH, resolve_scene_path


def test_demo_is_the_only_default_scene() -> None:
    assert DEFAULT_SCENE_PATH.name == "scene.json"
    assert DEFAULT_SCENE_PATH.parent.name == "demo"
    assert resolve_scene_path() == DEFAULT_SCENE_PATH
