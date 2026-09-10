from pathlib import Path

import pytest

from guide_agent.app_settings import load_app_settings, resolve_product_path
from guide_agent.documents import load_documents
from guide_agent.route import plan_route
from guide_agent.scene import load_scene
from guide_agent.service import GuideService


ROOT = Path(__file__).parents[1]
SCENE_PATH = ROOT / "scenes" / "hfut_management_center" / "scene.json"


ROUTES = [
    ("main_entrance", "hall_1", ["main_entrance", "lobby", "hall_1"], 11),
    ("main_entrance", "hall_2", ["main_entrance", "lobby", "hall_1", "hall_2"], 15),
    ("lobby", "hall_3", ["lobby", "hall_1", "hall_2", "hall_3"], 15),
    ("hall_1", "hall_5", ["hall_1", "hall_2", "hall_3", "hall_5"], 12),
    ("hall_2", "hall_5", ["hall_2", "hall_3", "hall_5"], 8),
    ("hall_5", "main_entrance", ["hall_5", "hall_3", "hall_2", "hall_1", "lobby", "main_entrance"], 23),
    ("main_entrance", "info_point", ["main_entrance", "lobby", "info_point"], 6),
    ("main_entrance", "room_306", ["main_entrance", "lobby", "elevator_1f", "elevator_3f", "room_306"], 22),
    ("lobby", "room_306", ["lobby", "elevator_1f", "elevator_3f", "room_306"], 18),
    ("hall_1", "room_306", ["hall_1", "lobby", "elevator_1f", "elevator_3f", "room_306"], 25),
    ("room_306", "lobby", ["room_306", "elevator_3f", "elevator_1f", "lobby"], 18),
    ("room_306", "research_platforms", ["room_306", "elevator_3f", "research_platforms"], 11),
    ("lobby", "smart_manufacturing_research", ["lobby", "elevator_1f", "elevator_3f", "research_platforms", "smart_manufacturing_research"], 25),
    ("main_entrance", "intelligent_decision_research", ["main_entrance", "lobby", "elevator_1f", "elevator_3f", "research_platforms", "intelligent_decision_research"], 30),
    ("main_entrance", "wang_gang_office_1106", ["main_entrance", "lobby", "elevator_1f", "elevator_3f", "elevator_11f", "wang_gang_office_1106"], 42),
    ("main_entrance", "ren_minglun_office_1108", ["main_entrance", "lobby", "elevator_1f", "elevator_3f", "elevator_11f", "ren_minglun_office_1108"], 43),
    ("lobby", "wang_gang_office_1106", ["lobby", "elevator_1f", "elevator_3f", "elevator_11f", "wang_gang_office_1106"], 38),
    ("wang_gang_office_1106", "ren_minglun_office_1108", ["wang_gang_office_1106", "elevator_11f", "ren_minglun_office_1108"], 9),
    ("ren_minglun_office_1108", "room_306", ["ren_minglun_office_1108", "elevator_11f", "elevator_3f", "room_306"], 29),
    ("wang_gang_office_1106", "hall_2", ["wang_gang_office_1106", "elevator_11f", "elevator_3f", "elevator_1f", "lobby", "hall_1", "hall_2"], 49),
    ("main_entrance", "discipline_office_1401", ["main_entrance", "lobby", "elevator_1f", "elevator_3f", "elevator_14f", "discipline_office_1401"], 51),
    ("room_306", "discipline_office_1401", ["room_306", "elevator_3f", "elevator_14f", "discipline_office_1401"], 37),
    ("wang_gang_office_1106", "discipline_office_1401", ["wang_gang_office_1106", "elevator_11f", "elevator_3f", "elevator_14f", "discipline_office_1401"], 57),
]


@pytest.mark.parametrize("start,end,expected_path,expected_distance", ROUTES)
def test_supplied_route_acceptance(start, end, expected_path, expected_distance) -> None:
    result = plan_route(load_scene(SCENE_PATH), start, end)
    assert result.path == expected_path
    assert result.distance == expected_distance
    assert result.reason is None


def test_real_scene_and_knowledge_document_are_complete() -> None:
    scene = load_scene(SCENE_PATH)
    documents = load_documents(scene, SCENE_PATH.parent)
    assert scene.scene_id == "hfut_management_intelligent_manufacturing_center"
    assert len(scene.pois) == 24
    assert scene.route_graph.distance_unit == "simulation_weight"
    assert len(documents) == 1
    assert "屯溪路校区" in documents[0].text
    assert "工程管理与智能制造研究中心（管理学院）导览知识库" in documents[0].text


def test_extended_people_metadata_is_not_silently_discarded() -> None:
    scene = load_scene(SCENE_PATH)
    people = {poi.id: poi for poi in scene.pois}
    assert people["ren_minglun_profile"].entity_type == "person"
    assert people["ren_minglun_profile"].public_info["office"] == "管理学院1108"
    assert people["yang_shanlin_office"].navigation_status == "location_unverified"


@pytest.mark.parametrize(
    "end,status",
    [
        ("restricted_internal_area", "destination_restricted"),
        ("yang_shanlin_office", "location_unverified"),
        ("yang_shanlin_profile", "not_navigable"),
    ],
)
def test_policy_blocks_routes_even_when_graph_or_poi_exists(end: str, status: str) -> None:
    result = GuideService(load_scene(SCENE_PATH)).plan_route("main_entrance", end)
    assert result["status"] == status
    assert result["path"] == []
    assert result["distance"] is None


def test_product_route_exposes_visitor_facing_names_and_simulation_unit() -> None:
    result = GuideService(load_scene(SCENE_PATH)).plan_route("main_entrance", "hall_1")
    assert result["distance_unit"] == "simulation_weight"
    assert result["path_names"] == [
        "工程管理与智能制造研究中心主入口",
        "一层公共大厅",
        "第一学术报告厅",
    ]


def test_server_configuration_selects_real_scene_and_minicpm() -> None:
    settings = load_app_settings()
    assert settings.model.name == "MiniCPM5-2B"
    assert settings.model.api_format == "chat_completions"
    assert settings.retrieval.minimum_score == 0.50
    assert resolve_product_path(settings.scene.path) == SCENE_PATH
