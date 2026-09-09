# pytest 是整个测试框架的核心入口：提供 fixture、parametrize、raises 等测试机制。
# ValidationError 是 pydantic 的校验异常类型，接下来会在非法输入构造模型时断言它会被抛出。
import pytest
from pydantic import ValidationError

# 被测对象从项目模块里导入；后续测试围绕 POI 模型、仓库与 Position 坐标展开。
from guide_agent.poi import POI, POIRepository, Position


@pytest.fixture
def pois() -> list[POI]:
    # fixture：每个测试用例需要用到 POI 列表时，pytest 会自动执行这个函数并把返回值注入参数 `pois`。
    # 写 fixture 的好处是：复用测试数据、集中管理输入、每个用例拿到的是新实例（不共享可变状态）。
    # `-> list[POI]` 是返回值类型标注，说明这个函数预期返回“由 POI 对象组成的列表”。
    return [
        POI(
            id="robot",
            name="Guide Robot",
            description="A multimodal reception robot.",
            aliases=["Reception Assistant"],
            tags=["AI", "Entrance"],
            position=Position(x=1.5, y=2.0, floor=1),
        ),
        POI(
            id="lab",
            name="Autonomous Driving Lab",
            aliases=["AD Lab"],
            tags=["Research", "Vehicle"],
            position=Position(x=8.0, y=5.5, floor=2),
        ),
    ]


def test_poi_defaults_are_independent() -> None:
    # 这里验证“可变默认值”不会互相污染：每个 POI 实例的列表字段（aliases/tags）都应是独立对象。
    # 如果实现错误地让多个实例共用同一个默认列表，修改 first.aliases 就可能污染 second.aliases。
    first = POI(id="a", name="A", position=Position(x=0, y=0))
    second = POI(id="b", name="B", position=Position(x=1, y=1))

    # 只修改第一个对象；下面会检查第二个对象仍然保持自己的空列表。
    first.aliases.append("first-only")

    # `assert actual == expected`：左侧是程序实际值，右侧是测试预期值；不相等时 pytest 报告失败。
    assert first.description == ""
    assert second.aliases == []
    assert second.tags == []
    # 创建 Position 时没有传 floor，因此这里检查默认楼层是否为 1。
    assert second.position.floor == 1


@pytest.mark.parametrize("field", ["id", "name"])
def test_poi_rejects_blank_required_text(field: str) -> None:
    # parametrize 会把 `field` 分别取值为 "id"、"name"，因此该测试函数会生成 2 个用例。
    # 对每个用例我们先给出一组合法的基础参数，然后把待测字段改为空白字符串触发校验失败。
    values = {
        "id": "valid-id",
        "name": "Valid name",
        "position": Position(x=0, y=0),
    }
    values[field] = "   "

    # pydantic 会在构造 POI 时进行类型/约束校验；这里通过 with pytest.raises 断言必须抛出 ValidationError。
    with pytest.raises(ValidationError):
        # `**values` 把字典按键名展开成关键字参数，等价于 POI(id=..., name=..., position=...)。
        POI(**values)


def test_repository_rejects_duplicate_ids(pois: list[POI]) -> None:
    # `model_copy(update=...)` 会基于已有模型快速生成副本并改动少量字段。
    # 这里刻意把 `duplicate` 的 name 改掉，但保留相同 id，测试仓库应按 id 检测重复并抛错。
    duplicate = pois[0].model_copy(update={"name": "Duplicate"})

    # 使用 `with pytest.raises(..., match=...)` 除了检查异常类型，还用正则表达式搜索异常信息。
    # 当前 match 文本没有特殊正则符号，可直观理解为异常信息里应出现 `duplicate POI id: robot`。
    # 方括号中 `[*pois, duplicate]` 是列表解包：先展开 pois 里每个元素，再追加 duplicate，语义等价于拷贝 + 追加。
    with pytest.raises(ValueError, match="duplicate POI id: robot"):
        POIRepository([*pois, duplicate])


def test_get_by_id(pois: list[POI]) -> None:
    # 测试 `get_by_id` 的两类行为：命中返回对象，不命中返回 None。
    repository = POIRepository(pois)

    # `pois[1]` 是 fixture 列表里的第二个 POI，也就是 id 为 "lab" 的对象。
    assert repository.get_by_id("lab") == pois[1]
    # `is None` 检查返回值是否就是 Python 的单例空值 None；它表示“没有找到”，而不是程序报错。
    assert repository.get_by_id("missing") is None


@pytest.mark.parametrize(
    ("keyword", "expected_ids"),
    [
        ("guide", ["robot"]),
        ("assistant", ["robot"]),
        ("ai", ["robot"]),
        ("LAB", ["lab"]),
        ("vehicle", ["lab"]),
        ("   ", []),
        ("unknown", []),
    ],
)
def test_search(pois: list[POI], keyword: str, expected_ids: list[str]) -> None:
    # `parametrize` 这里传入 7 组 (keyword, expected_ids)，所以该测试函数会被执行 7 次。
    # 加上前面的 1 个默认值测试、2 个空白字段测试、1 个重复 ID 测试和 1 个 ID 查询测试，合计 12 项。
    # 这些数据同时覆盖名称、别名、标签、大小写不敏感、空白输入和无匹配结果。
    # 最后用列表推导式 `[poi.id for poi in repository.search(keyword)]` 提取搜索结果中的 id 列表。
    # 列表推导式比手写循环更紧凑：每一项 `poi` 来自 search() 迭代结果，依次取 `.id`。
    repository = POIRepository(pois)

    # 左侧是实际搜索结果的 ID 列表，右侧是当前参数化案例给出的预期 ID 列表。
    assert [poi.id for poi in repository.search(keyword)] == expected_ids
