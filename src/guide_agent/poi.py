r"""POI 领域模型与确定性查找工具的教学文件。

本模块定义了一个 `Position` 坐标模型、一个 `POI`（Point of Interest）模型，
以及一个内存中的 POI 仓库类 `POIRepository`，用于按 id 精确取数与关键词检索。

Lesson 01 已实现字段校验、重复 ID 检查、精确查询与关键词搜索。
使用 ``uv run pytest -q`` 运行验收测试。
"""

# 模块级文档字符串（module docstring）：
# - 写在文件最前面的三引号字符串，会保存到模块的 __doc__ 属性中；
# - 前缀 r 表示“原始字符串”（raw string），反斜杠通常按普通字符处理，
#   因此 Windows 路径 ``.\.venv\...`` 不容易被误认为 \n、\t 等转义序列；
# - 双反引号是文档工具常用的代码格式标记，不影响 Python 的执行。


from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# 这里从 pydantic 导入三个核心对象：
# - BaseModel：提供字段校验、默认值、类型处理和模型序列化能力。类继承它后，属性会按声明进行校验。
# - Field：用于声明字段的默认值、长度约束、默认工厂等。
# - field_validator：用于给一个或多个字段添加自定义校验逻辑。


class Position(BaseModel):
    """场景中的一个点（Point）。

    继承 `BaseModel` 使本类成为 Pydantic 数据模型：
    - 字段值会经过类型检查；
    - 支持 `model_dump()` / `model_dump_json()` 等 Pydantic v2 模型方法；
    - 可用于与外部输入做安全的数据约束。
    """

    model_config = ConfigDict(extra="forbid")

    # x、y 是必须提供的浮点坐标；floor 是整数楼层，未传入时默认位于第 1 层。
    x: float
    y: float
    floor: int = 1


class POI(BaseModel):
    """可在多个导览场景复用的兴趣点（Point Of Interest）模型。

    同样继承 `BaseModel`，这样每个 POI 在创建时可以自动进行字段校验。
    """

    model_config = ConfigDict(extra="forbid")

    # - id 和 name 要求非空字符串（通常用于唯一标识和显示名）。
    # - description 可为空字符串默认值，给 POI 一个文本说明。
    # - aliases 是可选别名列表，便于检索匹配；默认空列表。
    # - tags 是标签列表，常用于分类；默认空列表。
    # - position 引用上面的 Position，用于表示点在场景中的位置。
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # 未采集坐标时保留 null，不能用 (0, 0, 一楼) 冒充实测位置。
    position: Position | None = None
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    entity_type: Literal["place", "person", "knowledge"] = "place"
    navigation_status: Literal[
        "navigable",
        "restricted",
        "location_unverified",
        "not_navigable",
        "simulation_only",
    ] = "navigable"
    public_info: dict[str, object] = Field(default_factory=dict)

    @field_validator("id", "name")
    @classmethod
    def required_text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class POIRepository:
    """内存中的 POI 查询仓库（暂不含持久化）。

    这个类面向工具调用：
    - __init__ 接收一组 POI 列表并构建便于快速查询的内部存储；
    - get_by_id 按精确 id 返回单个 POI；
    - search 做关键字模糊匹配并保持原始顺序。
    """

    def __init__(self, pois: list[POI]) -> None:
        # __init__ 是构造函数：实例化对象时自动执行，`self` 指向当前对象本身。
        # 参数 `pois: list[POI]` 是类型标注，表示传入一个 POI 列表；返回类型标注为 None，表示不返回其它值。
        # 列表保存原始POI顺序，供search按输入顺序返回结果。
        self._pois = list(pois)
        # 字典建立ID索引，供get_by_id进行平均O(1)的精确查询。
        self._by_id: dict[str, POI] = {}
        for poi in self._pois:
            if poi.id in self._by_id:
                raise ValueError(f"duplicate POI id: {poi.id}")

            self._by_id[poi.id] = poi

    def get_by_id(self, poi_id: str) -> POI | None:
        """按照完全相同的 id 返回 POI；不存在时返回 None。"""
        # poi_id: str 表示调用者应传入字符串类型的 id。
        # “精确 id”表示不做模糊匹配，也不自动忽略大小写。
        # 返回类型 `POI | None`（Python 3.10+）：
        # - 找到匹配 id 时返回一个 POI 对象；
        # - 未找到时返回 None（空值），由调用方决定后续处理。
        # 函数参数 `poi_id` 预期是字符串。
        return self._by_id.get(poi_id)

    def search(self, keyword: str) -> list[POI]:
        """在名称、别名和标签中执行大小写不敏感的子字符串搜索。

        `strip()`：去除关键词首尾空格。
        - `casefold()`：统一大小写，所以`"LAB"`能匹配`"Lab"`。
        - `*poi.aliases`：把所有别名展开到待搜索列表。
        - `*poi.tags`：把所有标签展开。
        - `in`：进行子字符串匹配，因此`"guide"`能匹配`"Guide Robot"`。
        - `any(...)`：名称、任意别名或任意标签有一个匹配即可。
        - 遍历`self._pois`：保持原始输入顺序。
        - 每个POI最多添加一次，即使它有多个字段同时匹配。
        """
        normalized_keyword = keyword.strip().casefold()

        if not normalized_keyword:
            return []

        results: list[POI] = []

        for poi in self._pois:
            searchable_texts = [
                poi.name,
                *poi.aliases,
                *poi.tags,
            ]

            if any(
                normalized_keyword in text.casefold()
                for text in searchable_texts
            ):
                results.append(poi)

        return results
