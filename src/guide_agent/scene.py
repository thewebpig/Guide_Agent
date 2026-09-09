"""场景模型与场景文件加载逻辑。"""

from json import JSONDecodeError
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from guide_agent.config import load_json
from guide_agent.poi import POI


class RouteEdge(BaseModel):
    """场景里“两个POI之间一条路”的数据模型。

    继承 BaseModel 后，pydantic 会在构建对象时自动检查字段类型与约束，
    所以后续校验都基于“可信已清洗”的对象进行。
    """

    # 每条边起点和终点的长度至少为1，可拒绝空字符串；跨节点有效性由Scene统一检查。
    from_id: str = Field(min_length=1)
    to_id: str = Field(min_length=1)
    # 距离必须是 0 或更大；负数不被允许，避免“倒退里程”这种不合理输入。
    distance: float = Field(ge=0)


class RouteGraph(BaseModel):
    """场景里全部路线的集合。

    directed 表示有向图/无向图；当前作业场景主要是无向图，默认 False。
    edges 用 list[RouteEdge] 存边，默认工厂创建空列表，避免共享可变对象。
    """

    directed: bool = False
    # false 表示尚无核实路网；区别于已建图中某对节点不连通。
    data_available: bool = True
    # default_factory=list 会在每个RouteGraph实例里各自创建 []，避免多个实例误共享同一列表。
    edges: list[RouteEdge] = Field(default_factory=list)


class Scene(BaseModel):
    """单个场景的完整模型（ID、名称、POI、路网、提示词、文档引用）。"""

    scene_id: str = Field(min_length=1)  # 场景唯一标识，不能为空
    name: str = Field(min_length=1)  # 场景名
    pois: list[POI] = Field(min_length=1)  # 至少要有一个地点，否则场景不成立
    route_graph: RouteGraph  # 嵌套模型；Pydantic 会把 dict 自动转成 RouteGraph，再转成 RouteEdge
    system_prompt: str = Field(min_length=1)  # 与这个场景有关的系统提示词
    source_docs: list[str] = Field(default_factory=list)  # 可选参考文档路径列表

    # 校验1：POI列表中的 id 是否重复。
    # 字段级校验（Field）负责“单字段合法性”，这里是整表级业务校验（成组约束）。
    @model_validator(mode="after")
    def validate_unique_poi_ids(self) -> Self:
        seen_ids: set[str] = set()

        for poi in self.pois:
            # 只要发现重复 id，说明场景状态不一致；抛出 ValueError 交给上层统一转换。
            if poi.id in seen_ids:
                raise ValueError(f"duplicate POI id: {poi.id}")

            seen_ids.add(poi.id)

        # after 模式的校验器必须返回 self；类型标注 Self 明确“返回当前实例”。
        return self

    # 校验2：路线两端节点都必须存在于 pois 中。
    # 这是场景层面的完整性规则，不只是字段格式的问题。
    @model_validator(mode="after")
    def validate_route_nodes(self) -> Self:
        poi_ids = {poi.id for poi in self.pois}

        for edge in self.route_graph.edges:
            for node_id in (edge.from_id, edge.to_id):
                if node_id not in poi_ids:
                    raise ValueError(f"unknown route node: {node_id}")

        return self


class SceneLoadError(ValueError):
    """场景读取/解析/校验失败时的统一业务错误。

    这里继承自 ValueError，是“输入不合规”的一类异常入口，便于上层统一捕获。
    """


def load_scene(path: str | Path) -> Scene:
    """加载并校验场景文件。

    流程：
    1) 统一转为 Path（支持 str/Path 两种入参）；
    2) 调用 config.load_json 读并解析JSON文本为Python对象；
    3) 用 Scene.model_validate 触发 pydantic 字段校验 + 模型级校验（duplicate/unknown）。
    4) 统一把底层异常包装为 SceneLoadError，并保留 __cause__ 链，便于定位根因。
    """

    scene_path = Path(path)

    try:
        raw_data = load_json(scene_path)
        return Scene.model_validate(raw_data)
    # FileNotFoundError：文件不存在；JSONDecodeError：JSON语法错误；ValidationError：字段/模型校验失败。
    except (FileNotFoundError, JSONDecodeError, ValidationError) as error:
        # 保留原始异常作为 cause，调用方可通过 error.__cause__ 读取真实底层错误类型。
        raise SceneLoadError(
            f"failed to load scene {scene_path}: {error}"
        ) from error
