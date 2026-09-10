"""面向调用者的稳定导览业务接口。

本模块不重新实现地点查询或最短路径算法，而是把已经完成的
POIRepository 与 route.plan_route 组合起来，并把内部模型、None 和业务异常
统一转换成字段稳定的普通字典，供后续 CLI、FastAPI 或 LLM Tool 复用。
"""

from guide_agent import route
from guide_agent.poi import POIRepository
from guide_agent.scene import Scene


class GuideService:
    """组合场景、地点仓库和路线算法的普通Python服务。

    Scene 由调用者加载后传入，这种“从外部提供依赖”的方式叫依赖注入。
    服务类因此不写死 demo 场景路径，也不会在每次查询时重复读取JSON。
    """

    def __init__(self, scene: Scene) -> None:
        """保存同一个场景，并为其中的POI建立精确查询仓库。"""

        # 下划线表示它们是服务内部使用的实现属性，调用者应使用公开方法，
        # 不必了解Scene、POIRepository和路线算法如何组合。
        # 保存完整场景，因为路线规划不仅需要POI，还需要其中的route_graph。
        self._scene = scene
        # 仓库只接收POI列表并建立ID索引；GuideService负责把Scene中的pois传给它。
        # 这是“组合已有对象”，不是在服务层复制一遍地点查询逻辑。
        self._poi_repository = POIRepository(scene.pois)

    def lookup_poi(self, poi_id: str) -> dict[str, object]:
        """按照精确ID查询地点，并返回字段稳定、可序列化的字典。

        返回类型写成dict[str, object]，是因为键始终为字符串，而值会同时包含
        字符串、嵌套字典和None。无论是否找到，都固定返回status、poi和reason。
        """

        # 复用POIRepository已经测试过的精确ID查询；不存在时它返回None，
        # “查询合法但没有该地点”不是程序异常。
        poi = self._poi_repository.get_by_id(poi_id)

        if poi is None:
            # not_found是稳定的业务状态，调用者不需要通过捕获异常判断未命中。
            return {
                "status": "not_found",
                "poi": None,
                "reason": f"unknown POI: {poi_id}",
            }

        return {
            "status": "ok",
            # POI是Pydantic模型。model_dump(mode="json")会把它以及嵌套的
            # Position转换为普通、JSON兼容的字典，而不是把模型对象泄露给调用者。
            "poi": poi.model_dump(mode="json"),
            "reason": None,
        }

    def plan_route(
        self,
        start_id: str,
        end_id: str,
    ) -> dict[str, object]:
        """调用底层最短路径算法，并转换成稳定的服务层字典。"""

        try:
            # 当前公开方法也叫plan_route。通过route.模块命名空间调用
            # route.py中的底层函数，可清楚区分二者并避免误写成递归调用self.plan_route。
            result = route.plan_route(
                self._scene,
                start_id,
                end_id,
            )
        except route.RouteInputError as error:
            # 只把预期的“起点或终点不存在”转换为invalid_input业务结果。
            # 不捕获Exception：否则TypeError等真正的程序错误也会被隐藏，难以排查。
            return {
                "status": "invalid_input",
                "path": [],
                "distance": None,
                "reason": str(error),
            }

        # 底层RouteResult用reason="unreachable"表达合法地点之间没有路径；
        # 服务层把它提升为明确status。其余正常路线（包括同起终点）都是ok。
        business_failures = {
            "unreachable",
            "route_data_unavailable",
            "destination_restricted",
            "location_unverified",
            "not_navigable",
            "invalid_start_location",
        }
        status = result.reason if result.reason in business_failures else "ok"

        response = {
            "status": status,
            # model_dump先得到path、distance和reason字典；前面的**是字典展开语法，
            # 会把这些键值加入当前字典，与服务层新增的status组成完整结果。
            **result.model_dump(mode="json"),
        }
        if self._scene.route_graph.distance_unit != "distance":
            response["distance_unit"] = self._scene.route_graph.distance_unit
            names = {poi.id: poi.name for poi in self._scene.pois}
            response["path_names"] = [names[item] for item in result.path]
        return response
