"""最短路线规划：基于 Dijkstra 算法从起点到终点找最短可达路径。"""

from heapq import heappop, heappush
from math import inf

from pydantic import BaseModel, Field

from guide_agent.scene import Scene


class RouteResult(BaseModel):
    """路线规划的结构化返回值。

    path：按访问顺序的地点 ID 列表，例如 ["A", "B", "C"]。
    distance：路线总距离；为 None 表示该终点不可达。
    reason：区分不可达和路网资料缺失，有结果时为 None。
    """

    # path 是一个普通 Python 列表，保存的是完整路径节点的顺序，不包含权重信息。
    path: list[str] = Field(default_factory=list)
    # distance 至少为 0；路径正常时必须是一个非负数。
    # 如果路径不可达，函数会返回 distance=None 来区别于“距离为 0 的正常路径”。
    distance: float | None = Field(default=None, ge=0)
    # reason 是附加说明；正常可达时为 None，不可达时一般是 "unreachable"。
    reason: str | None = None


class RouteInputError(ValueError):
    """起点或终点不存在时抛出的输入错误类型。"""


def plan_route(scene: Scene, start_id: str, end_id: str) -> RouteResult:
    """规划起点到终点的最短路线。

    关键约定：
    1. 如果起点/终点 ID 不在场景内，立即抛 RouteInputError（输入错误）。
    2. 如果起点与终点相同，直接返回零距离单点路径。
    3. 其余情况下，按边权（distance）执行 Dijkstra。
    """

    # 收集场景中所有 POI 的 ID，后续用于“ID 存在性检查”和图初始化。
    pois_by_id = {poi.id: poi for poi in scene.pois}
    poi_ids = set(pois_by_id)

    if start_id not in poi_ids:
        # 这是“调用者给了一个 scene 外部不存在的起点 ID”时的业务错误，
        # 不属于路径不可达，而是输入参数本身不合法。
        raise RouteInputError(f"unknown start POI: {start_id}")

    if end_id not in poi_ids:
        # 同上，终点不存在时抛同类输入错误。
        raise RouteInputError(f"unknown end POI: {end_id}")

    blocked_reasons = {
        "restricted": "destination_restricted",
        "location_unverified": "location_unverified",
        "not_navigable": "not_navigable",
    }
    end_status = pois_by_id[end_id].navigation_status
    if end_status in blocked_reasons:
        return RouteResult(reason=blocked_reasons[end_status])

    start_status = pois_by_id[start_id].navigation_status
    if start_status in {"location_unverified", "not_navigable"}:
        return RouteResult(reason="invalid_start_location")

    if not scene.route_graph.data_available:
        return RouteResult(reason="route_data_unavailable")

    if start_id == end_id:
        # 同起点和终点：空操作也应成立，按约定返回 1 个节点和 0.0 距离。
        return RouteResult(path=[start_id], distance=0.0)

    # 邻接表：每个地点（key）对应它能直接到达的邻居列表。
    # 列表元素是 (neighbor_id, edge_distance)，即下一跳和边权重（里程/距离）。
    # 用“邻接表”而不是全矩阵，能更直接表示稀疏图，查询和遍历更省内存。
    graph: dict[str, list[tuple[str, float]]] = {
        poi_id: [] for poi_id in poi_ids
    }

    for edge in scene.route_graph.edges:
        # 每条边都先按定义方向 from -> to 加入。
        graph[edge.from_id].append((edge.to_id, edge.distance))

        # 无向图中的路线可以双向通行，因此需要补充反向连接。
        if not scene.route_graph.directed:
            # 有向/无向：这是关键业务区分。
            # 无向图会自动补一条反向边；有向图不补。
            graph[edge.to_id].append((edge.from_id, edge.distance))

    # distances：当前已发现的“起点到某点最短路”的最小值估计。
    # 起始时只知起点到自身为 0，其它为正无穷（inf 表示还未到达）。
    distances: dict[str, float] = {
        poi_id: inf for poi_id in poi_ids
    }
    distances[start_id] = 0.0

    # previous：每个点的前驱节点，用于最终把“每个点最优前一步”反向拼成完整路径。
    previous: dict[str, str | None] = {
        poi_id: None for poi_id in poi_ids
    }

    # 优先队列中的元素：(当前已知距离, 节点ID)，小距离优先弹出。
    # 这就是 Dijkstra 的核心：每次拿到局部最小未确定节点。
    priority_queue: list[tuple[float, str]] = [
        (0.0, start_id)
    ]

    while priority_queue:
        # 弹出当前离起点最近的候选节点。
        current_distance, current_id = heappop(priority_queue)

        # 过期队列项：某个节点可能被更短路径更新过多次，旧记录会留在堆里。
        # 如果这次弹出的距离比 best-known 大，说明是过期条目，直接跳过。
        if current_distance > distances[current_id]:
            continue

        # 找到终点后可提前结束：在 Dijkstra 非负边权前提下，
        # 当前弹出的终点距离已经是最短距离，继续跑也不会更优。
        if current_id == end_id:
            break

        for neighbor_id, edge_distance in graph[current_id]:
            # 松弛（relaxation）：
            # 看“起点 -> current -> neighbor”这条新路是否比旧路更短。
            new_distance = current_distance + edge_distance

            if new_distance < distances[neighbor_id]:
                # 更新更短距离，并记录前驱，用于路径回溯。
                distances[neighbor_id] = new_distance
                previous[neighbor_id] = current_id
                heappush(
                    priority_queue,
                    (new_distance, neighbor_id),
                )

    # 若终点始终是 inf，说明在可达集合中没有被更新到 -> 不可达。
    if distances[end_id] == inf:
        return RouteResult(
            path=[],
            distance=None,
            reason="unreachable",
        )

    # 从终点反向根据 previous 回溯到起点，再整体反转得到正向路径。
    # previous 的本质：每个节点都存“最优路径里，它前一个是谁”。
    path: list[str] = []
    path_node: str | None = end_id

    while path_node is not None:
        path.append(path_node)
        path_node = previous[path_node]

    path.reverse()

    return RouteResult(
        path=path,
        distance=distances[end_id],
        reason=None,
    )
