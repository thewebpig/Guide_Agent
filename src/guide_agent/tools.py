from dataclasses import dataclass

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from guide_agent.retrieval import (
    KnowledgeIndex,
    KnowledgeSearchError,
    search_knowledge,
)
from guide_agent.service import GuideService


class LookupPoiArguments(BaseModel):
    """lookup_poi工具的参数。"""

    model_config = ConfigDict(extra="forbid")

    poi_id: str = Field(min_length=1)

    @field_validator("poi_id")
    @classmethod
    def poi_id_not_blank(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("poi_id must not be blank")

        return cleaned_value


class PlanRouteArguments(BaseModel):
    """plan_route工具的参数。"""

    model_config = ConfigDict(extra="forbid")

    start_id: str = Field(min_length=1)
    end_id: str = Field(min_length=1)

    @field_validator("start_id", "end_id")
    @classmethod
    def poi_ids_not_blank(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("POI ID must not be blank")

        return cleaned_value


class SearchKnowledgeArguments(BaseModel):
    """search_knowledge工具的参数。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(
        # default=5,
        gt=0,
        strict=True,
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("query must not be blank")

        return cleaned_value


@dataclass(frozen=True)
class ToolDefinition:
    """提供给模型查看的Tool定义。"""

    name: str
    description: str
    arguments_model: type[BaseModel]

    def to_schema(self) -> dict[str, object]:
        """转换成模型可读取的名称、说明和参数Schema。"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": (
                self.arguments_model.model_json_schema()
            ),
        }


LOOKUP_POI_TOOL = ToolDefinition(
    name="lookup_poi",
    description=(
        "根据精确POI ID查询地点的名称、描述、"
        "别名、标签和位置信息。"
    ),
    arguments_model=LookupPoiArguments,
)

PLAN_ROUTE_TOOL = ToolDefinition(
    name="plan_route",
    description=(
        "根据起点和终点POI ID规划最短路线，"
        "并返回路径、距离或不可达原因。"
    ),
    arguments_model=PlanRouteArguments,
)

SEARCH_KNOWLEDGE_TOOL = ToolDefinition(
    name="search_knowledge",
    description=(
        "从访客指南中检索开放时间、访问规则、"
        "地点服务和安全要求等知识证据。"
    ),
    arguments_model=SearchKnowledgeArguments,
)

TOOL_DEFINITIONS = (
    SEARCH_KNOWLEDGE_TOOL,
    LOOKUP_POI_TOOL,
    PLAN_ROUTE_TOOL,
)


def get_tool_schemas() -> list[dict[str, object]]:
    """返回全部注册Tool的模型可读Schema。"""

    return [
        tool.to_schema()
        for tool in TOOL_DEFINITIONS
    ]


class ToolRegistry:
    """校验并执行允许列表中的Tool。"""

    def __init__(
        self,
        service: GuideService,
        knowledge_index: KnowledgeIndex,
    ) -> None:
        self._service = service
        self._knowledge_index = knowledge_index
        self._definitions = {
            tool.name: tool
            for tool in TOOL_DEFINITIONS
        }

    def schemas(self) -> list[dict[str, object]]:
        """返回当前注册表中的全部Tool Schema。"""

        return [
            definition.to_schema()
            for definition in self._definitions.values()
        ]

    def execute(
        self,
        tool_name: object,
        raw_arguments: object,
    ) -> dict[str, object]:
        """校验并执行一次Tool请求。"""

        if (
            not isinstance(tool_name, str)
            or tool_name not in self._definitions
        ):
            return {
                "status": "unknown_tool",
                "tool_name": str(tool_name),
                "data": None,
                "error": {
                    "type": "unknown_tool",
                    "message": "tool is not registered",
                },
            }

        definition = self._definitions[tool_name]

        try:
            arguments = (
                definition.arguments_model.model_validate(
                    raw_arguments
                )
            )
        except ValidationError as error:
            return {
                "status": "invalid_arguments",
                "tool_name": tool_name,
                "data": None,
                "error": {
                    "type": "validation_error",
                    "details": error.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    ),
                },
            }

        try:
            if isinstance(
                arguments,
                LookupPoiArguments,
            ):
                data = self._service.lookup_poi(
                    arguments.poi_id
                )
            elif isinstance(
                arguments,
                PlanRouteArguments,
            ):
                data = self._service.plan_route(
                    arguments.start_id,
                    arguments.end_id,
                )
            elif isinstance(
                arguments,
                SearchKnowledgeArguments,
            ):
                data = search_knowledge(
                    self._knowledge_index,
                    arguments.query,
                    arguments.top_k,
                )
            else:
                raise RuntimeError(
                    "registered Tool has unsupported "
                    "arguments model"
                )
        except KnowledgeSearchError as error:
            return {
                "status": "tool_error",
                "tool_name": tool_name,
                "data": None,
                "error": {
                    "type": "knowledge_search_error",
                    "message": str(error),
                },
            }

        return {
            "status": "ok",
            "tool_name": tool_name,
            "data": data,
            "error": None,
        }
