"""Configuration for an OpenAI-compatible chat provider."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from guide_agent.app_settings import ProductModelConfig


class ModelConfigurationError(ValueError):
    """The provider configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    """Settings consumed by LangChain's ``ChatOpenAI`` integration."""

    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    api_format: str = "responses"


def load_model_settings(
    environ: Mapping[str, str] | None = None,
    defaults: ProductModelConfig | None = None,
) -> ModelSettings:
    """Load one OpenAI-compatible provider from environment variables.

    ``OPENAI_BASE_URL`` is the API root (for example
    ``https://provider.example/v1``), never an endpoint suffix.
    """

    source = os.environ if environ is None else environ
    model = source.get("OPENAI_MODEL", defaults.name if defaults else "").strip()
    api_key = source.get("OPENAI_API_KEY", "").strip()
    base_url = source.get(
        "OPENAI_BASE_URL", defaults.base_url if defaults else ""
    ).strip() or None
    api_format = source.get(
        "OPENAI_API_FORMAT", defaults.api_format if defaults else "responses"
    ).strip() or (defaults.api_format if defaults else "responses")
    if not model:
        raise ModelConfigurationError("OPENAI_MODEL is required")
    if not api_key:
        raise ModelConfigurationError("OPENAI_API_KEY is required")
    if api_format not in {"responses", "chat_completions"}:
        raise ModelConfigurationError(
            "OPENAI_API_FORMAT must be 'responses' or 'chat_completions'"
        )
    return ModelSettings(model=model, api_key=api_key, base_url=base_url, api_format=api_format)
