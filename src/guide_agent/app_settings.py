"""Server-owned, non-secret product configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ProductModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_format: Literal["responses", "chat_completions"] = "chat_completions"


class ProductSceneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    minimum_score: float = Field(default=0.50, ge=0, le=1)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    request_timeout_seconds: float = Field(default=15, gt=0)
    model_concurrency: int = Field(default=6, ge=1)
    queue_size: int = Field(default=14, ge=0)
    requests_per_minute: int = Field(default=30, ge=1)


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_turns: int = Field(default=20, ge=1, le=100)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ProductModelConfig
    scene: ProductSceneConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)


class AppConfigurationError(ValueError):
    """The YAML product configuration is missing or invalid."""


def resolve_config_path(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("GUIDE_CONFIG_PATH", "").strip()
    candidate = Path(configured) if configured else DEFAULT_CONFIG_PATH
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_app_settings(path: str | Path | None = None) -> AppSettings:
    config_path = resolve_config_path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return AppSettings.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise AppConfigurationError(
            f"failed to load product configuration {config_path}: {error}"
        ) from error


def resolve_product_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
