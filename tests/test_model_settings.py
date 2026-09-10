import pytest
from pydantic import SecretStr

import guide_agent.app_settings as app_settings_module
from guide_agent.app_settings import (
    AppConfigurationError,
    ProductModelConfig,
    load_app_settings,
)
from guide_agent.model_settings import ModelConfigurationError, load_model_settings


def test_loads_chat_completions_provider() -> None:
    settings = load_model_settings({"OPENAI_MODEL": "MiniCPM5-2B", "OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "https://provider.example/v1", "OPENAI_API_FORMAT": "chat_completions"})
    assert settings.model == "MiniCPM5-2B"
    assert settings.base_url == "https://provider.example/v1"
    assert settings.api_format == "chat_completions"


@pytest.mark.parametrize("env", [{}, {"OPENAI_MODEL": "x"}, {"OPENAI_MODEL": "x", "OPENAI_API_KEY": "y", "OPENAI_API_FORMAT": "other"}])
def test_rejects_incomplete_or_unknown_config(env: dict[str, str]) -> None:
    with pytest.raises(ModelConfigurationError):
        load_model_settings(env)


def test_start_environment_key_overrides_configuration_file_key() -> None:
    defaults = ProductModelConfig(
        name="MiniCPM5-2B",
        base_url="https://developer.amd.com.cn/radeon/api/v1",
        api_format="chat_completions",
        api_key=SecretStr("file-key"),
    )
    from_file = load_model_settings({}, defaults)
    overridden = load_model_settings({"OPENAI_API_KEY": "start-key"}, defaults)
    assert from_file.api_key == "file-key"
    assert overridden.api_key == "start-key"


def test_private_local_yaml_fragment_overlays_tracked_configuration(
    tmp_path, monkeypatch
) -> None:
    private_config = tmp_path / "config.local.yaml"
    private_config.write_text("model:\n  api_key: private-file-key\n", encoding="utf-8")
    monkeypatch.setattr(app_settings_module, "LOCAL_CONFIG_PATH", private_config)

    settings = load_app_settings()

    assert settings.model.name == "MiniCPM5-2B"
    assert settings.model.api_key is not None
    assert settings.model.api_key.get_secret_value() == "private-file-key"


def test_invalid_yaml_error_does_not_echo_secret_excerpt(tmp_path) -> None:
    invalid = tmp_path / "private.yaml"
    invalid.write_text(
        "model:MiniCPM5-2B\n  api_key: secret-must-not-appear\n",
        encoding="utf-8",
    )

    with pytest.raises(AppConfigurationError) as error_info:
        load_app_settings(invalid)

    message = str(error_info.value)
    assert "line 2, column 10" in message
    assert "secret-must-not-appear" not in message
