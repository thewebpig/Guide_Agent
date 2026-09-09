import pytest

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
