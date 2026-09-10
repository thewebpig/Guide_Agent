"""Validate startup configuration without printing secret values."""

from guide_agent.app_settings import AppConfigurationError, load_app_settings
from guide_agent.model_settings import ModelConfigurationError, load_model_settings


def main() -> int:
    try:
        product = load_app_settings()
        load_model_settings(defaults=product.model)
    except (AppConfigurationError, ModelConfigurationError) as error:
        print(f"配置错误：{error}")
        return 1
    print("配置检查通过：模型、接口地址和 API Key 已配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
