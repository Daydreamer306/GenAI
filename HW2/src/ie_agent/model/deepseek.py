"""保留原有 DeepSeek 适配器入口。"""

from ie_agent.model.compatible import OpenAICompatibleModel


class DeepSeekModel(OpenAICompatibleModel):
    """仅使用 DeepSeek 配置。"""

    provider = "deepseek"
