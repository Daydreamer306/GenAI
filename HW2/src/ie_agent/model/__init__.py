"""大模型适配器。"""

from ie_agent.config import Settings
from ie_agent.model.deepseek import DeepSeekModel
from ie_agent.model.minimax import MiniMaxModel


def create_model(settings: Settings) -> DeepSeekModel | MiniMaxModel:
    """按照配置显式选用模型，不进行跨提供商回退。"""

    if settings.model_provider == "minimax":
        return MiniMaxModel(settings)
    return DeepSeekModel(settings)


__all__ = ["DeepSeekModel", "MiniMaxModel", "create_model"]
