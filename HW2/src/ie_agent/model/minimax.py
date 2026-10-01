"""MiniMax OpenAI 兼容接口。"""

from ie_agent.model.compatible import OpenAICompatibleModel


class MiniMaxModel(OpenAICompatibleModel):
    """仅使用 MiniMax 配置；缺少 key 或调用失败不会回退 DeepSeek。"""

    provider = "minimax"
