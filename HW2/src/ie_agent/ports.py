"""业务模块依赖的接口。"""

from collections.abc import Sequence
from typing import Protocol

from ie_agent.contracts import ChatMessage, ModelResult


class ChatModel(Protocol):
    """所有对话模型适配器都需要实现的接口。"""

    def generate(self, messages: Sequence[ChatMessage]) -> ModelResult:
        """根据消息列表生成一次回答。"""

        ...


class EmbeddingProvider(Protocol):
    """向量模型需要提供的最小接口。"""

    @property
    def model_name(self) -> str:
        """返回实际使用的模型名称。"""

        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """为教材片段生成归一化向量。"""

        ...

    def embed_query(self, query: str) -> list[float]:
        """使用检索指令为问题生成归一化向量。"""

        ...
