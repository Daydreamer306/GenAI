"""项目配置。

所有密钥只从环境变量或本机 ``.env`` 读取，不写入日志和业务结果。
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """IE-Agent 的运行配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    model_provider: Literal["minimax", "deepseek"] = Field(
        default="minimax",
        validation_alias="IE_AGENT_MODEL_PROVIDER",
    )
    minimax_api_key: SecretStr | None = Field(default=None, validation_alias="MINIMAX_API_KEY")
    minimax_base_url: str = Field(
        default="https://api.minimax.cn/v1",
        validation_alias="MINIMAX_BASE_URL",
    )
    minimax_model: str = Field(default="MiniMax-M2.7", validation_alias="MINIMAX_MODEL")
    minimax_max_tokens: int = Field(
        default=8192,
        ge=1,
        le=32768,
        validation_alias="MINIMAX_MAX_TOKENS",
    )
    minimax_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
        validation_alias="MINIMAX_TIMEOUT_SECONDS",
    )

    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias="DEEPSEEK_BASE_URL",
    )
    deepseek_model: str = Field(
        default="deepseek-v4-pro",
        validation_alias="DEEPSEEK_MODEL",
    )
    deepseek_thinking: Literal["enabled", "disabled"] = Field(
        default="enabled",
        validation_alias="DEEPSEEK_THINKING",
    )
    deepseek_reasoning_effort: Literal["high", "max"] = Field(
        default="high",
        validation_alias="DEEPSEEK_REASONING_EFFORT",
    )
    deepseek_max_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        validation_alias="DEEPSEEK_MAX_TOKENS",
    )
    deepseek_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=600,
        validation_alias="DEEPSEEK_TIMEOUT_SECONDS",
    )

    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        validation_alias="IE_AGENT_EMBEDDING_MODEL",
    )
    knowledge_dir: Path = Field(
        default=Path("knowledge/demo"),
        validation_alias="IE_AGENT_KNOWLEDGE_DIR",
    )
    rag_backend: Literal["qwen", "tfidf"] = Field(
        default="tfidf",
        validation_alias="IE_AGENT_RAG_BACKEND",
    )
    top_k: int = Field(default=5, ge=1, le=20, validation_alias="IE_AGENT_TOP_K")
    device: str = Field(default="auto", validation_alias="IE_AGENT_DEVICE")
    planner: Literal["hybrid", "rules", "json", "tool_calls"] = Field(
        default="hybrid",
        validation_alias="IE_AGENT_PLANNER",
    )
    max_rounds: int = Field(default=3, ge=1, le=5, validation_alias="IE_AGENT_MAX_ROUNDS")
    runs_dir: Path = Field(default=Path("runs"), validation_alias="IE_AGENT_RUNS_DIR")

    @property
    def active_model(self) -> str:
        return getattr(self, f"{self.model_provider}_model")

    @property
    def has_model_key(self) -> bool:
        key = getattr(self, f"{self.model_provider}_api_key")
        return bool(key and key.get_secret_value().strip())

    @property
    def has_deepseek_key(self) -> bool:
        """只返回密钥是否存在，不返回密钥内容。"""

        return bool(self.deepseek_api_key and self.deepseek_api_key.get_secret_value().strip())
