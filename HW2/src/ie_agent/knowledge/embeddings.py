"""Qwen3-Embedding-0.6B 本地向量适配器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


class QwenEmbeddingProvider:
    """延迟加载 Qwen 模型，避免查看帮助时自动下载权重。"""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        device: str = "auto",
        batch_size: int = 4,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.model_factory = model_factory
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """教材正文不添加 query instruction。"""

        if not texts:
            return []
        vectors = self._load_model().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        return self._to_lists(vectors)

    def embed_query(self, query: str) -> list[float]:
        """按官方模型卡使用内置 query prompt。"""

        vectors = self._load_model().encode(
            [query],
            prompt_name="query",
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self._to_lists(vectors)[0]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        factory = self.model_factory
        if factory is None:
            from sentence_transformers import SentenceTransformer

            factory = SentenceTransformer
        model_path = self._cached_model_path()
        self._model = factory(
            str(model_path) if model_path else self.model_name,
            cache_folder=str(self.cache_dir),
            device=self._resolve_device(),
        )
        if hasattr(self._model, "tokenizer"):
            self._model.tokenizer.padding_side = "left"
        self._model.max_seq_length = 2048
        return self._model

    def _cached_model_path(self) -> Path | None:
        """优先使用已经下载完成的 Hugging Face 快照。"""

        if (self.cache_dir / "config.json").is_file():
            return self.cache_dir
        repo_name = f"models--{self.model_name.replace('/', '--')}"
        repo_dir = self.cache_dir / repo_name
        main_ref = repo_dir / "refs" / "main"
        if not main_ref.is_file():
            return None
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = repo_dir / "snapshots" / revision
        if (snapshot / "config.json").is_file():
            return snapshot
        return None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _to_lists(vectors: Any) -> list[list[float]]:
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [[float(value) for value in vector] for vector in vectors]
