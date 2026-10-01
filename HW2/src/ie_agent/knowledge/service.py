"""统一知识库导入和检索入口。"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from ie_agent.config import Settings
from ie_agent.contracts import IngestResult, RagResult
from ie_agent.knowledge.chunker import MarkdownChunker
from ie_agent.knowledge.embeddings import QwenEmbeddingProvider
from ie_agent.ports import EmbeddingProvider

if TYPE_CHECKING:
    from ie_agent.knowledge.tfidf_store import TfidfStore
    from ie_agent.knowledge.vector_store import ChromaVectorStore


class KnowledgeService:
    """让命令行、Agent、TUI 和 Web 复用同一检索实现。"""

    def __init__(
        self,
        settings: Settings,
        backend: str | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chunker: MarkdownChunker | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or settings.rag_backend
        if self.backend not in {"qwen", "tfidf"}:
            raise ValueError(f"不支持的检索后端：{self.backend}")
        self.chunker = chunker or MarkdownChunker()
        self.embedding_provider = embedding_provider

    def ingest(self, clean_dir: Path) -> IngestResult:
        """读取全部清洗教材并重建所选索引。"""

        started = perf_counter()
        chunks = self.chunker.chunk_directory(clean_dir)
        if not chunks:
            raise ValueError(f"没有在 {clean_dir} 中找到可导入的教材")
        if self.backend == "qwen":
            stored = self._qwen_store().rebuild(chunks)
        else:
            stored = self._tfidf_store().rebuild(chunks)
        sources = len({chunk.source_id for chunk in chunks})
        return IngestResult(
            backend=self.backend,
            sources=sources,
            chunks=stored,
            latency_ms=(perf_counter() - started) * 1000,
        )

    def search(
        self,
        query: str,
        course_tags: list[str] | None = None,
        top_k: int | None = None,
    ) -> RagResult:
        """使用明确后端检索，不在失败时偷偷切换算法。"""

        started = perf_counter()
        course_tags = course_tags or []
        top_k = top_k or self.settings.top_k
        if self.backend == "qwen":
            chunks = self._qwen_store().search(query, course_tags, top_k)
        else:
            chunks = self._tfidf_store().search(query, course_tags, top_k)
        return RagResult(
            chunks=chunks,
            latency_ms=(perf_counter() - started) * 1000,
            backend=self.backend,
        )

    def _qwen_store(self) -> ChromaVectorStore:
        from ie_agent.knowledge.vector_store import ChromaVectorStore

        provider = self.embedding_provider
        if provider is None:
            provider = QwenEmbeddingProvider(
                model_name=self.settings.embedding_model,
                cache_dir=self.settings.knowledge_dir / "models" / "qwen3-embedding-0.6b",
                device=self.settings.device,
            )
            self.embedding_provider = provider
        return ChromaVectorStore(
            self.settings.knowledge_dir / "index" / "qwen_chroma",
            provider,
        )

    def _tfidf_store(self) -> TfidfStore:
        from ie_agent.knowledge.tfidf_store import TfidfStore

        return TfidfStore(self.settings.knowledge_dir / "index" / "tfidf" / "index.pkl")
