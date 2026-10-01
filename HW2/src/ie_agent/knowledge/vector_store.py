"""使用本地 Chroma 保存 Qwen 教材向量。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from ie_agent.contracts import Citation, KnowledgeChunk, RetrievedChunk
from ie_agent.ports import EmbeddingProvider


def _read_collection(client: Any, collection_name: str) -> Any:
    """读取正式集合；如果切换中断，则回退到备份集合。"""

    from chromadb.errors import NotFoundError

    get_collection = client.get_collection
    try:
        return get_collection(collection_name, embedding_function=None)
    except NotFoundError:
        return get_collection(f"{collection_name}_backup", embedding_function=None)


def chroma_index_ready(path: Path, collection_name: str = "ie_agent_qwen") -> bool:
    """只有完整 collection 存在且包含数据时才认为索引就绪。"""

    if not path.is_dir() or not (path / "chroma.sqlite3").is_file():
        return False
    try:
        import chromadb
        from chromadb.errors import ChromaError
    except ImportError:
        return False
    try:
        client = chromadb.PersistentClient(path=str(path))
        return _read_collection(client, collection_name).count() > 0
    except (ChromaError, OSError, sqlite3.Error, ValueError):
        return False


class ChromaVectorStore:
    """显式传入向量的 Chroma 持久化存储。"""

    def __init__(
        self,
        path: Path,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "ie_agent_qwen",
        batch_size: int = 64,
    ) -> None:
        self.path = path
        self.embedding_provider = embedding_provider
        self.collection_name = collection_name
        self.batch_size = batch_size

    def rebuild(self, chunks: list[KnowledgeChunk]) -> int:
        """在临时集合中建库，成功后再替换旧集合。"""

        import chromadb
        from chromadb.errors import NotFoundError

        self.path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.path))
        building_name = f"{self.collection_name}_building"
        backup_name = f"{self.collection_name}_backup"

        # 上次若在两次改名之间退出，先恢复可检索的正式名称。
        try:
            client.get_collection(self.collection_name, embedding_function=None)
        except NotFoundError:
            try:
                backup = client.get_collection(backup_name, embedding_function=None)
                backup.modify(name=self.collection_name)
            except NotFoundError:
                pass
        else:
            try:
                client.delete_collection(backup_name)
            except NotFoundError:
                pass

        try:
            client.delete_collection(building_name)
        except NotFoundError:
            pass
        collection = client.get_or_create_collection(
            building_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

        try:
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                texts = [chunk.text for chunk in batch]
                embeddings = self.embedding_provider.embed_documents(texts)
                collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=[self._metadata(chunk) for chunk in batch],
                )

            stored = collection.count()
            if stored != len(chunks):
                raise RuntimeError(f"向量索引数量校验失败：{stored}/{len(chunks)}")

            # 先把旧集合改为备份，新集合提升失败时可立即回滚。
            try:
                previous = client.get_collection(
                    self.collection_name,
                    embedding_function=None,
                )
            except NotFoundError:
                previous = None
            if previous is not None:
                previous.modify(name=backup_name)

            try:
                collection.modify(name=self.collection_name)
            except Exception:
                if previous is not None:
                    backup = client.get_collection(backup_name, embedding_function=None)
                    backup.modify(name=self.collection_name)
                raise

            if previous is not None:
                try:
                    client.delete_collection(backup_name)
                except NotFoundError:
                    pass
            return stored
        except Exception:
            try:
                client.delete_collection(building_name)
            except NotFoundError:
                pass
            raise

    def search(
        self,
        query: str,
        course_tags: list[str],
        top_k: int,
        min_score: float = 0.20,
    ) -> list[RetrievedChunk]:
        """使用余弦距离查询，并返回统一的 0--1 分数。"""

        import chromadb
        from chromadb.errors import NotFoundError

        if not self.path.is_dir():
            raise FileNotFoundError(f"Qwen Chroma 索引不存在：{self.path}")
        client = chromadb.PersistentClient(path=str(self.path))
        try:
            collection = _read_collection(client, self.collection_name)
        except NotFoundError as exc:
            raise FileNotFoundError("Qwen Chroma 集合尚未建立") from exc
        count = collection.count()
        if count == 0:
            return []

        where = {"course": {"$in": course_tags}} if course_tags else None
        result = collection.query(
            query_embeddings=[self.embedding_provider.embed_query(query)],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            if text is None or metadata is None or distance is None:
                continue
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            if score < min_score:
                continue
            page = metadata.get("page")
            chunks.append(
                RetrievedChunk(
                    text=text,
                    citation=Citation(
                        source_id=str(metadata["source_id"]),
                        book_title=str(metadata["book_title"]),
                        section=str(metadata["section"]),
                        page=int(page) if page is not None else None,
                        chunk_id=str(metadata["chunk_id"]),
                        score=score,
                        backend="qwen",
                    ),
                    course_tags=[str(metadata["course"])],
                )
            )
        return chunks

    @staticmethod
    def _metadata(chunk: KnowledgeChunk) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "book_title": chunk.book_title,
            "course": chunk.course_tags[0],
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
        }
        if chunk.page is not None:
            metadata["page"] = chunk.page
        return metadata
