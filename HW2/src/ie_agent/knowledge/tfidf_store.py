"""字符 n-gram TF-IDF 检索基线。"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ie_agent.contracts import Citation, KnowledgeChunk, RetrievedChunk


class TfidfStore:
    """保存和查询轻量的词面匹配索引。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._payload: dict[str, Any] | None = None

    def rebuild(self, chunks: list[KnowledgeChunk]) -> int:
        """使用与 Qwen 完全相同的 chunk 建立对照索引。"""

        if not chunks:
            raise ValueError("没有可导入的教材片段")
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        payload = {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "chunks": [chunk.model_dump() for chunk in chunks],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        self._payload = payload
        return len(chunks)

    def search(
        self,
        query: str,
        course_tags: list[str],
        top_k: int,
        min_score: float = 0.01,
    ) -> list[RetrievedChunk]:
        """计算稀疏余弦相似度，并稳定返回最高分片段。"""

        payload = self._load()
        vectorizer: TfidfVectorizer = payload["vectorizer"]
        matrix = payload["matrix"]
        chunks = [KnowledgeChunk.model_validate(item) for item in payload["chunks"]]
        query_vector = vectorizer.transform([query])
        scores = (matrix @ query_vector.T).toarray().ravel()

        allowed = set(course_tags)
        if allowed:
            for index, chunk in enumerate(chunks):
                if not allowed.intersection(chunk.course_tags):
                    scores[index] = -1.0
        ordered = np.argsort(-scores, kind="stable")[:top_k]

        results: list[RetrievedChunk] = []
        for index in ordered:
            score = float(scores[index])
            if score < min_score:
                continue
            chunk = chunks[int(index)]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    citation=Citation(
                        source_id=chunk.source_id,
                        book_title=chunk.book_title,
                        section=chunk.section,
                        page=chunk.page,
                        chunk_id=chunk.chunk_id,
                        score=max(0.0, min(1.0, score)),
                        backend="tfidf",
                    ),
                    course_tags=chunk.course_tags,
                )
            )
        return results

    def _load(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        if not self.path.is_file():
            raise FileNotFoundError(f"TF-IDF 索引不存在：{self.path}")
        with self.path.open("rb") as file:
            payload = pickle.load(file)  # noqa: S301 - 只加载本项目在本机生成的忽略文件
        if not isinstance(payload, dict):
            raise ValueError("TF-IDF 索引格式错误")
        self._payload = payload
        return payload
