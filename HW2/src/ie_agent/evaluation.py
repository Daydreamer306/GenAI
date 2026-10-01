"""小规模教材检索评估。"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from ie_agent.contracts import StrictModel
from ie_agent.knowledge.service import KnowledgeService


class EvaluationQuestion(StrictModel):
    """一条人工标注的课程问题。"""

    question: str = Field(min_length=1)
    course: str = Field(min_length=1)
    expected_source_ids: list[str] = Field(min_length=1)


class EvaluationItem(StrictModel):
    """单个问题的实际排名结果。"""

    question: str
    expected_source_ids: list[str]
    retrieved_source_ids: list[str]
    first_relevant_rank: int | None
    latency_ms: float = Field(ge=0.0)


class EvaluationReport(StrictModel):
    """Recall@K、MRR 和平均耗时。"""

    backend: str
    top_k: int = Field(ge=1)
    questions: int = Field(ge=0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    average_latency_ms: float = Field(ge=0.0)
    items: list[EvaluationItem]


class RetrievalEvaluator:
    """使用同一问题集比较 Qwen 与 TF-IDF。"""

    @staticmethod
    def load_questions(path: Path) -> list[EvaluationQuestion]:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [EvaluationQuestion.model_validate(item) for item in raw.get("questions", [])]

    def run(
        self,
        service: KnowledgeService,
        questions: list[EvaluationQuestion],
        top_k: int = 5,
    ) -> EvaluationReport:
        items: list[EvaluationItem] = []
        for question in questions:
            result = service.search(
                question.question,
                course_tags=[question.course],
                top_k=top_k,
            )
            source_ids = [chunk.citation.source_id for chunk in result.chunks]
            expected = set(question.expected_source_ids)
            rank = next(
                (
                    index
                    for index, source_id in enumerate(source_ids, start=1)
                    if source_id in expected
                ),
                None,
            )
            items.append(
                EvaluationItem(
                    question=question.question,
                    expected_source_ids=question.expected_source_ids,
                    retrieved_source_ids=source_ids,
                    first_relevant_rank=rank,
                    latency_ms=result.latency_ms,
                )
            )

        count = len(items)
        hits = sum(item.first_relevant_rank is not None for item in items)
        reciprocal_ranks = [
            1.0 / item.first_relevant_rank if item.first_relevant_rank is not None else 0.0
            for item in items
        ]
        return EvaluationReport(
            backend=service.backend,
            top_k=top_k,
            questions=count,
            recall_at_k=hits / count if count else 0.0,
            mrr=sum(reciprocal_ranks) / count if count else 0.0,
            average_latency_ms=sum(item.latency_ms for item in items) / count if count else 0.0,
            items=items,
        )
