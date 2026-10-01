"""有界的 Solver → Reviewer → gate 协作循环。"""

from dataclasses import dataclass, field

from ie_agent.agent.reviewer import ReviewerAgent, evidence_gate
from ie_agent.agent.solver import SolverAgent
from ie_agent.contracts import AgentExchange, AgentRound, ExecutionStep, ModelResult, PromptContext


@dataclass
class HarnessResult:
    answer: str = ""
    outcome: str = "failed"
    stop_reason: str = "solver_error"
    rounds: list[AgentRound] = field(default_factory=list)
    messages: list[AgentExchange] = field(default_factory=list)
    trace: list[ExecutionStep] = field(default_factory=list)


class ReviewHarness:
    def __init__(self, solver: SolverAgent, reviewer: ReviewerAgent, max_rounds: int) -> None:
        self.solver = solver
        self.reviewer = reviewer
        self.max_rounds = max_rounds
        if not 1 <= max_rounds <= 5:
            raise ValueError("max_rounds 必须在 1–5 之间")

    def run(self, context: PromptContext, *, enable_review: bool = True) -> HarnessResult:
        result = HarnessResult()
        feedback = None
        draft = ""
        for number in range(1, self.max_rounds + 1):
            generated = self.solver.solve(context, draft, feedback)
            record = AgentRound(number=number, solver_result=generated)
            result.rounds.append(record)
            self._step(
                result,
                "model",
                f"Solver 第 {number} 轮",
                generated,
                "生成草稿" if generated.status == "success" else "解题模型失败",
                {"round": number, "agent": "solver"},
            )
            if generated.status != "success":
                result.stop_reason = "solver_error"
                break
            draft = generated.text
            result.answer = draft
            result.messages.append(
                AgentExchange(
                    round_number=number,
                    sender="solver",
                    recipient="reviewer" if enable_review else "harness",
                    content=draft,
                )
            )
            if not enable_review:
                result.outcome, result.stop_reason = "unreviewed", "review_disabled"
                break
            reviewed, verdict = self.reviewer.review(context, draft)
            record.reviewer_result = reviewed
            self._step(
                result,
                "review",
                f"Reviewer 第 {number} 轮",
                reviewed,
                "审核已完成" if verdict is not None else "审核不可用，禁止放行",
                {"round": number, "agent": "reviewer"},
            )
            if verdict is None:
                result.stop_reason = "reviewer_error"
                break
            feedback = evidence_gate(context, draft, verdict)
            record.verdict = feedback
            result.messages.append(
                AgentExchange(
                    round_number=number,
                    sender="reviewer",
                    recipient="harness" if feedback.approved else "solver",
                    content=feedback.model_dump_json(),
                )
            )
            result.trace.append(
                ExecutionStep(
                    step_id=len(result.trace) + 1,
                    stage="gate",
                    name=f"审核门 第 {number} 轮",
                    status="success" if feedback.approved else "error",
                    summary=feedback.summary,
                    metadata={"round": number, **feedback.model_dump()},
                )
            )
            if feedback.approved:
                result.outcome, result.stop_reason = "passed", "review_approved"
                break
            result.stop_reason = "max_rounds_exceeded"
        result.trace.append(
            ExecutionStep(
                step_id=len(result.trace) + 1,
                stage="stop",
                name="Harness 停止",
                status="error" if result.outcome == "failed" else "success",
                summary=result.stop_reason,
                metadata={
                    "outcome": result.outcome,
                    "rounds": len(result.rounds),
                    "max_rounds": self.max_rounds,
                },
            )
        )
        return result

    @staticmethod
    def _step(
        result: HarnessResult,
        stage: str,
        name: str,
        model: ModelResult,
        summary: str,
        metadata: dict,
    ) -> None:
        result.trace.append(
            ExecutionStep(
                step_id=len(result.trace) + 1,
                stage=stage,
                name=name,
                status="success" if model.status == "success" else "error",
                summary=summary,
                latency_ms=model.latency_ms,
                metadata={
                    **metadata,
                    "provider": model.provider,
                    "model": model.model,
                    "input_tokens": model.input_tokens,
                    "output_tokens": model.output_tokens,
                    "error_type": model.error_type,
                },
            )
        )
