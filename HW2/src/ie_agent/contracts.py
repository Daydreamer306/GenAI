"""跨模块使用的数据对象。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """拒绝没有声明的字段，尽早发现接口使用错误。"""

    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    """发送给对话模型的一条消息。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ModelResult(StrictModel):
    """一次模型调用的结构化结果。"""

    text: str = ""
    provider: str = "deepseek"
    model: str
    status: Literal["success", "error", "fallback"]
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    finish_reason: str | None = None
    error_type: str | None = None
    error: str | None = None


class PlannerResult(StrictModel):
    """模型规划器返回的决策和用量。"""

    route: "RouteDecision | None" = None
    model_result: ModelResult


class PromptSource(StrictModel):
    """准备放入提示词的一条教材证据。"""

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section: str | None = None


class PromptToolEvidence(StrictModel):
    """准备放入提示词的一条确定性工具结果。"""

    tool_name: str = Field(min_length=1)
    result: dict[str, object]
    formula: str | None = None


class PromptContext(StrictModel):
    """PromptBuilder 一次组装所需的全部输入。"""

    query: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    sources: list[PromptSource] = Field(default_factory=list)
    tool_evidence: list[PromptToolEvidence] = Field(default_factory=list)
    skill_instruction: str | None = None


class KnowledgeChunk(StrictModel):
    """一段可以写入索引的教材正文。"""

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    course_tags: list[str] = Field(min_length=1)
    section: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=16)


class Citation(StrictModel):
    """由检索服务生成的一条真实教材引用。"""

    source_id: str
    book_title: str
    section: str
    page: int | None = None
    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)
    backend: Literal["qwen", "tfidf"]


class RetrievedChunk(StrictModel):
    """带引用信息的一段检索结果。"""

    text: str
    citation: Citation
    course_tags: list[str]


class RagResult(StrictModel):
    """一次教材检索的结果。"""

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    backend: Literal["qwen", "tfidf"]
    warnings: list[str] = Field(default_factory=list)


class IngestResult(StrictModel):
    """一次知识库导入的数量摘要。"""

    backend: Literal["qwen", "tfidf"]
    sources: int = Field(ge=0)
    chunks: int = Field(ge=0)
    skipped: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)


class PlannedToolCall(StrictModel):
    """Router 生成的一次待执行工具调用。"""

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, object]
    reason: str = Field(min_length=1)


class RouteDecision(StrictModel):
    """Agent 对本轮任务的可解释决策。"""

    intent: Literal["chat", "knowledge", "calculation", "formula", "unknown"]
    planner: Literal["rules", "json", "tool_calls", "direct_json"] = "rules"
    course_tags: list[str] = Field(default_factory=list)
    need_rag: bool = False
    tool_calls: list[PlannedToolCall] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class AgentRequest(StrictModel):
    """一次完整的 Agent 提问。"""

    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    planner: Literal["hybrid", "rules", "json", "tool_calls"] | None = None
    enabled_skills: list[str] | None = None
    enable_rag: bool = True
    enable_tools: bool = True
    mcp_enabled: bool = True
    enable_review: bool = True
    max_rounds: int | None = Field(default=None, ge=1, le=5)
    structured_answer: bool = False


class ToolResult(StrictModel):
    """确定性工具的统一返回对象。"""

    call_id: str
    tool_name: str
    status: Literal["success", "error"]
    result: dict[str, object] = Field(default_factory=dict)
    formula: str | None = None
    steps: list[str] = Field(default_factory=list)
    artifacts: list["ToolArtifact"] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class ToolArtifact(StrictModel):
    """工具生成并可由界面展示的本地文件。"""

    name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ExecutionStep(StrictModel):
    """Evidence 面板中的一条真实调用记录。"""

    step_id: int = Field(ge=1)
    stage: Literal[
        "input", "planning", "skill", "rag", "tool", "model", "answer", "review", "gate", "stop"
    ]
    name: str = Field(min_length=1)
    status: Literal["success", "error", "skipped"]
    summary: str = Field(min_length=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class ToolSpec(StrictModel):
    """用于界面展示的工具说明。"""

    name: str
    description: str
    input_schema: dict[str, object]


class SkillSpec(StrictModel):
    """仓库内可审查 Skill 的说明。"""

    name: str
    description: str


class AgentResponse(StrictModel):
    """TUI、CLI 和 Web 共用的最终回答。"""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    route: RouteDecision
    model_result: ModelResult
    trace: list[ExecutionStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    outcome: Literal["passed", "failed", "unreviewed"] = "unreviewed"
    stop_reason: str = "single_pass"
    rounds: list["AgentRound"] = Field(default_factory=list)
    messages: list["AgentExchange"] = Field(default_factory=list)
    metrics: "RunMetrics" = Field(default_factory=lambda: RunMetrics())
    run_id: str = ""
    artifact_dir: str = ""


class ReviewVerdict(StrictModel):
    approved: bool
    summary: str = Field(min_length=1)
    issues: list[str]


class AgentRound(StrictModel):
    number: int = Field(ge=1)
    solver_result: ModelResult
    reviewer_result: ModelResult | None = None
    verdict: ReviewVerdict | None = None


class AgentExchange(StrictModel):
    round_number: int = Field(ge=1)
    sender: Literal["solver", "reviewer", "harness"]
    recipient: Literal["solver", "reviewer", "harness"]
    content: str


class RunMetrics(StrictModel):
    execution_mode: Literal["live", "simulated"] = "live"
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)


AgentResponse.model_rebuild()
