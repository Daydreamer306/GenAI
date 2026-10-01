"""把路由、工具、RAG、Skill 和模型组织为一次回答。"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from ie_agent.agent.harness import ReviewHarness
from ie_agent.agent.planner import AgentPlanner
from ie_agent.agent.prompt import PromptBuilder
from ie_agent.agent.reviewer import ReviewerAgent
from ie_agent.agent.router import IntentRouter
from ie_agent.agent.solver import SolverAgent
from ie_agent.config import Settings
from ie_agent.contracts import (
    AgentRequest,
    AgentResponse,
    ExecutionStep,
    ModelResult,
    PromptContext,
    PromptSource,
    PromptToolEvidence,
    RunMetrics,
    ToolResult,
)
from ie_agent.knowledge import KnowledgeService
from ie_agent.mcp_server import InProcessMCPTools
from ie_agent.model import create_model
from ie_agent.ports import ChatModel
from ie_agent.run_artifacts import RunArtifactWriter
from ie_agent.skills import SkillLoader
from ie_agent.tools import ToolRegistry


class AgentOrchestrator:
    """按路由结果组织检索、工具和模型调用。"""

    def __init__(
        self,
        settings: Settings,
        model: ChatModel | None = None,
        knowledge: KnowledgeService | None = None,
        router: IntentRouter | None = None,
        tools: ToolRegistry | None = None,
        skills: SkillLoader | None = None,
        prompt_builder: PromptBuilder | None = None,
        planner: AgentPlanner | None = None,
        reviewer_model: ChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.model = model or create_model(settings)
        self.knowledge = knowledge or KnowledgeService(settings)
        self.router = router or IntentRouter()
        self.tools = tools or ToolRegistry(settings.knowledge_dir / "artifacts")
        self.skills = skills or SkillLoader()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.solver = SolverAgent(self.model, self.prompt_builder)
        self.reviewer = ReviewerAgent(
            reviewer_model or (self.model if model else create_model(settings))
        )
        self.artifact_writer = RunArtifactWriter(settings)
        self.planner = planner or AgentPlanner(
            settings,
            self.model,
            self.router,
            self.tools,
            self.skills,
        )
        self.mcp_tools = InProcessMCPTools(
            settings,
            self.tools,
            self.knowledge,
            self.skills,
        )

    def answer(self, request: AgentRequest) -> AgentResponse:
        """尽量保留已经成功的本地结果，并明确报告降级。"""

        trace: list[ExecutionStep] = []
        started = perf_counter()
        input_summary = (
            "收到 JSON 工具请求。"
            if request.query.lstrip().startswith("{")
            else "收到自然语言问题。"
        )
        self._append_step(
            trace,
            "input",
            "输入解析",
            "success",
            input_summary,
            metadata={
                "planner": request.planner or self.settings.planner,
                "rag_enabled": request.enable_rag,
                "tools_enabled": request.enable_tools,
                "mcp_enabled": request.mcp_enabled,
                "skill_limit": request.enabled_skills,
            },
        )

        outcome = self.planner.plan(
            request.query,
            planner=request.planner,
            enabled_skills=request.enabled_skills,
            enable_tools=request.enable_tools,
        )
        route = outcome.route
        warnings: list[str] = []
        if outcome.warning:
            warnings.append(outcome.warning)
        planning_metadata: dict[str, object] = {
            "planner": route.planner,
            "intent": route.intent,
            "need_rag": route.need_rag,
            "skills": route.skill_names,
            "tools": [call.tool_name for call in route.tool_calls],
        }
        if outcome.model_result is not None:
            planning_metadata.update(
                {
                    "input_tokens": outcome.model_result.input_tokens,
                    "output_tokens": outcome.model_result.output_tokens,
                    "model": outcome.model_result.model,
                }
            )
        self._append_step(
            trace,
            "planning",
            f"{route.planner} planner",
            "error" if outcome.warning and outcome.model_result else "success",
            route.reason,
            outcome.latency_ms,
            planning_metadata,
        )

        skill_parts: list[str] = []
        if route.skill_names:
            for name in route.skill_names:
                started_at = perf_counter()
                try:
                    skill_parts.append(self.skills.render(name, {"intent": route.intent}))
                    self._append_step(
                        trace,
                        "skill",
                        name,
                        "success",
                        "已将本地 Skill 加入本轮提示词。",
                        self._elapsed_ms(started_at),
                    )
                except (OSError, ValueError) as exc:
                    warning = f"Skill {name} 加载失败：{self._safe_error(exc)}"
                    warnings.append(warning)
                    self._append_step(
                        trace,
                        "skill",
                        name,
                        "error",
                        warning,
                        self._elapsed_ms(started_at),
                    )
        else:
            summary = "本轮不需要专用 Skill。"
            if request.enabled_skills == []:
                summary = "用户在能力管理中关闭了全部 Skill。"
            self._append_step(trace, "skill", "Skill 选择", "skipped", summary)

        retrieved = []
        if route.need_rag and request.enable_rag:
            started_at = perf_counter()
            try:
                rag = self.knowledge.search(
                    request.query,
                    course_tags=route.course_tags,
                    top_k=self.settings.top_k,
                )
                retrieved = rag.chunks
                warnings.extend(rag.warnings)
                if not retrieved:
                    warnings.append("未检索到达到阈值的教材片段。")
                self._append_step(
                    trace,
                    "rag",
                    f"{rag.backend.upper()} 教材检索",
                    "success",
                    f"返回 {len(retrieved)} 条教材片段。",
                    rag.latency_ms,
                    {
                        "backend": rag.backend,
                        "top_k": self.settings.top_k,
                        "hits": len(retrieved),
                        "course_tags": route.course_tags,
                    },
                )
            except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                warning = f"教材检索失败：{self._safe_error(exc)}"
                warnings.append(warning)
                self._append_step(
                    trace,
                    "rag",
                    "教材检索",
                    "error",
                    warning,
                    self._elapsed_ms(started_at),
                )
        elif route.need_rag:
            self._append_step(
                trace,
                "rag",
                "教材检索",
                "skipped",
                "用户在能力管理中关闭了教材检索。",
            )
        else:
            self._append_step(trace, "rag", "教材检索", "skipped", "规划结果不需要教材证据。")

        tool_results: list[ToolResult] = []
        if route.tool_calls and request.enable_tools:
            for call in route.tool_calls:
                executor = self.mcp_tools if request.mcp_enabled else self.tools
                result = executor.run(call)
                tool_results.append(result)
                summary = "工具执行成功。"
                if result.status == "error":
                    summary = f"工具执行失败：{result.error}"
                    warnings.append(f"工具 {result.tool_name} 失败：{result.error}")
                self._append_step(
                    trace,
                    "tool",
                    result.tool_name,
                    result.status,
                    summary,
                    result.latency_ms,
                    {
                        "executor": ("mcp_in_process" if request.mcp_enabled else "local_registry"),
                        "arguments": call.arguments,
                        "result": result.result,
                        "artifacts": [item.path for item in result.artifacts],
                    },
                )
        elif route.tool_calls:
            self._append_step(
                trace,
                "tool",
                "专业工具",
                "skipped",
                "用户在能力管理中关闭了专业工具。",
            )
        else:
            self._append_step(trace, "tool", "专业工具", "skipped", "规划结果没有选择工具。")

        context = PromptContext(
            query=request.query,
            history=request.history,
            sources=[
                PromptSource(
                    title=chunk.citation.book_title,
                    section=chunk.citation.section,
                    text=chunk.text,
                )
                for chunk in retrieved
            ],
            tool_evidence=[
                PromptToolEvidence(
                    tool_name=result.tool_name,
                    result=result.result,
                    formula=result.formula,
                )
                for result in tool_results
                if result.status == "success"
            ],
            skill_instruction="\n\n".join(skill_parts) or None,
        )
        if request.structured_answer:
            context.skill_instruction = (context.skill_instruction or "") + (
                '\n最终答案必须额外附上一行 <final_answer>{"value": 数字或数字数组, '
                '"unit": "单位字符串，无量纲用空字符串"}</final_answer>。'
                "电学量与时间使用 SI 单位；这行必须和正文、工具结果一致。"
            )
        rounds, exchanges = [], []
        if any(t.status == "error" for t in tool_results) or (
            route.need_rag and request.enable_rag and not retrieved
        ):
            outcome = "failed"
            stop_reason = (
                "tool_error"
                if any(t.status == "error" for t in tool_results)
                else "evidence_unavailable"
            )
            answer = "任务未通过：必要工具或教材证据不可用。\n\n" + self._fallback_answer(
                tool_results, retrieved
            )
            model_result = ModelResult(
                text=answer,
                provider=self.settings.model_provider,
                model=self.settings.active_model,
                status="fallback",
                error_type=stop_reason,
            )
            self._append_step(trace, "stop", "Harness 停止", "error", stop_reason)
        else:
            harness = ReviewHarness(
                self.solver, self.reviewer, request.max_rounds or self.settings.max_rounds
            )
            result = harness.run(context, enable_review=request.enable_review)
            rounds, exchanges = result.rounds, result.messages
            outcome, stop_reason = result.outcome, result.stop_reason
            for step in result.trace:
                trace.append(step.model_copy(update={"step_id": len(trace) + 1}))
            model_result = rounds[-1].solver_result
            answer = result.answer or self._fallback_answer(tool_results, retrieved)
            if outcome == "failed":
                warnings.append(f"任务未验收：{stop_reason}。")
                answer = "未通过审核，以下内容仅供检查，不是已验收答案。\n\n" + answer
                model_result = model_result.model_copy(
                    update={"status": "fallback", "error_type": stop_reason}
                )
            for record in rounds:
                for call in (record.solver_result, record.reviewer_result):
                    if call is not None and call.error:
                        warnings.append(call.error)
        self._append_step(
            trace,
            "answer",
            "最终回答",
            "error" if outcome == "failed" else "success",
            f"返回 {len(retrieved)} 条引用和 {len(tool_results)} 个工具结果。",
        )
        model_steps = [
            s
            for s in trace
            if s.stage in {"model", "review"} or (s.stage == "planning" and "model" in s.metadata)
        ]
        response = AgentResponse(
            answer=answer,
            citations=[chunk.citation for chunk in retrieved],
            tool_results=tool_results,
            route=route,
            model_result=model_result,
            trace=trace,
            warnings=warnings,
            outcome=outcome,
            stop_reason=stop_reason,
            rounds=rounds,
            messages=exchanges,
            metrics=RunMetrics(
                execution_mode="simulated"
                if getattr(self.model, "provider", "") == "mock"
                else "live",
                model_calls=len(model_steps),
                input_tokens=sum(int(s.metadata.get("input_tokens", 0)) for s in model_steps),
                output_tokens=sum(int(s.metadata.get("output_tokens", 0)) for s in model_steps),
                total_latency_ms=self._elapsed_ms(started),
                tool_calls=len(tool_results),
            ),
        )
        try:
            self.artifact_writer.save(request, response, context)
        except OSError:
            response.outcome, response.stop_reason = "failed", "artifact_write_error"
            response.answer = "任务未验收：运行证据保存失败。\n\n" + response.answer
            response.model_result = response.model_result.model_copy(
                update={"status": "fallback", "error_type": "artifact_write_error"}
            )
            self._append_step(
                response.trace, "stop", "运行证据保存", "error", "artifact_write_error"
            )
            response.warnings.append("运行产物保存失败，请检查 runs 目录权限和磁盘空间。")
        return response

    @staticmethod
    def _fallback_answer(tool_results: list[ToolResult], retrieved: list) -> str:
        blocks = ["远程模型未完成生成，下面保留本地已经得到的结果。"]
        successful = [result for result in tool_results if result.status == "success"]
        for result in successful:
            content = json.dumps(result.result, ensure_ascii=False, sort_keys=True)
            blocks.append(f"工具 {result.tool_name}（{result.formula}）：{content}")
        for index, chunk in enumerate(retrieved, start=1):
            preview = " ".join(chunk.text.split())[:300]
            blocks.append(
                f"[资料{index}] {chunk.citation.book_title}｜{chunk.citation.section}：{preview}"
            )
        if not successful and not retrieved:
            blocks.append("当前没有可展示的工具结果或教材片段，请检查配置后重试。")
        return "\n\n".join(blocks)

    @staticmethod
    def _safe_error(error: Exception) -> str:
        message = str(error)
        cwd = str(Path.cwd())
        home = str(Path.home())
        return message.replace(cwd, "<project>").replace(home, "<home>")

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return max(0.0, (perf_counter() - started_at) * 1000)

    @staticmethod
    def _append_step(
        trace: list[ExecutionStep],
        stage: str,
        name: str,
        status: str,
        summary: str,
        latency_ms: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        trace.append(
            ExecutionStep(
                step_id=len(trace) + 1,
                stage=stage,
                name=name,
                status=status,
                summary=summary,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )
