"""把课程工具、知识检索和 Skill 暴露为 MCP 能力。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from ie_agent.config import Settings
from ie_agent.contracts import PlannedToolCall, ToolResult
from ie_agent.knowledge import KnowledgeService
from ie_agent.knowledge.vector_store import chroma_index_ready
from ie_agent.skills import SkillLoader
from ie_agent.tools import ToolRegistry
from ie_agent.tools.common import Measurement


def create_mcp_server(
    settings: Settings,
    registry: ToolRegistry | None = None,
    knowledge: KnowledgeService | None = None,
    skills: SkillLoader | None = None,
    *,
    streamable_http_path: str = "/mcp",
) -> FastMCP:
    """创建复用项目核心模块的 MCP Server。"""

    registry = registry or ToolRegistry(settings.knowledge_dir / "artifacts")
    knowledge = knowledge or KnowledgeService(settings)
    skills = skills or SkillLoader()
    server = FastMCP(
        "IE-Agent",
        instructions="信息工程课程学习工具、教材检索和学习提示。",
        json_response=True,
        log_level="WARNING",
        stateless_http=True,
        streamable_http_path=streamable_http_path,
    )

    def run_tool(name: str, arguments: dict[str, object]) -> dict[str, Any]:
        result = registry.run(
            PlannedToolCall(
                call_id=f"mcp-{uuid4().hex[:8]}",
                tool_name=name,
                arguments=arguments,
                reason="由 MCP 客户端调用。",
            )
        )
        if result.status == "error":
            raise ValueError(result.error or "工具执行失败")
        return result.model_dump(mode="json")

    @server.tool(name="ohms_law", description="根据任意两个电学量计算欧姆定律中的未知量。")
    def mcp_ohms_law(
        voltage: Measurement | None = None,
        current: Measurement | None = None,
        resistance: Measurement | None = None,
    ) -> dict[str, Any]:
        return run_tool(
            "ohms_law",
            {
                "voltage": voltage.model_dump() if voltage else None,
                "current": current.model_dump() if current else None,
                "resistance": resistance.model_dump() if resistance else None,
            },
        )

    @server.tool(name="rc_time_constant", description="计算一阶 RC 电路时间常数。")
    def mcp_rc_time_constant(
        resistance: Measurement,
        capacitance: Measurement,
    ) -> dict[str, Any]:
        return run_tool(
            "rc_time_constant",
            {
                "resistance": resistance.model_dump(),
                "capacitance": capacitance.model_dump(),
            },
        )

    @server.tool(name="discrete_convolution", description="计算两个有限离散序列的卷积。")
    def mcp_discrete_convolution(x: list[float], h: list[float]) -> dict[str, Any]:
        return run_tool("discrete_convolution", {"x": x, "h": h})

    @server.tool(name="dft_basic", description="用定义式计算短实序列的 DFT。")
    def mcp_dft_basic(sequence: list[float]) -> dict[str, Any]:
        return run_tool("dft_basic", {"sequence": sequence})

    @server.tool(name="truth_table", description="解析布尔表达式并生成真值表。")
    def mcp_truth_table(expression: str) -> dict[str, Any]:
        return run_tool("truth_table", {"expression": expression})

    @server.tool(name="entropy", description="计算离散信源熵。")
    def mcp_entropy(probabilities: list[float], base: float = 2.0) -> dict[str, Any]:
        return run_tool("entropy", {"probabilities": probabilities, "base": base})

    @server.tool(name="scientific_calculator", description="计算带变量的科学数学表达式。")
    def mcp_scientific_calculator(
        expression: str,
        variables: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return run_tool(
            "scientific_calculator",
            {"expression": expression, "variables": variables or {}},
        )

    @server.tool(name="plot_function", description="绘制一元函数并返回 PNG 文件信息。")
    def mcp_plot_function(
        expression: str,
        x_min: float = -10.0,
        x_max: float = 10.0,
        points: int = 400,
        title: str | None = None,
    ) -> dict[str, Any]:
        return run_tool(
            "plot_function",
            {
                "expression": expression,
                "x_min": x_min,
                "x_max": x_max,
                "points": points,
                "title": title,
            },
        )

    @server.tool(name="plot_sequence", description="绘制有限离散序列并返回 PNG 文件信息。")
    def mcp_plot_sequence(
        values: list[float],
        start_index: int = 0,
        title: str | None = None,
    ) -> dict[str, Any]:
        return run_tool(
            "plot_sequence",
            {"values": values, "start_index": start_index, "title": title},
        )

    @server.tool(name="search_course_materials", description="从本地教材索引检索相关片段。")
    def search_course_materials(
        query: str,
        course_tags: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        result = knowledge.search(query, course_tags=course_tags, top_k=top_k)
        return result.model_dump(mode="json")

    prompt_names = {
        "explain_concept": "concept_explanation",
        "solve_problem": "problem_solving",
        "explain_formula": "formula_explanation",
        "guide_experiment": "experiment_guidance",
    }

    @server.prompt(name="explain_concept", description="按课程学习结构解释概念。")
    def explain_concept(question: str) -> str:
        return f"{skills.render(prompt_names['explain_concept'])}\n\n待解释概念：{question}"

    @server.prompt(name="solve_problem", description="按规范步骤完成课程习题。")
    def solve_problem(question: str) -> str:
        return f"{skills.render(prompt_names['solve_problem'])}\n\n待求解问题：{question}"

    @server.prompt(name="explain_formula", description="解释公式符号、条件和含义。")
    def explain_formula(question: str) -> str:
        return f"{skills.render(prompt_names['explain_formula'])}\n\n待解释公式：{question}"

    @server.prompt(name="guide_experiment", description="给出课程实验的实施与分析结构。")
    def guide_experiment(question: str) -> str:
        return f"{skills.render(prompt_names['guide_experiment'])}\n\n实验任务：{question}"

    @server.resource(
        "ie-agent://course-catalog",
        name="course_catalog",
        description="项目中已分类的教材目录。",
        mime_type="application/yaml",
    )
    def course_catalog() -> str:
        return Path("knowledge/catalog.yaml").read_text(encoding="utf-8")

    @server.resource(
        "ie-agent://capabilities",
        name="capabilities",
        description="当前模型、检索、工具和 Skill 的非敏感状态。",
        mime_type="application/json",
    )
    def capabilities() -> str:
        index_dir = settings.knowledge_dir / "index"
        data = {
            "provider": settings.model_provider,
            "model": settings.active_model,
            "rag_backend": settings.rag_backend,
            "qwen_index_ready": chroma_index_ready(index_dir / "qwen_chroma"),
            "tools": [item.name for item in registry.list()],
            "skills": [item.name for item in skills.list()],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    return server


class InProcessMCPTools:
    """通过 FastMCP 对象调用工具，不启动 HTTP 端口。"""

    def __init__(
        self,
        settings: Settings,
        registry: ToolRegistry,
        knowledge: KnowledgeService,
        skills: SkillLoader,
    ) -> None:
        self.server = create_mcp_server(settings, registry, knowledge, skills)

    def run(self, call: PlannedToolCall) -> ToolResult:
        started_at = perf_counter()
        try:
            response = asyncio.run(self.server.call_tool(call.tool_name, call.arguments))
            data = response[1] if isinstance(response, tuple) else response
            if not isinstance(data, dict):
                raise ValueError("MCP 工具未返回结构化结果")
            result = ToolResult.model_validate(data)
            return result.model_copy(update={"call_id": call.call_id})
        except Exception as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                error=f"MCP 调用失败：{exc}",
                latency_ms=(perf_counter() - started_at) * 1000,
            )
