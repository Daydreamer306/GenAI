"""选择规则、JSON 或 Tool Calls 规划方式。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ie_agent.agent.router import IntentRouter
from ie_agent.config import Settings
from ie_agent.contracts import ModelResult, PlannedToolCall, RouteDecision
from ie_agent.skills import SkillLoader
from ie_agent.tools import ToolRegistry

_TOOL_ACTION = re.compile(r"(?:计算(?!机)|求值|算出|算一下|帮我算|画出|绘制|画图|作图|画一下)")


@dataclass
class PlanningOutcome:
    route: RouteDecision
    latency_ms: float
    model_result: ModelResult | None = None
    warning: str | None = None


class AgentPlanner:
    """把四种入口统一转换为 RouteDecision。"""

    def __init__(
        self,
        settings: Settings,
        model: Any,
        router: IntentRouter,
        tools: ToolRegistry,
        skills: SkillLoader,
    ) -> None:
        self.settings = settings
        self.model = model
        self.router = router
        self.tools = tools
        self.skills = skills

    def plan(
        self,
        query: str,
        *,
        planner: str | None = None,
        enabled_skills: list[str] | None = None,
        enable_tools: bool = True,
    ) -> PlanningOutcome:
        started_at = perf_counter()
        all_skills = self.skills.list()
        allowed_skill_names = (
            {item.name for item in all_skills} if enabled_skills is None else set(enabled_skills)
        )
        available_skills = [item for item in all_skills if item.name in allowed_skill_names]
        available_tools = self.tools.list() if enable_tools else []

        direct = self._direct_json(query) if enable_tools else None
        if direct is not None:
            direct = direct.model_copy(
                update={
                    "skill_names": [
                        name for name in direct.skill_names if name in allowed_skill_names
                    ]
                }
            )
            return PlanningOutcome(direct, self._elapsed_ms(started_at))

        rule_route = self.router.route(query)
        rule_route = rule_route.model_copy(
            update={
                "skill_names": [
                    name for name in rule_route.skill_names if name in allowed_skill_names
                ],
                "tool_calls": rule_route.tool_calls if enable_tools else [],
            }
        )
        mode = planner or self.settings.planner
        if mode == "rules":
            return PlanningOutcome(rule_route, self._elapsed_ms(started_at))
        asks_for_tool = bool(_TOOL_ACTION.search(query))
        if (
            mode == "hybrid"
            and rule_route.intent != "unknown"
            and (rule_route.tool_calls or not asks_for_tool)
        ):
            return PlanningOutcome(rule_route, self._elapsed_ms(started_at))

        planner_mode = "tool_calls" if mode == "hybrid" else mode
        if not hasattr(self.model, "plan"):
            return PlanningOutcome(
                rule_route,
                self._elapsed_ms(started_at),
                warning="当前模型适配器不支持规划，已使用规则路由。",
            )
        planned = self.model.plan(
            query,
            planner_mode,
            available_tools,
            available_skills,
        )
        if planned.route is None:
            return PlanningOutcome(
                rule_route,
                self._elapsed_ms(started_at),
                model_result=planned.model_result,
                warning=planned.model_result.error or "模型规划失败，已使用规则路由。",
            )
        return PlanningOutcome(
            planned.route,
            self._elapsed_ms(started_at),
            model_result=planned.model_result,
        )

    def _direct_json(self, query: str) -> RouteDecision | None:
        text = query.strip()
        if not text.startswith("{"):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or set(data) != {"tool_name", "arguments"}:
            return None
        name = data.get("tool_name")
        arguments = data.get("arguments")
        known_tools = {item.name for item in self.tools.list()}
        if not isinstance(name, str) or name not in known_tools or not isinstance(arguments, dict):
            return None
        return RouteDecision(
            intent="calculation",
            planner="direct_json",
            tool_calls=[
                PlannedToolCall(
                    call_id="json-direct-1",
                    tool_name=name,
                    arguments=arguments,
                    reason="用户通过 JSON 明确指定工具。",
                )
            ],
            skill_names=["problem_solving"],
            reason=f"JSON 输入指定调用 {name}。",
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return max(0.0, (perf_counter() - started_at) * 1000)
