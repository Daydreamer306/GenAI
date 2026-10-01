"""DeepSeek 与 MiniMax 共用的 OpenAI 兼容模型适配器。"""

import json
import re
from collections.abc import Sequence
from time import perf_counter
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from ie_agent.config import Settings
from ie_agent.contracts import (
    ChatMessage,
    ModelResult,
    PlannedToolCall,
    PlannerResult,
    RouteDecision,
    SkillSpec,
    ToolSpec,
)


class OpenAICompatibleModel:
    """只调用明确选择的提供商，不在失败时切换账户或服务。"""

    provider: Literal["deepseek", "minimax"] = "deepseek"

    @property
    def provider_label(self) -> str:
        return "MiniMax" if self.provider == "minimax" else "DeepSeek"

    def _setting(self, name: str) -> Any:
        return getattr(self.settings, f"{self.provider}_{name}")

    @property
    def model_name(self) -> str:
        return str(self._setting("model"))

    @property
    def has_key(self) -> bool:
        key = self._setting("api_key")
        return bool(key and key.get_secret_value().strip())

    def _completion_parameters(self, *, planning: bool = False) -> dict[str, Any]:
        if self.provider == "minimax":
            # M2.x thinking cannot be disabled; budget includes thinking tokens.
            return {"max_tokens": self._setting("max_tokens")}
        parameters: dict[str, Any] = {
            "max_tokens": 1_200 if planning else self.settings.deepseek_max_tokens,
            "extra_body": {
                "thinking": {
                    "type": "disabled" if planning else self.settings.deepseek_thinking,
                }
            },
        }
        if not planning:
            parameters["reasoning_effort"] = self.settings.deepseek_reasoning_effort
        return parameters

    def _public_content(self, content: str) -> str:
        if self.provider != "minimax":
            return content.strip()
        # Never expose inline thinking, including an unfinished thinking block.
        return re.sub(r"<think>[\s\S]*?(?:</think>|\Z)", "", content).strip()

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    def generate(self, messages: Sequence[ChatMessage]) -> ModelResult:
        """完成一次非流式调用，不保存模型的推理文本。"""

        started_at = perf_counter()
        if not messages:
            return self._error_result(
                started_at,
                "request_error",
                "模型消息不能为空。",
            )
        if not self.has_key:
            return self._error_result(
                started_at,
                "configuration_error",
                f"未配置 {self.provider_label} API key，请先填写本机 .env。",
            )

        client = self._client
        if client is None:
            client = self._create_client()
            self._client = client
        payload = [message.model_dump(mode="json") for message in messages]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=payload,
                stream=False,
                **self._completion_parameters(),
            )
        except AuthenticationError:
            return self._error_result(
                started_at,
                "authentication_error",
                f"{self.provider_label} 身份验证失败，请检查 API key。",
            )
        except RateLimitError:
            return self._error_result(
                started_at,
                "rate_limit_error",
                f"{self.provider_label} 请求过于频繁，请稍后重试。",
            )
        except APITimeoutError:
            return self._error_result(
                started_at,
                "timeout_error",
                f"{self.provider_label} 请求超时，请检查网络后重试。",
            )
        except APIConnectionError:
            return self._error_result(
                started_at,
                "connection_error",
                f"无法连接 {self.provider_label} 服务，请检查网络和 API 地址。",
            )
        except BadRequestError:
            return self._error_result(
                started_at,
                "request_error",
                f"{self.provider_label} 拒绝了请求参数，请检查模型配置。",
            )
        except APIStatusError as exc:
            return self._status_error_result(started_at, exc.status_code)
        except OpenAIError:
            return self._error_result(
                started_at,
                "api_error",
                f"{self.provider_label} 调用失败，请稍后重试。",
            )

        if not response.choices:
            return self._error_result(
                started_at,
                "response_error",
                f"{self.provider_label} 没有返回可用回答。",
            )

        choice = response.choices[0]
        text = self._public_content(choice.message.content or "")
        if choice.finish_reason == "length":
            return self._error_result(
                started_at,
                "truncated_response",
                f"{self.provider_label} 输出达到 token 上限，请提高对应 MAX_TOKENS 配置。",
            )
        if not text.strip():
            return self._error_result(
                started_at,
                "response_error",
                f"{self.provider_label} 返回了空回答。",
            )

        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        return ModelResult(
            text=text.strip(),
            provider=self.provider,
            model=str(getattr(response, "model", None) or self.model_name),
            status="success",
            input_tokens=self._nonnegative_int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=self._nonnegative_int(getattr(usage, "completion_tokens", 0)),
            reasoning_tokens=self._nonnegative_int(getattr(details, "reasoning_tokens", 0)),
            latency_ms=self._elapsed_ms(started_at),
            finish_reason=choice.finish_reason,
        )

    def plan(
        self,
        query: str,
        mode: Literal["json", "tool_calls"],
        tools: Sequence[ToolSpec],
        skills: Sequence[SkillSpec],
    ) -> PlannerResult:
        """让模型用 JSON 或原生 Tool Calls 生成一次行动计划。"""

        started_at = perf_counter()
        if not self.has_key:
            return PlannerResult(
                model_result=self._error_result(
                    started_at,
                    "configuration_error",
                    f"未配置 {self.provider_label} API key，无法使用模型规划。",
                )
            )
        client = self._client
        if client is None:
            client = self._create_client()
            self._client = client

        try:
            if mode == "json":
                response = self._request_json_plan(client, query, tools, skills)
                route = self._route_from_json(
                    self._json_content(response.choices[0].message.content or ""),
                    tools,
                    skills,
                )
            else:
                response = self._request_tool_plan(client, query, tools, skills)
                route = self._route_from_tool_calls(response.choices[0].message, tools, skills)
        except (IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return PlannerResult(
                model_result=self._error_result(
                    started_at,
                    "planner_response_error",
                    f"模型规划结果无法解析：{exc}",
                )
            )
        except OpenAIError:
            return PlannerResult(
                model_result=self._error_result(
                    started_at,
                    "planner_api_error",
                    f"{self.provider_label} 规划调用失败，请稍后重试。",
                )
            )

        if response.choices[0].finish_reason == "length":
            return PlannerResult(
                model_result=self._error_result(
                    started_at,
                    "truncated_response",
                    "模型规划达到 token 上限，结果不可用。",
                )
            )
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        model_result = ModelResult(
            text=route.model_dump_json(),
            provider=self.provider,
            model=str(getattr(response, "model", None) or self.model_name),
            status="success",
            input_tokens=self._nonnegative_int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=self._nonnegative_int(getattr(usage, "completion_tokens", 0)),
            reasoning_tokens=self._nonnegative_int(getattr(details, "reasoning_tokens", 0)),
            latency_ms=self._elapsed_ms(started_at),
            finish_reason=response.choices[0].finish_reason,
        )
        return PlannerResult(route=route, model_result=model_result)

    def _json_content(self, content: str) -> str:
        text = self._public_content(content)
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        return text

    def _request_json_plan(
        self,
        client: Any,
        query: str,
        tools: Sequence[ToolSpec],
        skills: Sequence[SkillSpec],
    ) -> Any:
        capabilities = {
            "tools": [item.model_dump() for item in tools],
            "skills": [item.model_dump() for item in skills],
        }
        prompt = (
            "请根据用户问题生成 JSON 行动计划，不要使用 Markdown 代码围栏。"
            "字段必须包含 intent、course_tags、"
            "need_rag、skill_names、tool_calls、reason。intent 只能是 chat、knowledge、"
            "calculation、formula、unknown。tool_calls 中每项包含 tool_name、arguments、"
            "reason。只选择能力列表中存在的名称，参数不足时不要调用工具。\n\n"
            f"能力列表：{json.dumps(capabilities, ensure_ascii=False)}\n\n"
            f"用户问题：{query}"
        )
        return client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是课程学习 Agent 的行动规划器，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            **({"response_format": {"type": "json_object"}} if self.provider == "deepseek" else {}),
            stream=False,
            **self._completion_parameters(planning=True),
        )

    def _request_tool_plan(
        self,
        client: Any,
        query: str,
        tools: Sequence[ToolSpec],
        skills: Sequence[SkillSpec],
    ) -> Any:
        function_tools = [
            {
                "type": "function",
                "function": {
                    "name": item.name,
                    "description": item.description,
                    "parameters": item.input_schema,
                },
            }
            for item in tools
        ]
        function_tools.append(
            {
                "type": "function",
                "function": {
                    "name": "search_course_materials",
                    "description": "需要教材定义、出处或课程证据时检索本地教材。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "course_tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )
        if skills:
            function_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "activate_skill",
                        "description": "选择最适合本轮回答方式的本地 Skill。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": [item.name for item in skills],
                                }
                            },
                            "required": ["name"],
                        },
                    },
                }
            )
        return client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是课程学习 Agent 的行动规划器。需要教材时调用检索，"
                        "需要特定回答方式时激活 Skill，需要计算或绘图时调用专业工具。"
                        "参数不足时不要猜测，也不要调用工具。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            tools=function_tools,
            tool_choice="auto",
            stream=False,
            **self._completion_parameters(planning=True),
        )

    @staticmethod
    def _route_from_json(
        content: str,
        tools: Sequence[ToolSpec],
        skills: Sequence[SkillSpec],
    ) -> RouteDecision:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("规划结果不是 JSON 对象")
        known_tools = {item.name for item in tools}
        known_skills = {item.name for item in skills}
        planned_calls = []
        for index, item in enumerate(data.get("tool_calls", []), start=1):
            if not isinstance(item, dict) or item.get("tool_name") not in known_tools:
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            planned_calls.append(
                PlannedToolCall(
                    call_id=f"json-tool-{index}",
                    tool_name=item["tool_name"],
                    arguments=arguments,
                    reason=str(item.get("reason") or "模型 JSON 规划"),
                )
            )
        skill_names = [
            str(name) for name in data.get("skill_names", []) if str(name) in known_skills
        ]
        return RouteDecision(
            intent=data.get("intent", "unknown"),
            planner="json",
            course_tags=[str(item) for item in data.get("course_tags", [])],
            need_rag=bool(data.get("need_rag", False)),
            tool_calls=planned_calls,
            skill_names=list(dict.fromkeys(skill_names)),
            reason=str(data.get("reason") or "模型 JSON 规划完成。"),
        )

    @staticmethod
    def _route_from_tool_calls(
        message: Any,
        tools: Sequence[ToolSpec],
        skills: Sequence[SkillSpec],
    ) -> RouteDecision:
        known_tools = {item.name for item in tools}
        known_skills = {item.name for item in skills}
        planned_calls: list[PlannedToolCall] = []
        skill_names: list[str] = []
        course_tags: list[str] = []
        need_rag = False
        for index, call in enumerate(message.tool_calls or [], start=1):
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")
            if not isinstance(arguments, dict):
                continue
            if name == "search_course_materials":
                need_rag = True
                course_tags.extend(str(item) for item in arguments.get("course_tags", []))
            elif name == "activate_skill":
                skill_name = str(arguments.get("name", ""))
                if skill_name in known_skills:
                    skill_names.append(skill_name)
            elif name in known_tools:
                planned_calls.append(
                    PlannedToolCall(
                        call_id=str(getattr(call, "id", None) or f"model-tool-{index}"),
                        tool_name=name,
                        arguments=arguments,
                        reason="模型 Tool Calls 选择该工具。",
                    )
                )

        if planned_calls:
            intent = "calculation"
        elif "formula_explanation" in skill_names:
            intent = "formula"
        elif need_rag:
            intent = "knowledge"
        else:
            intent = "chat"
        selected = [call.tool_name for call in planned_calls]
        selected.extend(skill_names)
        if need_rag:
            selected.append("search_course_materials")
        reason = "模型 Tool Calls 未选择额外能力。"
        if selected:
            reason = "模型 Tool Calls 选择：" + "、".join(selected) + "。"
        return RouteDecision(
            intent=intent,
            planner="tool_calls",
            course_tags=list(dict.fromkeys(course_tags)),
            need_rag=need_rag,
            tool_calls=planned_calls,
            skill_names=list(dict.fromkeys(skill_names)),
            reason=reason,
        )

    def _create_client(self) -> OpenAI:
        """只在真正调用时创建 SDK 客户端。"""

        key = self._setting("api_key")
        assert key is not None
        return OpenAI(
            api_key=key.get_secret_value(),
            base_url=self._setting("base_url"),
            timeout=self._setting("timeout_seconds"),
            max_retries=0,
        )

    def _status_error_result(self, started_at: float, status_code: int) -> ModelResult:
        """把服务状态码转换为不含请求细节的提示。"""

        if status_code == 402:
            return self._error_result(
                started_at,
                "balance_error",
                f"{self.provider_label} 账户余额不足，请检查账户状态。",
            )
        if status_code in {500, 503}:
            return self._error_result(
                started_at,
                "server_error",
                f"{self.provider_label} 服务暂时不可用，请稍后重试。",
            )
        return self._error_result(
            started_at,
            "api_status_error",
            f"{self.provider_label} 服务返回状态码 {status_code}。",
        )

    def _error_result(self, started_at: float, error_type: str, message: str) -> ModelResult:
        return ModelResult(
            provider=self.provider,
            model=self.model_name,
            status="error",
            latency_ms=self._elapsed_ms(started_at),
            error_type=error_type,
            error=message,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return max(0.0, (perf_counter() - started_at) * 1000)

    @staticmethod
    def _nonnegative_int(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0
