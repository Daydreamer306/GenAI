"""明确标记的离线脚本模型，用真实工具结果验证 harness，不冒充真实 LLM。"""

import json
import re

from ie_agent.agent.answer_contract import answers_equal, parse_final_answer, tool_answer
from ie_agent.contracts import ModelResult, PromptToolEvidence


class DemoModel:
    provider = "mock"

    def __init__(self, scenario: str = "repair") -> None:
        self.scenario = scenario
        self.solver_calls = 0

    def generate(self, messages) -> ModelResult:
        reviewing = "Reviewer Agent" in messages[0].content and "独立审核" in messages[0].content
        if reviewing:
            if self.scenario == "reviewer-error":
                return self._result("", "error")
            if self.scenario == "invalid-review":
                return self._result("这不是有效的审核 JSON")
            payload = json.loads(messages[-1].content)
            expected = (
                tool_answer(PromptToolEvidence.model_validate(payload["tools"][-1]))
                if payload["tools"]
                else None
            )
            try:
                approved = expected is None or answers_equal(
                    parse_final_answer(payload["draft"]), expected
                )
            except ValueError:
                approved = False
            if self.scenario == "reject":
                approved = False
            return self._result(
                json.dumps(
                    {
                        "approved": approved,
                        "summary": "数值、单位和证据通过审核。"
                        if approved
                        else "数值或单位需要修订。",
                        "issues": [] if approved else ["请按真实工具结果修正最终数值及单位。"],
                    },
                    ensure_ascii=False,
                )
            )
        self.solver_calls += 1
        if self.scenario == "solver-error":
            return self._result("", "error")
        payload = messages[1].content
        match = re.search(r"名称：(.*?)\n公式：.*?\n结构化结果：(\{[^\n]+\})", payload)
        final = {"value": 0, "unit": ""}
        if match:
            evidence = PromptToolEvidence(tool_name=match[1], result=json.loads(match[2]))
            final = tool_answer(evidence) or final
        wrong = self.scenario == "reject" or (self.scenario == "repair" and self.solver_calls == 1)
        if wrong:
            value = final["value"]
            final = {
                **final,
                "value": [v + 1 for v in value] if isinstance(value, list) else value + 1,
            }
        citation = " 依据教材 [资料1]。" if "[资料1]" in payload else ""
        text = (
            f"[离线模拟 Agent 输出] 工具结果用于求解，公式与单位见证据。{citation}\n"
            + "<final_answer>"
            + json.dumps(final, ensure_ascii=False)
            + "</final_answer>"
        )
        return self._result(text)

    @staticmethod
    def _result(text: str, status: str = "success") -> ModelResult:
        return ModelResult(
            text=text,
            provider="mock",
            model="offline-scripted-model",
            status=status,
            error_type="simulated_error" if status == "error" else None,
            error="模拟模型失败" if status == "error" else None,
        )
