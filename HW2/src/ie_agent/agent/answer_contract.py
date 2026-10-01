"""数值任务的可复核最终答案协议；不通过文本包含关系判断正确率。"""

import json
import math
import re

from ie_agent.contracts import PromptToolEvidence


def tool_answer(evidence: PromptToolEvidence) -> dict | None:
    result = evidence.result
    if evidence.tool_name == "entropy":
        return {"value": result["entropy"], "unit": result["unit"]}
    if evidence.tool_name == "ohms_law":
        return {"value": result["value_si"], "unit": result["si_unit"]}
    if evidence.tool_name == "rc_time_constant":
        return {"value": result["time_constant_seconds"], "unit": "s"}
    if evidence.tool_name == "discrete_convolution":
        return {"value": result["sequence"], "unit": ""}
    if evidence.tool_name == "scientific_calculator":
        return {"value": result["value"], "unit": ""}
    return None


def parse_final_answer(text: str) -> dict:
    matches = re.findall(r"<final_answer>(.*?)</final_answer>", text, flags=re.DOTALL)
    if len(matches) != 1:
        raise ValueError("必须恰有一个 final_answer")
    data = json.loads(matches[0])
    if not isinstance(data, dict) or set(data) != {"value", "unit"}:
        raise ValueError("最终答案必须只有 value 和 unit")
    if not isinstance(data["unit"], str):
        raise ValueError("单位必须是字符串")
    return data


def answers_equal(actual: dict, expected: dict) -> bool:
    def equal(a, b) -> bool:
        if isinstance(b, list):
            return (
                isinstance(a, list)
                and len(a) == len(b)
                and all(equal(x, y) for x, y in zip(a, b, strict=True))
            )
        return (
            type(a) in {int, float}
            and type(b) in {int, float}
            and math.isfinite(a)
            and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
        )

    return actual.get("unit") == expected.get("unit") and equal(
        actual.get("value"), expected.get("value")
    )
