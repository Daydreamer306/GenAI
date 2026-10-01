"""信息论课程的确定性计算工具。"""

from __future__ import annotations

import math

from pydantic import Field, field_validator

from ie_agent.contracts import StrictModel
from ie_agent.tools.common import ToolComputation


class EntropyInput(StrictModel):
    probabilities: list[float] = Field(min_length=1, max_length=128)
    base: float = 2.0

    @field_validator("probabilities")
    @classmethod
    def probabilities_are_valid(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError("每个概率都必须位于 0 到 1")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("概率之和必须等于 1")
        return values

    @field_validator("base")
    @classmethod
    def base_is_valid(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0 or math.isclose(value, 1.0):
            raise ValueError("对数底必须大于 0 且不等于 1")
        return value


def entropy(data: EntropyInput) -> ToolComputation:
    """计算离散信源熵及每个符号的贡献。"""

    contributions = [
        0.0 if probability == 0 else -probability * math.log(probability, data.base)
        for probability in data.probabilities
    ]
    value = sum(contributions)
    if math.isclose(data.base, 2.0):
        unit = "bit"
    elif math.isclose(data.base, math.e):
        unit = "nat"
    elif math.isclose(data.base, 10.0):
        unit = "Hartley"
    else:
        unit = f"log base {data.base:g} unit"
    return ToolComputation(
        result={
            "entropy": value,
            "unit": unit,
            "contributions": contributions,
            "probability_sum": sum(data.probabilities),
        },
        formula="H(X) = -Σ_i p_i log_b(p_i)",
        steps=["先检查各概率之和为 1。", "分别计算 -p_i log_b(p_i)，再求和。"],
    )
