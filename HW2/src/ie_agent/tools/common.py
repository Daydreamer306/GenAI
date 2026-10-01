"""多个专业工具共用的单位和返回结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import Field, field_validator

from ie_agent.contracts import StrictModel, ToolArtifact


class Measurement(StrictModel):
    """一个带单位的实数。"""

    value: float
    unit: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("数值必须是有限实数")
        return value


@dataclass(frozen=True)
class ToolComputation:
    """工具处理函数返回给 Registry 的中间结果。"""

    result: dict[str, object]
    formula: str
    steps: list[str]
    artifacts: list[ToolArtifact] = field(default_factory=list)


VOLTAGE_UNITS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "kv": 1e3}
CURRENT_UNITS = {"a": 1.0, "ma": 1e-3, "ua": 1e-6, "ka": 1e3}
RESISTANCE_UNITS = {
    "ohm": 1.0,
    "ω": 1.0,
    "kohm": 1e3,
    "kω": 1e3,
    "mohm": 1e6,
    "mω": 1e6,
}
CAPACITANCE_UNITS = {
    "f": 1.0,
    "mf": 1e-3,
    "uf": 1e-6,
    "nf": 1e-9,
    "pf": 1e-12,
}


def to_si(measurement: Measurement, units: dict[str, float], quantity: str) -> float:
    """把常见工程单位换算成 SI。"""

    key = measurement.unit.strip().replace("μ", "u").replace("µ", "u").lower()
    if key not in units:
        supported = ", ".join(sorted(units))
        raise ValueError(f"{quantity}单位不支持：{measurement.unit}；可用 {supported}")
    return measurement.value * units[key]


def engineering_display(value: float, quantity: str) -> tuple[float, str]:
    """为计算结果选择便于阅读的常用单位。"""

    absolute = abs(value)
    tables = {
        "voltage": [(1e3, "kV"), (1.0, "V"), (1e-3, "mV"), (1e-6, "uV")],
        "current": [(1.0, "A"), (1e-3, "mA"), (1e-6, "uA")],
        "resistance": [(1e6, "MOhm"), (1e3, "kOhm"), (1.0, "Ohm")],
        "capacitance": [(1.0, "F"), (1e-3, "mF"), (1e-6, "uF"), (1e-9, "nF")],
        "time": [(1.0, "s"), (1e-3, "ms"), (1e-6, "us")],
    }
    for factor, unit in tables[quantity]:
        if absolute >= factor or factor == tables[quantity][-1][0]:
            return value / factor, unit
    raise AssertionError("单位表不能为空")
