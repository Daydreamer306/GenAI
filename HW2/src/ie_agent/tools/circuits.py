"""基础电路计算工具。"""

from __future__ import annotations

from pydantic import model_validator

from ie_agent.contracts import StrictModel
from ie_agent.tools.common import (
    CAPACITANCE_UNITS,
    CURRENT_UNITS,
    RESISTANCE_UNITS,
    VOLTAGE_UNITS,
    Measurement,
    ToolComputation,
    engineering_display,
    to_si,
)


class OhmsLawInput(StrictModel):
    """电压、电流、电阻中必须给出任意两项。"""

    voltage: Measurement | None = None
    current: Measurement | None = None
    resistance: Measurement | None = None

    @model_validator(mode="after")
    def exactly_two_values(self) -> OhmsLawInput:
        if sum(value is not None for value in (self.voltage, self.current, self.resistance)) != 2:
            raise ValueError("电压、电流、电阻必须恰好给出两项")
        return self


class RCTimeConstantInput(StrictModel):
    """RC 时间常数所需参数。"""

    resistance: Measurement
    capacitance: Measurement


def ohms_law(data: OhmsLawInput) -> ToolComputation:
    """根据 U=IR 计算唯一未知量。"""

    voltage = to_si(data.voltage, VOLTAGE_UNITS, "电压") if data.voltage else None
    current = to_si(data.current, CURRENT_UNITS, "电流") if data.current else None
    resistance = to_si(data.resistance, RESISTANCE_UNITS, "电阻") if data.resistance else None
    if resistance is not None and resistance <= 0:
        raise ValueError("电阻必须大于 0")

    if voltage is None:
        assert current is not None and resistance is not None
        value = current * resistance
        unknown, si_unit, formula = "voltage", "V", "U = I R"
        substitution = f"U = {current:g} A × {resistance:g} Ohm"
    elif current is None:
        assert resistance is not None
        value = voltage / resistance
        unknown, si_unit, formula = "current", "A", "I = U / R"
        substitution = f"I = {voltage:g} V / {resistance:g} Ohm"
    else:
        if current == 0:
            raise ValueError("用 U/I 计算电阻时，电流不能为 0")
        value = voltage / current
        if value <= 0:
            raise ValueError("计算得到的电阻必须大于 0，请检查电压和电流方向")
        unknown, si_unit, formula = "resistance", "Ohm", "R = U / I"
        substitution = f"R = {voltage:g} V / {current:g} A"

    display_value, display_unit = engineering_display(value, unknown)
    return ToolComputation(
        result={
            "unknown": unknown,
            "value_si": value,
            "si_unit": si_unit,
            "display_value": display_value,
            "display_unit": display_unit,
        },
        formula=formula,
        steps=["先把已知量统一换算为 SI 单位。", substitution, "再换成便于阅读的工程单位。"],
    )


def rc_time_constant(data: RCTimeConstantInput) -> ToolComputation:
    """计算一阶 RC 电路的时间常数。"""

    resistance = to_si(data.resistance, RESISTANCE_UNITS, "电阻")
    capacitance = to_si(data.capacitance, CAPACITANCE_UNITS, "电容")
    if resistance <= 0 or capacitance <= 0:
        raise ValueError("电阻和电容必须大于 0")
    tau = resistance * capacitance
    display_value, display_unit = engineering_display(tau, "time")
    return ToolComputation(
        result={
            "time_constant_seconds": tau,
            "display_value": display_value,
            "display_unit": display_unit,
        },
        formula="τ = R C",
        steps=[
            f"R = {resistance:g} Ohm，C = {capacitance:g} F。",
            f"τ = {resistance:g} × {capacitance:g} = {tau:g} s。",
        ],
    )
