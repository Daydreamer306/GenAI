"""九个确定性工具的注册与参数校验。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ValidationError

from ie_agent.contracts import PlannedToolCall, ToolResult, ToolSpec
from ie_agent.tools.circuits import (
    OhmsLawInput,
    RCTimeConstantInput,
    ohms_law,
    rc_time_constant,
)
from ie_agent.tools.common import ToolComputation
from ie_agent.tools.digital_logic import TruthTableInput, truth_table
from ie_agent.tools.information import EntropyInput, entropy
from ie_agent.tools.scientific import (
    PlotFunctionInput,
    PlotSequenceInput,
    ScientificCalculatorInput,
    plot_function,
    plot_sequence,
    scientific_calculator,
)
from ie_agent.tools.signals import (
    DFTInput,
    DiscreteConvolutionInput,
    dft_basic,
    discrete_convolution,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], ToolComputation]


class ToolRegistry:
    """工具名称、Pydantic 输入和处理函数的唯一映射。"""

    def __init__(self, artifact_dir: Path = Path("knowledge/local/artifacts")) -> None:
        definitions = [
            ToolDefinition("ohms_law", "根据欧姆定律求电压、电流或电阻。", OhmsLawInput, ohms_law),
            ToolDefinition(
                "rc_time_constant",
                "计算一阶 RC 电路时间常数。",
                RCTimeConstantInput,
                rc_time_constant,
            ),
            ToolDefinition(
                "discrete_convolution",
                "计算两个有限离散序列的卷积。",
                DiscreteConvolutionInput,
                discrete_convolution,
            ),
            ToolDefinition("dft_basic", "用定义式计算短序列 DFT。", DFTInput, dft_basic),
            ToolDefinition(
                "truth_table",
                "安全解析布尔表达式并生成真值表。",
                TruthTableInput,
                truth_table,
            ),
            ToolDefinition("entropy", "计算离散信源熵。", EntropyInput, entropy),
            ToolDefinition(
                "scientific_calculator",
                "计算包含常用函数和变量的数学表达式。",
                ScientificCalculatorInput,
                scientific_calculator,
            ),
            ToolDefinition(
                "plot_function",
                "绘制指定区间上的一元函数曲线。",
                PlotFunctionInput,
                lambda data: plot_function(data, artifact_dir),
            ),
            ToolDefinition(
                "plot_sequence",
                "绘制有限离散序列的杆状图。",
                PlotSequenceInput,
                lambda data: plot_sequence(data, artifact_dir),
            ),
        ]
        self._definitions = {definition.name: definition for definition in definitions}

    def list(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name=definition.name,
                description=definition.description,
                input_schema=definition.input_model.model_json_schema(),
            )
            for definition in self._definitions.values()
        ]

    def run(self, call: PlannedToolCall) -> ToolResult:
        started_at = perf_counter()
        definition = self._definitions.get(call.tool_name)
        if definition is None:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                latency_ms=self._elapsed_ms(started_at),
                error=f"未知工具：{call.tool_name}",
            )
        try:
            arguments = definition.input_model.model_validate(call.arguments)
            computation = definition.handler(arguments)
        except ValidationError as exc:
            messages = [
                f"{'.'.join(str(item) for item in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                latency_ms=self._elapsed_ms(started_at),
                error="；".join(messages),
            )
        except (ImportError, ArithmeticError, ValueError) as exc:
            return ToolResult(
                call_id=call.call_id,
                tool_name=call.tool_name,
                status="error",
                latency_ms=self._elapsed_ms(started_at),
                error=str(exc),
            )
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status="success",
            result=computation.result,
            formula=computation.formula,
            steps=computation.steps,
            artifacts=computation.artifacts,
            latency_ms=self._elapsed_ms(started_at),
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return max(0.0, (perf_counter() - started_at) * 1000)
