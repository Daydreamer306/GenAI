"""信号与系统课程的序列计算工具。"""

from __future__ import annotations

import cmath
import math

from pydantic import Field

from ie_agent.contracts import StrictModel
from ie_agent.tools.common import ToolComputation


class DiscreteConvolutionInput(StrictModel):
    x: list[float] = Field(min_length=1, max_length=128)
    h: list[float] = Field(min_length=1, max_length=128)


class DFTInput(StrictModel):
    sequence: list[float] = Field(min_length=1, max_length=64)


def discrete_convolution(data: DiscreteConvolutionInput) -> ToolComputation:
    """直接按照卷积和定义计算有限序列。"""

    output = [0.0] * (len(data.x) + len(data.h) - 1)
    for x_index, x_value in enumerate(data.x):
        for h_index, h_value in enumerate(data.h):
            output[x_index + h_index] += x_value * h_value
    output = [_clean(value) for value in output]
    return ToolComputation(
        result={"sequence": output, "length": len(output), "start_index": 0},
        formula="y[n] = Σ_k x[k] h[n-k]",
        steps=[
            "默认两个输入序列都从 n=0 开始。",
            "逐项相乘，并把下标和相同的乘积累加。",
        ],
    )


def dft_basic(data: DFTInput) -> ToolComputation:
    """用定义式计算短序列 DFT。"""

    length = len(data.sequence)
    values: list[complex] = []
    for frequency in range(length):
        value = sum(
            sample * cmath.exp(-2j * math.pi * frequency * index / length)
            for index, sample in enumerate(data.sequence)
        )
        values.append(value)
    real = [_clean(value.real) for value in values]
    imaginary = [_clean(value.imag) for value in values]
    magnitude = [_clean(abs(value)) for value in values]
    phase = [_clean(cmath.phase(value)) if abs(value) > 1e-12 else 0.0 for value in values]
    return ToolComputation(
        result={
            "real": real,
            "imaginary": imaginary,
            "magnitude": magnitude,
            "phase_radians": phase,
            "length": length,
        },
        formula="X[k] = Σ_{n=0}^{N-1} x[n] e^{-j2πkn/N}",
        steps=[f"序列长度 N={length}。", "对每个 k=0,…,N-1 直接代入 DFT 定义式。"],
    )


def _clean(value: float) -> float:
    if abs(value) < 1e-12:
        return 0.0
    return round(float(value), 12)
