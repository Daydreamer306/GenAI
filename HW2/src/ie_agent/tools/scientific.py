"""通用科学计算和绘图工具。"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import Field, field_validator

from ie_agent.contracts import StrictModel, ToolArtifact
from ie_agent.tools.common import ToolComputation

if TYPE_CHECKING:
    import numpy as np


class ScientificCalculatorInput(StrictModel):
    expression: str = Field(min_length=1, max_length=200)
    variables: dict[str, float] = Field(default_factory=dict)

    @field_validator("variables")
    @classmethod
    def variables_must_be_simple(cls, values: dict[str, float]) -> dict[str, float]:
        for name, value in values.items():
            if not name.isidentifier():
                raise ValueError(f"变量名不合法：{name}")
            if not math.isfinite(value):
                raise ValueError(f"变量 {name} 必须是有限实数")
        return values


class PlotFunctionInput(StrictModel):
    expression: str = Field(min_length=1, max_length=200)
    x_min: float = -10.0
    x_max: float = 10.0
    points: int = Field(default=400, ge=100, le=2_000)
    title: str | None = Field(default=None, max_length=80)

    @field_validator("x_max")
    @classmethod
    def interval_must_increase(cls, value: float, info) -> float:
        x_min = info.data.get("x_min")
        if x_min is not None and value <= x_min:
            raise ValueError("x_max 必须大于 x_min")
        return value


class PlotSequenceInput(StrictModel):
    values: list[float] = Field(min_length=1, max_length=2_000)
    start_index: int = 0
    title: str | None = Field(default=None, max_length=80)

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("序列只能包含有限实数")
        return values


def scientific_calculator(data: ScientificCalculatorInput) -> ToolComputation:
    """使用 NumExpr 计算受限数学表达式。"""

    import numexpr as ne
    import numpy as np

    local_values = {**data.variables, "pi": np.pi, "e": np.e}
    expression = _normalize_expression(data.expression)
    try:
        value = ne.evaluate(expression, local_dict=local_values)
    except (KeyError, SyntaxError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"表达式无法计算：{exc}") from exc
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("科学计算器只接受结果为单个数值的表达式")
    result = float(array.item())
    if not math.isfinite(result):
        raise ValueError("计算结果不是有限实数")
    return ToolComputation(
        result={"value": result},
        formula=expression,
        steps=[f"代入变量：{data.variables or '无'}", f"计算结果：{result:.10g}"],
    )


def plot_function(data: PlotFunctionInput, artifact_dir: Path) -> ToolComputation:
    """在给定区间绘制一元函数。"""

    import numexpr as ne
    import numpy as np

    x = np.linspace(data.x_min, data.x_max, data.points)
    expression = _normalize_expression(data.expression)
    try:
        raw_y = ne.evaluate(
            expression,
            local_dict={"x": x, "pi": np.pi, "e": np.e},
        )
    except (KeyError, SyntaxError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(f"函数表达式无法计算：{exc}") from exc
    y = np.asarray(raw_y, dtype=float)
    if y.size == 1:
        y = np.full_like(x, float(y.item()))
    if y.shape != x.shape:
        raise ValueError("函数结果维度与横坐标不一致")
    if not np.isfinite(y).any():
        raise ValueError("指定区间内没有可绘制的有限值")

    artifact = _save_function_figure(data, x, y, artifact_dir)
    return ToolComputation(
        result={
            "expression": data.expression,
            "x_range": [data.x_min, data.x_max],
            "points": data.points,
            "artifact": artifact.name,
        },
        formula=f"y = {expression}",
        steps=[
            f"在 [{data.x_min}, {data.x_max}] 上生成 {data.points} 个采样点。",
            f"图像已保存为 {artifact.name}。",
        ],
        artifacts=[artifact],
    )


def plot_sequence(data: PlotSequenceInput, artifact_dir: Path) -> ToolComputation:
    """绘制有限离散序列。"""

    import numpy as np

    plt = _load_pyplot(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    name = f"sequence-{uuid4().hex[:12]}.png"
    indices = np.arange(data.start_index, data.start_index + len(data.values))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.stem(indices, data.values, basefmt="C0-")
    axis.set_xlabel("n")
    axis.set_ylabel("x[n]")
    axis.set_title(data.title or "Discrete sequence")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    path = artifact_dir / name
    figure.savefig(path, dpi=150)
    plt.close(figure)
    artifact = ToolArtifact(
        name=name,
        title=data.title or "离散序列图",
        mime_type="image/png",
        path=str(path),
    )
    return ToolComputation(
        result={
            "indices": indices.tolist(),
            "values": data.values,
            "artifact": name,
        },
        formula="x[n]",
        steps=[f"从 n={data.start_index} 开始绘制 {len(data.values)} 个离散样点。"],
        artifacts=[artifact],
    )


def _save_function_figure(
    data: PlotFunctionInput,
    x: np.ndarray,
    y: np.ndarray,
    artifact_dir: Path,
) -> ToolArtifact:
    plt = _load_pyplot(artifact_dir)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    name = f"function-{uuid4().hex[:12]}.png"
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(x, y, color="#2563eb", linewidth=2)
    axis.axhline(0, color="#64748b", linewidth=0.8)
    axis.axvline(0, color="#64748b", linewidth=0.8)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(data.title or f"y = {data.expression}")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    path = artifact_dir / name
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return ToolArtifact(
        name=name,
        title=data.title or "函数图像",
        mime_type="image/png",
        path=str(path),
    )


def _normalize_expression(expression: str) -> str:
    """把常见的数学幂符号转换为 NumExpr 支持的写法。"""

    return expression.replace("^", "**")


def _load_pyplot(artifact_dir: Path):
    """使用项目内缓存并加载无窗口绘图后端。"""

    cache_root = artifact_dir / ".cache"
    matplotlib_config = cache_root / "matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "SimHei",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    return plt
