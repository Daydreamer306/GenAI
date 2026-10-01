"""同一任务、模型和工具下比较单轮解答与审核循环。"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ie_agent.agent import AgentOrchestrator
from ie_agent.agent.answer_contract import answers_equal, parse_final_answer
from ie_agent.config import Settings
from ie_agent.contracts import AgentRequest
from ie_agent.model.demo import DemoModel


def run_comparison(
    settings: Settings,
    tasks_path: Path,
    *,
    offline: bool,
    limit: int = 0,
    progress: Callable[[dict], None] | None = None,
) -> Path:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("任务集不能为空")
    rows = []
    for task in tasks:
        for design in ("single_pass", "review_loop"):
            model = DemoModel("repair") if offline else None
            agent = AgentOrchestrator(settings, model=model)
            query = json.dumps(
                {"tool_name": task["tool_name"], "arguments": task["arguments"]}, ensure_ascii=False
            )
            result = agent.answer(
                AgentRequest(
                    session_id="comparison",
                    turn_id=uuid4().hex,
                    query=query,
                    planner="rules",
                    enable_review=design == "review_loop",
                    structured_answer=True,
                )
            )
            actual = None
            try:
                if result.outcome != "failed":
                    actual = parse_final_answer(result.answer)
            except (ValueError, TypeError):
                pass
            correct = actual is not None and answers_equal(actual, task["expected"])
            rows.append(
                {
                    "task_id": task["id"],
                    "design": design,
                    "correct": correct,
                    "expected": task["expected"],
                    "actual": actual,
                    "outcome": result.outcome,
                    "stop_reason": result.stop_reason,
                    "rounds": len(result.rounds),
                    "run_id": result.run_id,
                    **result.metrics.model_dump(),
                }
            )
            if progress is not None:
                progress(rows[-1])
    summary = []
    for design in ("single_pass", "review_loop"):
        group = [r for r in rows if r["design"] == design]
        summary.append(
            {
                "design": design,
                "tasks": len(group),
                "accuracy": sum(r["correct"] for r in group) / len(group),
                "failure_rate": sum(r["outcome"] == "failed" for r in group) / len(group),
                "input_tokens": sum(r["input_tokens"] for r in group),
                "output_tokens": sum(r["output_tokens"] for r in group),
                "model_calls": sum(r["model_calls"] for r in group),
                "average_latency_ms": sum(r["total_latency_ms"] for r in group) / len(group),
            }
        )
    directory = settings.runs_dir / (
        "comparison-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    )
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "execution_mode": "simulated" if offline else "live",
        "model": "demo-scripted" if offline else settings.active_model,
        "note": (
            "受控模拟故障注入仅验证流程，不代表真实模型准确率。"
            if offline
            else "真实模型小样本实验；不保证重新运行得到相同输出或准确率。"
        )
        + "成本使用观测 token，不估算货币费用。",
        "summary": summary,
        "items": rows,
    }
    (directory / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 两种设计的同任务比较",
        "",
        f"模式：{report['execution_mode']}",
        "",
        report["note"],
        "",
        "| 设计 | 任务数 | 正确率 | 失败率 | 模型调用 | 输入 token | 输出 token | 平均耗时 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        lines.append(
            f"| {s['design']} | {s['tasks']} | {s['accuracy']:.3f} | {s['failure_rate']:.3f} | "
            f"{s['model_calls']} | {s['input_tokens']} | {s['output_tokens']} | "
            f"{s['average_latency_ms']:.1f} |"
        )
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory
