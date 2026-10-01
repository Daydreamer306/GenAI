"""每次问答保存可检查的 trace、证据、报告、审核和测量数据。"""

import json
from datetime import UTC, datetime
from uuid import uuid4

from ie_agent.config import Settings
from ie_agent.contracts import AgentRequest, AgentResponse, PromptContext


class RunArtifactWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def redact(self, text: str) -> str:
        for key in (self.settings.minimax_api_key, self.settings.deepseek_api_key):
            if key and key.get_secret_value():
                text = text.replace(key.get_secret_value(), "<redacted>")
        return text

    def save(self, request: AgentRequest, response: AgentResponse, context: PromptContext) -> None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
        directory = self.settings.runs_dir / run_id
        directory.mkdir(parents=True, exist_ok=False)
        response.run_id, response.artifact_dir = run_id, str(directory)

        def write(name: str, text: str) -> None:
            (directory / name).write_text(self.redact(text) + "\n", encoding="utf-8")

        write("request.json", request.model_dump_json(indent=2))
        write("evidence.json", context.model_dump_json(indent=2))
        write("trace.jsonl", "\n".join(s.model_dump_json() for s in response.trace))
        write(
            "messages.json",
            json.dumps([m.model_dump() for m in response.messages], ensure_ascii=False, indent=2),
        )
        write(
            "review.json",
            json.dumps(
                {
                    "outcome": response.outcome,
                    "stop_reason": response.stop_reason,
                    "rounds": [r.model_dump() for r in response.rounds],
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        write("metrics.json", response.metrics.model_dump_json(indent=2))
        write("response.json", response.model_dump_json(indent=2))
        references = "\n".join(
            f"- [资料{n}] {c.book_title}｜{c.section}｜{c.chunk_id}"
            for n, c in enumerate(response.citations, 1)
        )
        write(
            "report.md",
            f"# IE-Agent 运行报告\n\n"
            f"运行：{run_id}\n\n模式：{response.metrics.execution_mode}\n\n"
            f"状态：{response.outcome}；停止原因：{response.stop_reason}\n\n"
            f"## 用户任务\n\n{request.query}\n\n## 回答\n\n{response.answer}\n\n"
            f"## 教材证据\n\n{references or '无教材引用'}\n\n"
            f"## 测量\n\n模型调用 {response.metrics.model_calls} 次；"
            f"输入 token {response.metrics.input_tokens}；"
            f"输出 token {response.metrics.output_tokens}；"
            f"总耗时 {response.metrics.total_latency_ms:.1f} ms。\n\n"
            f"## 提示\n\n" + "\n".join(f"- {w}" for w in response.warnings),
        )
