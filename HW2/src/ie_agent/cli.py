"""IE-Agent 的命令行入口。"""

import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import typer

from ie_agent import __version__
from ie_agent.agent import AgentOrchestrator, PromptBuilder
from ie_agent.config import Settings
from ie_agent.contracts import AgentRequest, PlannedToolCall, PromptContext
from ie_agent.evaluation import RetrievalEvaluator
from ie_agent.knowledge import KnowledgeCleaner, KnowledgeService
from ie_agent.knowledge.vector_store import chroma_index_ready
from ie_agent.model import create_model
from ie_agent.skills import SkillLoader
from ie_agent.tools import ToolRegistry

app = typer.Typer(
    name="ie-agent",
    help="面向信息工程课程学习的本地 RAG 智能体。",
    no_args_is_help=False,
    invoke_without_command=True,
)
prompt_app = typer.Typer(help="查看提示词工程结果。", no_args_is_help=True)
knowledge_app = typer.Typer(help="处理和检索本地教材。", no_args_is_help=True)
tool_app = typer.Typer(help="查看和运行确定性课程工具。", no_args_is_help=True)
skill_app = typer.Typer(help="查看仓库内的提示 Skill。", no_args_is_help=True)
app.add_typer(prompt_app, name="prompt")
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(tool_app, name="tool")
app.add_typer(skill_app, name="skill")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """没有子命令时启动全屏终端界面。"""

    if ctx.invoked_subcommand is None:
        from ie_agent.tui import run_tui

        run_tui(Settings())


@app.command()
def tui() -> None:
    """显式启动 Textual 全屏终端界面。"""

    from ie_agent.tui import run_tui

    run_tui(Settings())


@app.command("web")
def web_server(
    host: Annotated[
        str,
        typer.Option(help="监听地址，默认只允许本机访问。"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="后端服务端口。"),
    ] = 8000,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="服务启动后使用默认浏览器打开页面。",
        ),
    ] = True,
) -> None:
    """启动 FastAPI；若已构建前端则同时提供静态页面。"""

    import webbrowser
    from threading import Timer

    import uvicorn

    from ie_agent.web import create_app

    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo("提示：当前监听地址会向局域网开放模型接口，请只在可信网络中使用。")
    if host == "0.0.0.0":
        browser_host = "127.0.0.1"
    elif host == "::":
        browser_host = "[::1]"
    elif ":" in host:
        browser_host = f"[{host}]"
    else:
        browser_host = host
    url = f"http://{browser_host}:{port}"
    typer.echo(f"网页地址：{url}")
    if open_browser:
        timer = Timer(0.8, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(create_app(Settings()), host=host, port=port)


@app.command()
def version() -> None:
    """显示当前项目版本。"""

    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """检查模型配置和本地知识库状态。"""

    settings = Settings()
    knowledge_dir = settings.knowledge_dir
    clean_ready = any((knowledge_dir / "clean").rglob("*.md"))
    model_dir = knowledge_dir / "models" / "qwen3-embedding-0.6b"
    model_ready = any(model_dir.rglob("config.json")) and any(model_dir.rglob("model.safetensors"))
    index_dir = knowledge_dir / "index"
    typer.echo(f"模型提供商：{settings.model_provider}")
    typer.echo(f"模型 API key：{'已配置' if settings.has_model_key else '未配置'}")
    typer.echo(f"模型：{settings.active_model}")
    typer.echo(f"Agent Planner：{settings.planner}")
    if settings.model_provider == "deepseek":
        typer.echo(
            "思考模式："
            f"{settings.deepseek_thinking}，推理强度：{settings.deepseek_reasoning_effort}"
        )
    typer.echo(f"清洗教材：{'已就绪' if clean_ready else '未建立'}")
    typer.echo(f"Qwen 模型缓存：{'已就绪' if model_ready else '未下载'}")
    typer.echo(
        f"Qwen 向量索引：{'已建立' if chroma_index_ready(index_dir / 'qwen_chroma') else '未建立'}"
    )
    typer.echo(
        "TF-IDF 基线索引："
        f"{'已建立' if (index_dir / 'tfidf' / 'index.pkl').is_file() else '未建立'}"
    )


@app.command()
def ask(
    question: str = typer.Argument(min=1, help="发送给已选模型的问题。"),
    planner: Annotated[
        Literal["hybrid", "rules", "json", "tool_calls"],
        typer.Option(help="选择本轮 Agent 规划方式。"),
    ] = "hybrid",
    review: Annotated[
        bool, typer.Option("--review/--no-review", help="启用独立审核与修改循环。")
    ] = True,
    max_rounds: Annotated[int, typer.Option(min=1, max=5)] = 3,
) -> None:
    """通过统一 Agent 完成一次带工具、教材和引用的问答。"""

    settings = Settings()
    response = AgentOrchestrator(settings, model=create_model(settings)).answer(
        AgentRequest(
            session_id=f"cli-{uuid4().hex[:8]}",
            turn_id=uuid4().hex,
            query=question,
            planner=planner,
            enable_review=review,
            max_rounds=max_rounds,
        )
    )
    typer.echo(response.answer)
    if response.citations:
        typer.echo("\n引用：")
        for index, citation in enumerate(response.citations, start=1):
            typer.echo(
                f"[{index}] {citation.book_title}｜{citation.section}｜"
                f"{citation.backend} {citation.score:.3f}"
            )
    if response.warnings:
        typer.echo("\n提示：" + "；".join(response.warnings), err=True)
    _print_run(response)
    if response.outcome == "failed":
        raise typer.Exit(code=2)


def _print_run(response) -> None:
    for step in response.trace:
        typer.echo(f"[{step.step_id:02d}] {step.name} | {step.status} | {step.summary}")
    typer.echo(
        f"状态：{response.outcome}；停止原因：{response.stop_reason}；轮数：{len(response.rounds)}"
    )
    typer.echo(
        f"模式：{response.metrics.execution_mode}；模型调用：{response.metrics.model_calls}；"
        f"token：{response.metrics.input_tokens}/{response.metrics.output_tokens}"
    )
    typer.echo(f"运行产物：{response.artifact_dir}")


@app.command("demo")
def demo(
    scenario: Annotated[
        Literal[
            "repair",
            "pass",
            "reject",
            "solver-error",
            "reviewer-error",
            "invalid-review",
            "tool-error",
        ],
        typer.Option(help="离线场景，模拟模型但真实执行课程工具。"),
    ] = "repair",
) -> None:
    """无需 key 的端到端演示，清楚标记为模拟模型。"""
    from ie_agent.model.demo import DemoModel

    settings = Settings()
    settings = settings.model_copy(update={"rag_backend": "tfidf"})
    args = {"probabilities": [0.5, 0.6] if scenario == "tool-error" else [0.5, 0.5]}
    query = json.dumps({"tool_name": "entropy", "arguments": args})
    response = AgentOrchestrator(settings, model=DemoModel(scenario)).answer(
        AgentRequest(
            session_id="offline-demo",
            turn_id=uuid4().hex,
            query=query,
            planner="rules",
            structured_answer=True,
        )
    )
    typer.echo(response.answer)
    _print_run(response)
    if response.outcome == "failed":
        raise typer.Exit(code=2)


@app.command("benchmark")
def benchmark(
    offline: Annotated[
        bool, typer.Option("--offline", help="模拟模型，仅验证流程，不产生 API 费用。")
    ] = False,
    limit: Annotated[int, typer.Option(min=0, help="任务数量上限；0 为全部任务。")] = 0,
    tasks: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "evaluation/mas_tasks.json"
    ),
) -> None:
    """比较同一题集上的单轮解答与审核循环，保存正确率和 token 测量。"""
    from ie_agent.mas_evaluation import run_comparison

    settings = Settings()
    if not offline and (settings.model_provider != "minimax" or not settings.has_model_key):
        typer.echo("真实实验需在本机 .env 配置 MiniMax key；离线验证请加 --offline。", err=True)
        raise typer.Exit(code=1)

    def progress(row: dict) -> None:
        typer.echo(
            f"{row['task_id']} | {row['design']} | {row['outcome']} | "
            f"gold={'通过' if row['correct'] else '未通过'} | "
            f"token={row['input_tokens']}/{row['output_tokens']}"
        )

    directory = run_comparison(settings, tasks, offline=offline, limit=limit, progress=progress)
    typer.echo(f"实验报告：{directory}")


@app.command("reproduce")
def reproduce() -> None:
    """一条命令复现离线修改循环和同题设计对比，真实工具、模拟模型。"""
    demo("repair")
    benchmark(offline=True)


@prompt_app.command("preview")
def prompt_preview(question: str = typer.Argument(min=1, help="需要预览的问题。")) -> None:
    """显示本轮将发送的消息，不调用远程模型。"""

    messages = PromptBuilder().build(PromptContext(query=question))
    for message in messages:
        typer.echo(f"[{message.role}]")
        typer.echo(message.content)
        typer.echo()


@knowledge_app.command("clean")
def knowledge_clean(
    extracted_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="MinerU 输出目录。"),
    ] = Path("knowledge/local/extracted"),
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="清洗后 Markdown 的保存目录。"),
    ] = Path("knowledge/local/clean"),
    catalog: Annotated[
        Path,
        typer.Option(help="教材正式名称和课程分类目录。"),
    ] = Path("knowledge/catalog.yaml"),
    report: Annotated[
        Path,
        typer.Option(help="质量报告的保存位置。"),
    ] = Path("knowledge/local/reports/cleaning-report.json"),
) -> None:
    """清洗 MinerU Markdown，并生成逐本质量统计。"""

    try:
        result = KnowledgeCleaner(catalog).clean(extracted_dir, output_dir, report)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"清洗失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"已清洗 {len(result.books)} 本教材，共 {result.total_input_files} 个 Markdown 分段。"
    )
    typer.echo(f"质量报告：{report}")


@knowledge_app.command("ingest")
def knowledge_ingest(
    clean_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="清洗后的教材目录。"),
    ] = Path("knowledge/local/clean"),
    backend: Annotated[
        Literal["qwen", "tfidf"],
        typer.Option(help="明确选择 Qwen 或 TF-IDF 后端。"),
    ] = "qwen",
) -> None:
    """对清洗后的教材分块并重建本地索引。"""

    settings = Settings()
    try:
        result = KnowledgeService(settings, backend=backend).ingest(clean_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"导入失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    report = settings.knowledge_dir / "reports" / f"ingest-{backend}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"后端：{result.backend}")
    typer.echo(f"教材：{result.sources} 本，片段：{result.chunks} 个")
    typer.echo(f"耗时：{result.latency_ms / 1000:.2f} 秒")
    typer.echo(f"导入报告：{report}")


@knowledge_app.command("search")
def knowledge_search(
    question: str = typer.Argument(min=1, help="需要检索的课程问题。"),
    backend: Annotated[
        Literal["qwen", "tfidf"],
        typer.Option(help="明确选择 Qwen 或 TF-IDF 后端。"),
    ] = "qwen",
    course: Annotated[
        list[str] | None,
        typer.Option("--course", "-c", help="可重复填写的课程目录名。"),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, max=20, help="最多返回的片段数。"),
    ] = 5,
) -> None:
    """直接查看教材检索结果和代码生成的引用。"""

    try:
        result = KnowledgeService(Settings(), backend=backend).search(
            question,
            course_tags=course,
            top_k=top_k,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"检索失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"后端：{result.backend}，耗时：{result.latency_ms:.1f} ms")
    if not result.chunks:
        typer.echo("未检索到达到阈值的教材片段。")
        return
    for index, chunk in enumerate(result.chunks, start=1):
        citation = chunk.citation
        location = citation.section
        if citation.page is not None:
            location += f"，第 {citation.page} 页"
        typer.echo(f"\n[{index}] {citation.book_title}｜{location}｜score={citation.score:.3f}")
        typer.echo(chunk.text[:500].strip())


@app.command("evaluate")
def evaluate(
    backend: Annotated[
        Literal["qwen", "tfidf"],
        typer.Option(help="需要评估的检索后端。"),
    ] = "qwen",
    questions: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="人工标注的问题集。"),
    ] = Path("evaluation/retrieval_questions.yaml"),
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, max=20),
    ] = 5,
) -> None:
    """在同一问题集上计算检索 Recall@K、MRR 和耗时。"""

    settings = Settings()
    evaluator = RetrievalEvaluator()
    try:
        report = evaluator.run(
            KnowledgeService(settings, backend=backend),
            evaluator.load_questions(questions),
            top_k=top_k,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"评估失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc

    report_path = settings.knowledge_dir / "reports" / f"evaluation-{backend}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    typer.echo(f"后端：{backend}，问题数：{report.questions}")
    typer.echo(f"Recall@{top_k}：{report.recall_at_k:.3f}")
    typer.echo(f"MRR：{report.mrr:.3f}")
    typer.echo(f"平均检索耗时：{report.average_latency_ms:.1f} ms")
    typer.echo(f"评估报告：{report_path}")


@tool_app.command("list")
def tool_list() -> None:
    """列出九个可审查的本地工具。"""

    for spec in ToolRegistry().list():
        typer.echo(f"{spec.name}\t{spec.description}")


@tool_app.command("run")
def tool_run(
    name: str = typer.Argument(min=1, help="工具名称。"),
    arguments: Annotated[
        str,
        typer.Option("--arguments", "-a", help="JSON 格式的工具参数。"),
    ] = "{}",
) -> None:
    """用 JSON 参数独立运行一个确定性工具。"""

    settings = Settings()
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        typer.echo(f"参数不是有效 JSON：{exc.msg}", err=True)
        raise typer.Exit(code=1) from exc
    if not isinstance(parsed, dict):
        typer.echo("工具参数必须是 JSON 对象。", err=True)
        raise typer.Exit(code=1)
    result = ToolRegistry(settings.knowledge_dir / "artifacts").run(
        PlannedToolCall(
            call_id="cli-tool",
            tool_name=name,
            arguments=parsed,
            reason="命令行直接调用",
        )
    )
    if result.status == "error":
        typer.echo(f"工具失败：{result.error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


@skill_app.command("list")
def skill_list() -> None:
    """列出项目内置的只读 Skill。"""

    for spec in SkillLoader().list():
        typer.echo(f"{spec.name}\t{spec.description}")


@skill_app.command("show")
def skill_show(name: str = typer.Argument(min=1, help="需要查看的 Skill 名称。")) -> None:
    """查看一个 Skill 的完整提示内容。"""

    try:
        content = SkillLoader().render(name)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(content)


if __name__ == "__main__":
    app()
