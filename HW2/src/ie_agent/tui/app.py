"""IE-Agent 的 Textual 全屏终端界面。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    SelectionList,
    Static,
)
from textual_image.widget import Image as TerminalImage

from ie_agent.agent import AgentOrchestrator
from ie_agent.config import Settings
from ie_agent.contracts import (
    AgentRequest,
    AgentResponse,
    ChatMessage,
    SkillSpec,
    ToolArtifact,
)
from ie_agent.tui.formatting import latex_to_terminal_markdown


@dataclass
class CapabilitySelection:
    """TUI 中一组可切换的 Agent 能力。"""

    planner: str
    enabled_skills: list[str]
    enable_rag: bool = True
    enable_tools: bool = True
    enable_mcp: bool = True


class ChatBubble(Vertical):
    """显示一条带角色标签的对话消息。"""

    def __init__(self, role: str, content: str) -> None:
        super().__init__(classes=f"chat-bubble {role}")
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        role_name = "你" if self.role == "user" else "IE-Agent"
        yield Label(role_name, classes="bubble-role")
        content = self.content
        if self.role == "assistant":
            content = latex_to_terminal_markdown(content)
        yield Markdown(content, classes="bubble-content")


class ChatArtifact(Vertical):
    """在回答下方显示工具生成的图片和项目相对路径。"""

    def __init__(self, artifact: ToolArtifact) -> None:
        super().__init__(classes="chat-artifact")
        self.artifact = artifact

    def compose(self) -> ComposeResult:
        path = Path(self.artifact.path)
        yield Label(self.artifact.title, classes="artifact-title")
        if path.is_file():
            yield TerminalImage(path, classes="artifact-image")
        else:
            yield Static("图片文件不存在，请检查下方路径。", classes="artifact-missing")
        yield Static(f"保存位置：{self.artifact.path}", classes="artifact-path")


class HelpScreen(ModalScreen[None]):
    """显示快捷键和主要操作说明。"""

    BINDINGS = [Binding("escape", "close", "关闭", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Markdown(
                """
# IE-Agent 使用帮助

- 在底部输入问题，按 **Enter** 或点击“发送”。
- `F2` 选择回答方式，并开关教材、工具、Skill 和 MCP。
- `Ctrl+L` 清空会话，`Ctrl+E` 显示或隐藏证据区。
- `F1` 打开本帮助，`Ctrl+Q` 退出程序。
- 证据区会按顺序列出实际调用路径、工具参数、教材引用和降级提示。
                """.strip()
            )
            yield Button("返回对话", id="close-help", variant="primary")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close-help":
            self.dismiss(None)


class CapabilityScreen(ModalScreen[CapabilitySelection | None]):
    """选择本轮问答允许使用的能力。"""

    BINDINGS = [
        Binding("ctrl+s", "save", "应用", show=False),
        Binding("escape", "cancel", "取消", show=False),
    ]

    def __init__(self, selection: CapabilitySelection, skills: list[SkillSpec]) -> None:
        super().__init__()
        self.selection = selection
        self.skills = skills

    def compose(self) -> ComposeResult:
        with Vertical(id="capability-dialog"):
            yield Label("功能设置", classes="dialog-title")
            yield Label(
                "方向键选择，空格切换；勾选表示已开启",
                classes="dialog-hint",
            )
            yield Label("回答方式", classes="field-title")
            yield Select(
                [
                    ("自动选择（推荐）", "hybrid"),
                    ("只使用本地规则", "rules"),
                    ("模型 JSON 规划", "json"),
                    ("模型工具规划", "tool_calls"),
                ],
                allow_blank=False,
                value=self.selection.planner,
                id="planner-select",
            )
            yield Label("可用功能", classes="field-title")
            yield SelectionList[str](
                ("本地教材检索", "rag", self.selection.enable_rag),
                ("Python 计算与绘图", "tools", self.selection.enable_tools),
                ("学习辅助 Skill", "skills", bool(self.selection.enabled_skills)),
                ("MCP 工具调用（同进程）", "mcp", self.selection.enable_mcp),
                id="capability-list",
            )
            with Horizontal(id="capability-actions"):
                yield Button("全部开启", id="enable-all-capabilities")
                yield Button("应用设置", id="save-capabilities", variant="primary")
                yield Button("取消", id="cancel-capabilities")

    def on_mount(self) -> None:
        self.query_one("#planner-select", Select).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "enable-all-capabilities":
            self.query_one("#capability-list", SelectionList).select_all()
            return
        if event.button.id == "cancel-capabilities":
            self.action_cancel()
            return
        if event.button.id == "save-capabilities":
            self.action_save()

    def action_save(self) -> None:
        planner_value = self.query_one("#planner-select", Select).value
        enabled = set(self.query_one("#capability-list", SelectionList).selected)
        enabled_skills = [skill.name for skill in self.skills] if "skills" in enabled else []
        self.dismiss(
            CapabilitySelection(
                planner=str(planner_value),
                enabled_skills=enabled_skills,
                enable_rag="rag" in enabled,
                enable_tools="tools" in enabled,
                enable_mcp="mcp" in enabled,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class IEAgentApp(App[None]):
    """把模型、RAG 和工具结果组织为可交互的终端应用。"""

    CSS_PATH = "theme.tcss"
    TITLE = "IE-Agent"
    SUB_TITLE = "人工智能前沿课程设计"

    BINDINGS = [
        Binding("ctrl+q", "quit", "退出", priority=True),
        Binding("ctrl+l", "clear_chat", "清空", priority=True),
        Binding("ctrl+e", "toggle_evidence", "证据", priority=True),
        Binding("f1", "show_help", "帮助"),
        Binding("f2", "manage_capabilities", "能力"),
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        orchestrator: AgentOrchestrator | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or Settings()
        self.orchestrator = orchestrator or AgentOrchestrator(self.settings)
        self.session_id = f"tui-{uuid4().hex[:10]}"
        self.history: list[ChatMessage] = []
        self.last_response: AgentResponse | None = None
        self._busy = False
        self._evidence_visible = True
        self.capabilities = CapabilitySelection(
            planner=self.settings.planner,
            enabled_skills=[item.name for item in self.orchestrator.skills.list()],
        )

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self._system_status_text(), id="system-status")
        with Horizontal(id="workspace"):
            with Vertical(id="chat-panel"):
                yield Label("课程问答", classes="panel-title")
                with VerticalScroll(id="chat-log"):
                    yield ChatBubble(
                        "assistant",
                        "输入课程问题，我会在需要时查教材、调用 Python 工具，"
                        "并在右侧给出实际调用路径。",
                    )
            with Vertical(id="evidence-panel"):
                yield Label("本轮证据", classes="panel-title")
                with VerticalScroll(id="evidence-scroll"):
                    yield Markdown(
                        "尚未提问。回答后这里会显示完整调用路径。",
                        id="evidence-content",
                    )
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="输入课程问题，例如：计算 sqrt(3^2+4^2) 或绘制 y=sin(x)",
                id="question-input",
            )
            yield Button("发送", id="send-button", variant="primary")
        with Horizontal(id="task-row"):
            yield LoadingIndicator(id="loading")
            yield Static("就绪", id="task-status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#loading", LoadingIndicator).display = False
        self.query_one("#question-input", Input).focus()
        self._apply_responsive_layout(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        self._apply_responsive_layout(event.size.width)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "question-input":
            self._submit_question(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-button":
            value = self.query_one("#question-input", Input).value
            self._submit_question(value)

    def _submit_question(self, value: str) -> None:
        question = value.strip()
        if not question:
            self.notify("请先输入一个问题。", severity="warning")
            return
        if self._busy:
            self.notify("上一轮仍在处理中，请稍候。", severity="warning")
            return

        request = AgentRequest(
            session_id=self.session_id,
            turn_id=uuid4().hex,
            query=question,
            history=self.history[-8:],
            planner=self.capabilities.planner,
            enabled_skills=self.capabilities.enabled_skills,
            enable_rag=self.capabilities.enable_rag,
            enable_tools=self.capabilities.enable_tools,
            mcp_enabled=self.capabilities.enable_mcp,
        )
        self.history.append(ChatMessage(role="user", content=question))
        self._append_chat("user", question)
        question_input = self.query_one("#question-input", Input)
        question_input.value = ""
        self._set_busy(True, "正在路由、检索并生成回答……")
        self._answer_in_background(request)

    @work(thread=True, exclusive=True, group="agent-answer", exit_on_error=False)
    def _answer_in_background(self, request: AgentRequest) -> None:
        """同步模型调用放到线程中，防止阻塞终端刷新。"""

        try:
            response = self.orchestrator.answer(request)
        except Exception as exc:  # 界面边界需要兜住第三方库异常
            self.call_from_thread(self._render_error, type(exc).__name__)
            return
        self.call_from_thread(self._render_response, response)

    def _render_response(self, response: AgentResponse) -> None:
        self.last_response = response
        self.history.append(ChatMessage(role="assistant", content=response.answer))
        self._append_chat("assistant", response.answer)
        self._append_artifacts(response)
        self.query_one("#evidence-content", Markdown).update(self._format_evidence(response))
        self._set_busy(False, self._completion_status(response))

    def _render_error(self, error_name: str) -> None:
        answer = f"本轮执行失败（{error_name}）。请检查本地配置或稍后重试。"
        self.history.append(ChatMessage(role="assistant", content=answer))
        self._append_chat("assistant", answer)
        self.query_one("#evidence-content", Markdown).update(
            "### 执行提示\n\n本轮没有得到可用结果，界面未展示异常详情或敏感配置。"
        )
        self._set_busy(False, "执行失败")

    def _append_chat(self, role: str, content: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(ChatBubble(role, content))
        self.call_after_refresh(chat_log.scroll_end, animate=False)

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self.query_one("#question-input", Input).disabled = busy
        self.query_one("#send-button", Button).disabled = busy
        self.query_one("#loading", LoadingIndicator).display = busy
        self.query_one("#task-status", Static).update(status)
        if not busy:
            self.query_one("#question-input", Input).focus()

    def action_clear_chat(self) -> None:
        if self._busy:
            self.notify("当前回答完成后再清空会话。", severity="warning")
            return
        self.history.clear()
        self.last_response = None
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.remove_children()
        chat_log.mount(ChatBubble("assistant", "会话已经清空，可以开始新的问题。"))
        self.query_one("#evidence-content", Markdown).update("尚未提问。")
        self.query_one("#task-status", Static).update("会话已清空")

    def action_toggle_evidence(self) -> None:
        self._evidence_visible = not self._evidence_visible
        self.query_one("#evidence-panel", Vertical).display = self._evidence_visible
        state = "显示" if self._evidence_visible else "隐藏"
        self.notify(f"证据区已{state}。")

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_manage_capabilities(self) -> None:
        if self._busy:
            self.notify("当前回答完成后再调整能力。", severity="warning")
            return
        screen = CapabilityScreen(
            self.capabilities,
            self.orchestrator.skills.list(),
        )
        self.push_screen(screen, self._apply_capabilities)

    def _apply_capabilities(self, selection: CapabilitySelection | None) -> None:
        if selection is None:
            return
        self.capabilities = selection
        self.notify("功能设置已应用。")
        self.query_one("#system-status", Static).update(self._system_status_text())
        if self.last_response is None:
            self.query_one("#evidence-content", Markdown).update(self._format_capability_summary())

    def _apply_responsive_layout(self, width: int) -> None:
        workspace = self.query_one("#workspace", Horizontal)
        workspace.set_class(width < 96, "narrow")

    def _system_status_text(self) -> str:
        state = "模型已配置" if self.settings.has_model_key else "模型未配置"
        enabled = []
        if self.capabilities.enable_rag:
            enabled.append("教材")
        if self.capabilities.enable_tools:
            enabled.append("计算绘图")
        if self.capabilities.enabled_skills:
            enabled.append("学习辅助")
        if self.capabilities.enable_mcp:
            enabled.append("MCP")
        return f"{state} · 已启用：{'、'.join(enabled) or '仅对话'} · F2 设置"

    @staticmethod
    def _completion_status(response: AgentResponse) -> str:
        source_count = len(response.citations)
        tool_count = len(response.tool_results)
        return (
            f"{response.outcome} · {len(response.rounds)} 轮 · "
            f"{response.model_result.latency_ms:.0f} ms · "
            f"{source_count} 条引用 · {tool_count} 个工具"
        )

    @staticmethod
    def _format_evidence(response: AgentResponse) -> str:
        blocks = [
            f"### 审核状态：{response.outcome}",
            f"停止原因：{response.stop_reason}",
            f"运行产物：{response.artifact_dir}",
            "### 本轮调用路径",
        ]
        status_names = {"success": "成功", "error": "失败", "skipped": "跳过"}
        for step in response.trace:
            latency = f"，{step.latency_ms:.1f} ms" if step.latency_ms else ""
            blocks.append(
                f"{step.step_id}. **{step.name}** · `{step.stage}` · "
                f"{status_names[step.status]}{latency}  \n   {step.summary}"
            )
            visible_keys = {
                "planning": ("intent", "skills", "tools"),
                "rag": ("backend", "hits"),
                "tool": ("executor", "arguments", "result", "artifacts"),
            }.get(step.stage, ())
            metadata = {key: step.metadata[key] for key in visible_keys if key in step.metadata}
            if metadata:
                payload = json.dumps(metadata, ensure_ascii=False, default=str)
                blocks.append(f"   `数据：{payload}`")

        blocks.extend(
            [
                "\n### 本轮决策",
                f"- 回答方式：`{response.route.planner}`",
                f"- 学习辅助：{', '.join(response.route.skill_names) or '无'}",
                f"- 原因：{response.route.reason}",
            ]
        )
        if response.tool_results:
            blocks.append("\n### 本地 Python 工具")
            for result in response.tool_results:
                state = "成功" if result.status == "success" else "失败"
                formula = f"；公式：{result.formula}" if result.formula else ""
                blocks.append(f"\n#### `{result.tool_name}`")
                blocks.append(f"- 状态：{state}{formula}")
                if result.result:
                    payload = json.dumps(
                        result.result,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
                    blocks.extend(["- 结构化结果：", "```json", payload, "```"])
                if result.steps:
                    blocks.append("- 计算步骤：")
                    blocks.extend(
                        f"  {index}. {step}" for index, step in enumerate(result.steps, start=1)
                    )
                if result.error:
                    blocks.append(f"- 错误：{result.error}")
                if result.artifacts:
                    paths = "、".join(item.path for item in result.artifacts)
                    blocks.append(f"- 保存位置：{paths}")
        if response.citations:
            blocks.append("\n### 教材引用")
            for index, citation in enumerate(response.citations, start=1):
                page = f"，第 {citation.page} 页" if citation.page else ""
                blocks.append(
                    f"{index}. {citation.book_title}｜{citation.section}{page}｜"
                    f"{citation.backend} {citation.score:.3f}"
                )
        if response.warnings:
            blocks.append("\n### 系统提示")
            blocks.extend(f"- {warning}" for warning in response.warnings)
        return "\n".join(blocks)

    def _append_artifacts(self, response: AgentResponse) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        for result in response.tool_results:
            for artifact in result.artifacts:
                chat_log.mount(ChatArtifact(artifact))
        self.call_after_refresh(chat_log.scroll_end, animate=False)

    def _format_capability_summary(self) -> str:
        return (
            "### 当前功能\n\n"
            f"- 回答方式：`{self.capabilities.planner}`\n"
            f"- 本地教材：{'开启' if self.capabilities.enable_rag else '关闭'}\n"
            f"- Python 计算绘图：{'开启' if self.capabilities.enable_tools else '关闭'}\n"
            f"- 学习辅助：{'开启' if self.capabilities.enabled_skills else '关闭'}\n"
            f"- MCP 调用：{'开启' if self.capabilities.enable_mcp else '关闭'}"
        )


def run_tui(settings: Settings | None = None) -> None:
    """运行全屏终端界面。"""

    from threading import RLock

    from tqdm import tqdm
    from transformers.utils.logging import disable_progress_bar

    # Textual 在后台线程中加载 Qwen，使用线程锁可避免 tqdm 创建多进程锁。
    tqdm.set_lock(RLock())
    disable_progress_bar()
    IEAgentApp(settings=settings).run()
