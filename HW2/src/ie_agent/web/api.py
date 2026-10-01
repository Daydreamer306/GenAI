"""供 React 页面调用的轻量 FastAPI 接口。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, field_validator

from ie_agent import __version__
from ie_agent.agent import AgentOrchestrator
from ie_agent.config import Settings
from ie_agent.contracts import AgentRequest, AgentResponse, ChatMessage, StrictModel
from ie_agent.mcp_server import create_mcp_server
from ie_agent.tools import ToolRegistry


class WebChatRequest(StrictModel):
    """浏览器发送的一轮对话。"""

    session_id: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=4_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)

    @field_validator("session_id", "query")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """拒绝看似有长度、实际只有空白的输入。"""

        value = value.strip()
        if not value:
            raise ValueError("字段不能只包含空白字符")
        return value


def create_app(
    settings: Settings | None = None,
    orchestrator: AgentOrchestrator | None = None,
    *,
    frontend_dir: Path | None = None,
) -> FastAPI:
    """创建 API；调用方可以注入固定 Agent。"""

    settings = settings or Settings()
    registry = ToolRegistry(settings.knowledge_dir / "artifacts")
    agent = orchestrator or AgentOrchestrator(settings, tools=registry)
    mcp_server = create_mcp_server(
        settings,
        registry=registry,
        streamable_http_path="/",
    )
    mcp_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with mcp_server.session_manager.run():
            yield

    api = FastAPI(
        title="IE-Agent Web API",
        version=__version__,
        description="React 页面与 TUI 共用的课程学习 Agent 接口。",
        lifespan=lifespan,
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @api.post("/api/chat", response_model=AgentResponse)
    async def chat(payload: WebChatRequest) -> AgentResponse:
        request = AgentRequest(
            session_id=payload.session_id,
            turn_id=uuid4().hex,
            query=payload.query.strip(),
            history=payload.history,
            mcp_enabled=True,
        )
        try:
            return await run_in_threadpool(agent.answer, request)
        except Exception as exc:  # API 边界不返回第三方库的内部错误
            raise HTTPException(
                status_code=503,
                detail="当前问答服务未完成请求，请检查本地配置后重试。",
            ) from exc

    @api.get("/api/artifacts/{name}", response_class=FileResponse)
    def artifact(name: str) -> FileResponse:
        artifact_dir = (settings.knowledge_dir / "artifacts").resolve()
        requested = (artifact_dir / name).resolve()
        if not requested.is_relative_to(artifact_dir) or not requested.is_file():
            raise HTTPException(status_code=404, detail="图像不存在。")
        return FileResponse(requested)

    # MCP 与网页 API 在同一个 FastAPI 进程中运行。
    api.mount("/mcp", mcp_app, name="mcp")

    # 非编辑安装时 __file__ 位于虚拟环境中，因此从启动目录寻找前端构建结果。
    dist_dir = frontend_dir if frontend_dir is not None else Path.cwd() / "web" / "dist"
    if dist_dir.is_dir() and (dist_dir / "index.html").is_file():
        assets_dir = dist_dir / "assets"
        if assets_dir.is_dir():
            api.mount("/assets", StaticFiles(directory=assets_dir), name="web-assets")

        @api.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            if path.startswith(("api/", "mcp/")):
                raise HTTPException(status_code=404, detail="接口不存在。")
            requested = (dist_dir / path).resolve()
            if requested.is_relative_to(dist_dir.resolve()) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(dist_dir / "index.html")

    return api


app = create_app()
