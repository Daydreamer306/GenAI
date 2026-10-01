"""Agent 编排相关组件。"""

from importlib import import_module

__all__ = [
    "AgentOrchestrator",
    "AgentPlanner",
    "BaselinePromptBuilder",
    "IntentRouter",
    "PromptBuilder",
]

_MODULES = {
    "AgentOrchestrator": "orchestrator",
    "AgentPlanner": "planner",
    "BaselinePromptBuilder": "prompt",
    "IntentRouter": "router",
    "PromptBuilder": "prompt",
}


def __getattr__(name: str):
    """角色、审核门单测不需要导入数据库、MCP 或科学计算依赖。"""
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(import_module(f"ie_agent.agent.{_MODULES[name]}"), name)
