"""读取项目中保存的课程学习 Skill。"""

from __future__ import annotations

from importlib.resources import files

from ie_agent.contracts import SkillSpec


class SkillLoader:
    """Skill 只返回提示文字，不执行脚本或访问环境变量。"""

    _skills = {
        "formula_explanation": (
            "按公式、符号、单位、条件、步骤和含义解释课程公式。",
            "formula_explanation.md",
        ),
        "concept_explanation": (
            "结合教材解释课程概念、直观含义和相关知识。",
            "concept_explanation.md",
        ),
        "problem_solving": (
            "按已知条件、公式、代入和检查步骤完成课程习题。",
            "problem_solving.md",
        ),
        "experiment_guidance": (
            "按目的、方法、步骤、现象和分析说明课程实验。",
            "experiment_guidance.md",
        ),
    }

    def list(self) -> list[SkillSpec]:
        return [
            SkillSpec(name=name, description=description)
            for name, (description, _) in self._skills.items()
        ]

    def render(self, name: str, context: dict[str, object] | None = None) -> str:
        if name not in self._skills:
            raise ValueError(f"未知 Skill：{name}")
        _, filename = self._skills[name]
        template = files("ie_agent.skills").joinpath(filename).read_text(encoding="utf-8")
        if not context:
            return template.strip()
        details = "\n".join(f"- {key}: {value}" for key, value in sorted(context.items()))
        return f"{template.strip()}\n\n本轮已知信息：\n{details}"
