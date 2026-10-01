"""可解释的关键词路由和常见参数提取。"""

from __future__ import annotations

import re

from ie_agent.contracts import PlannedToolCall, RouteDecision
from ie_agent.tools.digital_logic import BooleanParser

_COURSE_KEYWORDS = {
    "artificial_intelligence": ("人工智能", "智能体", "搜索算法", "启发式", "机器学习"),
    "signals_and_systems": ("信号", "系统", "卷积", "傅里叶", "采样", "dft"),
    "information_theory": ("信息论", "信息熵", "互信息", "信道", "编码"),
    "digital_systems": ("数字电路", "数字系统", "逻辑函数", "真值表", "卡诺图", "触发器"),
    "discrete_mathematics": ("离散数学", "命题逻辑", "集合", "图论", "关系"),
    "electromagnetics": ("电磁场", "电磁波", "高斯定理", "麦克斯韦"),
    "stochastic_processes": ("随机过程", "马尔可夫", "平稳过程", "泊松过程"),
}
_FORMULA_WORDS = ("解释公式", "公式含义", "符号含义", "推导", "适用条件")
_KNOWLEDGE_WORDS = ("根据教材", "教材中", "查找资料", "引用来源", "出处")
_EXPERIMENT_WORDS = ("实验目的", "实验步骤", "实验设计", "实验方案", "误差分析")
_PLOT_WORDS = ("画出", "绘制", "画图", "作图", "画一下", "函数图像", "曲线")
_CALCULATION_WORDS = ("计算", "求值", "算出", "算一下", "帮我算")
_NUMBER_UNIT = re.compile(
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"(MOhm|kOhm|Ohm|MΩ|kΩ|Ω|kV|mV|uV|μV|µV|V|kA|mA|uA|μA|µA|A|"
    r"mF|uF|μF|µF|nF|pF|F)(?![A-Za-z])",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class IntentRouter:
    """只在能够可靠提取参数时创建工具调用。"""

    def route(self, query: str) -> RouteDecision:
        text = query.strip()
        lower = text.lower()
        course_tags = [
            course
            for course, keywords in _COURSE_KEYWORDS.items()
            if any(keyword.lower() in lower for keyword in keywords)
        ]
        tool_call = self._tool_call(text, lower)
        formula = any(word in text for word in _FORMULA_WORDS)
        force_knowledge = any(word in text for word in _KNOWLEDGE_WORDS)
        experiment = any(word in text for word in _EXPERIMENT_WORDS)

        if tool_call is not None:
            skills = ["problem_solving"]
            if formula:
                skills.append("formula_explanation")
            return RouteDecision(
                intent="calculation",
                course_tags=course_tags,
                need_rag=force_knowledge,
                tool_calls=[tool_call],
                skill_names=skills,
                reason=f"识别到可校验参数，调用 {tool_call.tool_name}。",
            )
        if experiment:
            return RouteDecision(
                intent="knowledge",
                course_tags=course_tags,
                need_rag=bool(course_tags) or force_knowledge,
                skill_names=["experiment_guidance"],
                reason="问题要求说明实验过程，启用实验指导 Skill。",
            )
        if formula:
            return RouteDecision(
                intent="formula",
                course_tags=course_tags,
                need_rag=bool(course_tags),
                skill_names=["formula_explanation"],
                reason="问题要求解释公式，启用公式说明 Skill。",
            )
        if force_knowledge:
            return RouteDecision(
                intent="knowledge",
                course_tags=course_tags,
                need_rag=True,
                skill_names=["concept_explanation"],
                reason="用户明确要求教材证据或引用。",
            )
        if course_tags:
            return RouteDecision(
                intent="chat",
                course_tags=course_tags,
                need_rag=True,
                skill_names=["concept_explanation"],
                reason="识别到课程概念，优先检索本地教材。",
            )
        if any(word in lower for word in ("你好", "谢谢", "hello", "hi")):
            return RouteDecision(intent="chat", reason="普通对话不需要工具或教材。")
        return RouteDecision(
            intent="unknown",
            reason="没有足够信息安全选择专业工具，按普通问答处理。",
        )

    def _tool_call(self, text: str, lower: str) -> PlannedToolCall | None:
        if "真值表" in text:
            expression = self._logic_expression(text)
            if expression and self._valid_logic_expression(expression):
                return self._call("truth_table", {"expression": expression}, "生成真值表")

        sequences = self._sequences(text)
        if "卷积" in text and len(sequences) >= 2:
            return self._call(
                "discrete_convolution",
                {"x": sequences[0], "h": sequences[1]},
                "计算离散卷积",
            )
        if ("dft" in lower or "离散傅里叶" in text) and sequences:
            return self._call("dft_basic", {"sequence": sequences[0]}, "计算短序列 DFT")
        entropy_requested = "信息熵" in text or "信源熵" in text or "entropy" in lower
        equal_binary_source = re.search(
            r"(?:等概率.{0,4}(?:二元|二进制)|(?:二元|二进制).{0,4}等概率)",
            text,
        )
        negated_equal_probability = re.search(
            r"(?:并不是|不是|并非|非|不)\s*等概率|不等概率",
            text,
        )
        if (
            entropy_requested
            and not sequences
            and equal_binary_source
            and not negated_equal_probability
        ):
            # “等概率二元信源”有唯一可确定的概率分布，不需要模型猜测。
            sequences = [[0.5, 0.5]]
        if entropy_requested and sequences:
            base_match = re.search(r"(?:base\s*=|以)\s*([\d.]+)", lower)
            base = float(base_match.group(1)) if base_match else 2.0
            return self._call(
                "entropy",
                {"probabilities": sequences[0], "base": base},
                "计算离散信源熵",
            )

        measurements = self._measurements(text)
        if ("时间常数" in text or re.search(r"\brc\b", lower)) and {
            "resistance",
            "capacitance",
        }.issubset(measurements):
            return self._call(
                "rc_time_constant",
                {
                    "resistance": measurements["resistance"],
                    "capacitance": measurements["capacitance"],
                },
                "计算 RC 时间常数",
            )
        electrical = {
            key: value
            for key, value in measurements.items()
            if key in {"voltage", "current", "resistance"}
        }
        if len(electrical) == 2 and (
            "欧姆" in text or self._asks_for_missing_electrical_value(text, electrical)
        ):
            return self._call("ohms_law", electrical, "根据欧姆定律计算未知量")

        plot_requested = any(word in text for word in _PLOT_WORDS)
        if plot_requested and sequences and "序列" in text:
            return self._call(
                "plot_sequence",
                {
                    "values": sequences[0],
                    "start_index": self._sequence_start_index(text),
                },
                "绘制有限离散序列",
            )
        expression = self._math_expression(text)
        if plot_requested and expression:
            x_min, x_max = self._plot_range(text)
            return self._call(
                "plot_function",
                {
                    "expression": expression,
                    "x_min": x_min,
                    "x_max": x_max,
                },
                "绘制一元函数图像",
            )
        if any(word in text for word in _CALCULATION_WORDS) and expression:
            return self._call(
                "scientific_calculator",
                {"expression": expression},
                "计算数学表达式",
            )
        return None

    @staticmethod
    def _valid_logic_expression(expression: str) -> bool:
        """复用真值表解析器，避免把“真值表是什么”当作表达式。"""

        try:
            BooleanParser(expression).parse()
        except ValueError:
            return False
        return True

    @staticmethod
    def _asks_for_missing_electrical_value(
        text: str,
        measurements: dict[str, dict[str, object]],
    ) -> bool:
        """只有问题明确询问缺少的电学量时才使用欧姆定律。"""

        target_words = {
            "voltage": ("求电压", "电压是多少", "电压为多少"),
            "current": ("求电流", "电流是多少", "电流为多少"),
            "resistance": ("求电阻", "电阻是多少", "电阻为多少"),
        }
        return any(
            quantity not in measurements and any(word in text for word in words)
            for quantity, words in target_words.items()
        )

    @staticmethod
    def _call(name: str, arguments: dict[str, object], reason: str) -> PlannedToolCall:
        return PlannedToolCall(
            call_id="tool-1",
            tool_name=name,
            arguments=arguments,
            reason=reason,
        )

    @staticmethod
    def _sequences(text: str) -> list[list[float]]:
        sequences: list[list[float]] = []
        for content in re.findall(r"[\[【]([^\]】]+)[\]】]", text):
            values = [float(item) for item in _NUMBER.findall(content)]
            if values:
                sequences.append(values)
        return sequences

    @staticmethod
    def _measurements(text: str) -> dict[str, dict[str, object]]:
        found: dict[str, dict[str, object]] = {}
        for match in _NUMBER_UNIT.finditer(text):
            value = float(match.group(1))
            unit = match.group(2)
            normalized = unit.replace("μ", "u").replace("µ", "u").lower()
            if normalized.endswith("v"):
                quantity = "voltage"
            elif normalized.endswith("a"):
                quantity = "current"
            elif normalized.endswith("ohm") or normalized.endswith("ω"):
                quantity = "resistance"
            elif normalized.endswith("f"):
                quantity = "capacitance"
            else:
                continue
            found[quantity] = {"value": value, "unit": unit}
        return found

    @staticmethod
    def _math_expression(text: str) -> str | None:
        """从明确的计算或绘图请求中提取 ASCII 数学表达式。"""

        quoted = re.search(
            r"[\"\u201c\u201d'\u2018\u2019`]([^\"\u201c\u201d'\u2018\u2019`]+)[\"\u201c\u201d'\u2018\u2019`]",
            text,
        )
        candidates = [quoted.group(1)] if quoted else []
        expression = r"([A-Za-z0-9_+\-*/^().,\s]+)"
        patterns = [
            rf"(?:y\s*=\s*){expression}",
            rf"(?:函数|表达式|算式)\s*(?:y\s*=\s*)?[：:]?\s*{expression}",
            rf"(?:画出|绘制|画图|作图|画一下)\s*(?:函数|曲线)?\s*(?:y\s*=\s*)?{expression}",
            rf"(?:计算|求值|算出|算一下|帮我算(?:一下)?)\s*(?:数学|信号)?\s*(?:表达式|算式)?\s*[：:]?\s*{expression}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidates.append(match.group(1))
        for candidate in candidates:
            value = candidate.strip().rstrip(".,")
            if value and re.search(r"[A-Za-z0-9]", value):
                return value
        return None

    @staticmethod
    def _plot_range(text: str) -> tuple[float, float]:
        number = r"([-+]?\d+(?:\.\d+)?)"
        match = re.search(
            rf"(?:在|从)\s*x?\s*{number}\s*(?:到|至|~|～)\s*{number}",
            text,
            re.IGNORECASE,
        )
        if match is None:
            match = re.search(
                rf"(?:区间|x\s*[∈=])\s*[\[【(（]\s*{number}\s*[,，]\s*{number}",
                text,
                re.IGNORECASE,
            )
        if match:
            x_min, x_max = float(match.group(1)), float(match.group(2))
            if x_min < x_max:
                return x_min, x_max
        return -10.0, 10.0

    @staticmethod
    def _sequence_start_index(text: str) -> int:
        match = re.search(r"(?:从\s*)?n\s*=\s*(-?\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _logic_expression(text: str) -> str:
        quoted = re.search(r"[\"“”'‘’`]([^\"“”'‘’`]+)[\"“”'‘’`]", text)
        if quoted:
            return quoted.group(1).strip()

        before, after = text.split("真值表", 1)
        suffix = after.lstrip("：:，, ").rstrip("。？? ")
        if suffix:
            return suffix

        # 同时支持“真值表：A and B”和“生成 A and B 的真值表”。
        prefix = re.sub(r"^[：:，,。？?\s]+|[：:，,。？?\s]+$", "", before)
        prefix = re.sub(
            r"^(?:(?:请你|请|帮我|生成|列出|绘制|给出|制作|求|计算)\s*)+",
            "",
            prefix,
        )
        prefix = re.sub(r"(?:对应的|的)\s*$", "", prefix)
        return prefix.strip()
