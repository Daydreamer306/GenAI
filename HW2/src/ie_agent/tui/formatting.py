"""终端回答中的 LaTeX 降级显示。"""

from __future__ import annotations

import re

from pylatexenc.latex2text import LatexNodes2Text

_CODE_BLOCK = re.compile(r"(```.*?```)", re.DOTALL)
_MATH_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
]
_CONVERTER = LatexNodes2Text(math_mode="text")


def latex_to_terminal_markdown(content: str) -> str:
    """用 pylatexenc 把数学片段转换为普通终端可显示的 Unicode 文本。"""

    parts = _CODE_BLOCK.split(content)
    for index in range(0, len(parts), 2):
        for pattern in _MATH_PATTERNS:
            parts[index] = pattern.sub(_convert_math, parts[index])
    return "".join(parts)


def _convert_math(match: re.Match[str]) -> str:
    return _CONVERTER.latex_to_text(match.group(1)).strip()
