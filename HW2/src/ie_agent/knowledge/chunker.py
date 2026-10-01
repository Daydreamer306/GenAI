"""按照教材标题和段落切分 Markdown。"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from ie_agent.contracts import KnowledgeChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；.!?;])")


class MarkdownChunker:
    """生成适合教材问答的稳定分块。"""

    def __init__(self, target_size: int = 800, max_size: int = 1000, overlap: int = 120):
        if not 0 <= overlap < target_size <= max_size:
            raise ValueError("分块长度必须满足 0 <= overlap < target_size <= max_size")
        self.target_size = target_size
        self.max_size = max_size
        self.overlap = overlap

    def chunk_directory(self, clean_dir: Path) -> list[KnowledgeChunk]:
        """读取清洗后的全部 Markdown，并去除同一来源内的重复片段。"""

        chunks: list[KnowledgeChunk] = []
        seen_hashes: set[tuple[str, str]] = set()
        for path in sorted(clean_dir.rglob("*.md")):
            for chunk in self.chunk_file(path):
                dedupe_key = (chunk.source_id, chunk.content_hash)
                if dedupe_key in seen_hashes:
                    continue
                seen_hashes.add(dedupe_key)
                chunks.append(chunk)
        return chunks

    def chunk_file(self, path: Path) -> list[KnowledgeChunk]:
        """切分单本带 YAML 元数据的教材。"""

        metadata, body = self._read_document(path)
        source_id = str(metadata.get("source_id", "")).strip()
        book_title = str(metadata.get("book_title", "")).strip()
        course = str(metadata.get("course", "")).strip()
        if not source_id or not book_title or not course:
            raise ValueError(f"教材元数据不完整：{path}")

        chunks: list[KnowledgeChunk] = []
        source_index = 0
        for section, section_text in self._collect_sections(body):
            for content in self._split_section(section_text):
                content = content.strip()
                if not content:
                    continue
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                identity = f"{source_id}|{section}|{source_index}|{content_hash}"
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source_id=source_id,
                        book_title=book_title,
                        course_tags=[course],
                        section=section,
                        chunk_index=source_index,
                        text=content,
                        content_hash=content_hash,
                    )
                )
                source_index += 1
        return chunks

    @staticmethod
    def _read_document(path: Path) -> tuple[dict[str, object], str]:
        text = path.read_text(encoding="utf-8", errors="strict")
        if not text.startswith("---\n"):
            raise ValueError(f"清洗后的 Markdown 缺少 YAML 头：{path}")
        end = text.find("\n---\n", 4)
        if end < 0:
            raise ValueError(f"YAML 头没有结束标记：{path}")
        metadata = yaml.safe_load(text[4:end]) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"YAML 头格式错误：{path}")
        return metadata, text[end + 5 :]

    @staticmethod
    def _collect_sections(body: str) -> list[tuple[str, str]]:
        """保留标题层级，让引用可以定位到最近的小节。"""

        heading_stack: list[str] = []
        current_lines: list[str] = []
        sections: list[tuple[str, str]] = []

        def flush() -> None:
            content = "\n".join(current_lines).strip()
            if content:
                section = " / ".join(heading_stack) if heading_stack else "正文"
                sections.append((section, content))
            current_lines.clear()

        for line in body.splitlines():
            match = _HEADING.match(line)
            if not match:
                current_lines.append(line)
                continue
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            heading_stack.append(title)
        flush()
        return sections

    def _split_section(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(self._split_long_paragraph(paragraph))

        chunks: list[str] = []
        current = ""
        for unit in units:
            separator = "\n\n" if current else ""
            if current and len(current) + len(separator) + len(unit) > self.max_size:
                chunks.append(current)
                tail = self._overlap_tail(current)
                if tail and len(tail) + 2 + len(unit) <= self.max_size:
                    current = f"{tail}\n\n{unit}"
                else:
                    current = unit
            else:
                current = f"{current}{separator}{unit}"

            if len(current) >= self.target_size:
                chunks.append(current)
                current = self._overlap_tail(current)

        if current:
            is_only_overlap = (
                bool(chunks) and len(current) <= self.overlap and chunks[-1].endswith(current)
            )
            if not is_only_overlap:
                chunks.append(current)
        return chunks

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        if len(paragraph) <= self.max_size:
            return [paragraph]
        sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(paragraph) if item.strip()]
        if len(sentences) == 1:
            return [
                paragraph[start : start + self.max_size]
                for start in range(0, len(paragraph), self.max_size)
            ]

        units: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self.max_size:
                if current:
                    units.append(current)
                    current = ""
                units.extend(
                    sentence[start : start + self.max_size]
                    for start in range(0, len(sentence), self.max_size)
                )
            elif current and len(current) + len(sentence) > self.max_size:
                units.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            units.append(current)
        return units

    def _overlap_tail(self, text: str) -> str:
        if self.overlap == 0:
            return ""
        tail = text[-self.overlap :]
        boundary = max(tail.rfind("。"), tail.rfind("\n"))
        return tail[boundary + 1 :].strip() if boundary >= 0 else tail.strip()
