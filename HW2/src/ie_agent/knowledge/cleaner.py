"""清洗 MinerU 生成的 Markdown。

本模块只做格式整理，不读取 PDF，也不会改写教材中的知识内容。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import Field

from ie_agent.contracts import StrictModel


class CatalogBook(StrictModel):
    """一本教材的人工登记信息。"""

    source_id: str = Field(min_length=1)
    course: str = Field(min_length=1)
    course_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    pages: int | None = Field(default=None, ge=1)


class BookQuality(StrictModel):
    """单本教材的清洗统计。"""

    source_id: str
    course: str
    title: str
    input_files: list[str]
    output_file: str
    content_sha256: str
    input_characters: int = Field(ge=0)
    output_characters: int = Field(ge=0)
    headings: int = Field(ge=0)
    formulas: int = Field(ge=0)
    images: int = Field(ge=0)
    removed_noise_lines: int = Field(ge=0)
    warnings: list[str]


class CleaningReport(StrictModel):
    """一次批量清洗的汇总报告。"""

    generated_at: str
    extractor: str = "MinerU precision extract"
    books: list[BookQuality]

    @property
    def total_input_files(self) -> int:
        return sum(len(book.input_files) for book in self.books)


_PART_PATTERN = re.compile(r"part-(\d+)-(\d+)$")
_EMPTY_TAG_PATTERN = re.compile(
    r"^\s*<(?:p|div|span|section|article)(?:\s+[^>]*)?>\s*</(?:p|div|span|section|article)>\s*$",
    re.IGNORECASE,
)
_IMAGE_PATTERN = re.compile(r"(!\[[^\]]*\]\()([^\s)]+)(\))")
_URL_ONLY_PATTERN = re.compile(
    r"^\s*(?:https?://)?(?:www\.)?[a-z0-9.-]+\.(?:com|cn|net|org)(?:/\S*)?\s*$",
    re.IGNORECASE,
)
_KNOWN_WATERMARKS = (
    "freekaoyan.com",
    "book118.com",
    "doc88.com",
    "wenku.baidu.com",
)


class KnowledgeCleaner:
    """将 MinerU 原始输出整理成可分块的 Markdown。"""

    def __init__(self, catalog_path: Path) -> None:
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog(catalog_path)

    def clean(self, extracted_dir: Path, output_dir: Path, report_path: Path) -> CleaningReport:
        """清洗目录中的全部已登记教材，并写出质量报告。"""

        groups = self._discover_markdown(extracted_dir)
        qualities: list[BookQuality] = []
        for key, markdown_files in sorted(groups.items()):
            course, extracted_title = key
            book = self._match_book(course, extracted_title)
            quality = self._clean_book(
                book=book,
                extracted_dir=extracted_dir,
                markdown_files=markdown_files,
                output_dir=output_dir,
            )
            qualities.append(quality)

        if not qualities:
            raise ValueError(f"没有在 {extracted_dir} 中找到 MinerU Markdown")

        report = CleaningReport(
            generated_at=datetime.now(UTC).isoformat(),
            books=qualities,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _load_catalog(path: Path) -> list[CatalogBook]:
        if not path.is_file():
            raise FileNotFoundError(f"教材目录不存在：{path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [CatalogBook.model_validate(item) for item in raw.get("books", [])]

    @staticmethod
    def _discover_markdown(extracted_dir: Path) -> dict[tuple[str, str], list[Path]]:
        """按“课程/书名”合并 MinerU 的多个页码分段。"""

        groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
        for path in extracted_dir.rglob("*.md"):
            relative = path.relative_to(extracted_dir)
            if len(relative.parts) < 3:
                continue
            course, title = relative.parts[0], relative.parts[1]
            groups[(course, title)].append(path)

        for paths in groups.values():
            paths.sort(key=KnowledgeCleaner._part_sort_key)
        return groups

    @staticmethod
    def _part_sort_key(path: Path) -> tuple[int, str]:
        for parent in path.parents:
            match = _PART_PATTERN.match(parent.name)
            if match:
                return int(match.group(1)), str(path)
        return 0, str(path)

    def _match_book(self, course: str, extracted_title: str) -> CatalogBook:
        candidates = [book for book in self.catalog if book.course == course]
        normalized = self._normalize_title(extracted_title)
        for book in candidates:
            source_stem = Path(book.source_file).stem
            if self._normalize_title(source_stem) == normalized:
                return book
        raise ValueError(f"教材未登记：{course}/{extracted_title}")

    @staticmethod
    def _normalize_title(value: str) -> str:
        return re.sub(r"[：:（）()\s_-]", "", value).lower()

    def _clean_book(
        self,
        *,
        book: CatalogBook,
        extracted_dir: Path,
        markdown_files: list[Path],
        output_dir: Path,
    ) -> BookQuality:
        cleaned_parts: list[str] = []
        input_characters = 0
        removed = 0
        missing_images = 0
        preserved_urls = 0
        warnings: list[str] = []

        for markdown_path in markdown_files:
            raw = markdown_path.read_text(encoding="utf-8", errors="replace")
            input_characters += len(raw)
            preserved_urls += sum(
                1
                for line in raw.splitlines()
                if _URL_ONLY_PATTERN.match(line)
                and not any(mark in line.lower() for mark in _KNOWN_WATERMARKS)
            )
            cleaned, part_removed = self._clean_text(raw)
            cleaned, part_missing_images = self._rewrite_image_paths(
                cleaned,
                markdown_path=markdown_path,
                extracted_dir=extracted_dir,
                clean_file=output_dir / book.course / f"{book.source_id}.md",
            )
            removed += part_removed
            missing_images += part_missing_images
            cleaned_parts.append(cleaned.strip())

        body = "\n\n".join(part for part in cleaned_parts if part).strip() + "\n"
        metadata = self._metadata_header(book)
        output_text = metadata + body
        output_file = output_dir / book.course / f"{book.source_id}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output_text, encoding="utf-8")

        headings = len(re.findall(r"(?m)^#{1,6}\s+\S", body))
        formulas = body.count("$$") // 2 + len(re.findall(r"(?<!\$)\$(?!\$)", body)) // 2
        images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", body))
        if len(body) < 1000:
            warnings.append("正文字符数过少，请人工检查 MinerU 输出")
        if headings == 0:
            warnings.append("没有识别到 Markdown 标题")
        if body.count("$$") % 2:
            warnings.append("独立公式定界符数量为奇数")
        if "�" in body:
            warnings.append("正文包含 Unicode 替换字符")
        if missing_images:
            warnings.append(f"有 {missing_images} 个图片链接找不到对应文件")
        if preserved_urls:
            warnings.append(f"正文保留 {preserved_urls} 行独立 URL，请人工确认用途")

        return BookQuality(
            source_id=book.source_id,
            course=book.course,
            title=book.title,
            input_files=[str(path) for path in markdown_files],
            output_file=str(output_file),
            content_sha256=hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            input_characters=input_characters,
            output_characters=len(output_text),
            headings=headings,
            formulas=formulas,
            images=images,
            removed_noise_lines=removed,
            warnings=warnings,
        )

    @staticmethod
    def _metadata_header(book: CatalogBook) -> str:
        page_value = "null" if book.pages is None else str(book.pages)
        return (
            "---\n"
            f"source_id: {book.source_id}\n"
            f"book_title: {json.dumps(book.title, ensure_ascii=False)}\n"
            f"course: {book.course}\n"
            f"course_name: {json.dumps(book.course_name, ensure_ascii=False)}\n"
            f"source_file: {json.dumps(book.source_file, ensure_ascii=False)}\n"
            f"pages: {page_value}\n"
            "extractor: MinerU precision extract\n"
            "---\n\n"
        )

    @staticmethod
    def _clean_text(raw: str) -> tuple[str, int]:
        """删除确定的排版噪声，同时保留公式、表格和正文。"""

        normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\ufeff", "").replace("\u200b", "")
        kept: list[str] = []
        removed = 0
        for line in normalized.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            is_watermark_line = (
                any(mark in lower for mark in _KNOWN_WATERMARKS) and len(stripped) <= 120
            )
            if _EMPTY_TAG_PATTERN.match(line) or is_watermark_line:
                removed += 1
                continue
            if kept and stripped and stripped == kept[-1].strip() and len(stripped) <= 80:
                removed += 1
                continue
            kept.append(line.rstrip())

        text = "\n".join(kept)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip() + "\n", removed

    @staticmethod
    def _rewrite_image_paths(
        text: str,
        *,
        markdown_path: Path,
        extracted_dir: Path,
        clean_file: Path,
    ) -> tuple[str, int]:
        """让 clean 目录中的图片仍然指向 MinerU 原图。"""

        missing = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal missing
            target = match.group(2)
            if "://" in target or target.startswith("/"):
                return match.group(0)
            absolute = (markdown_path.parent / target).resolve()
            try:
                absolute.relative_to(extracted_dir.resolve())
            except ValueError:
                return match.group(0)
            if not absolute.is_file():
                missing += 1
            # os.path.relpath 能正确处理书名和分段目录中的中文字符。
            rewritten = os.path.relpath(absolute, start=clean_file.parent)
            return f"{match.group(1)}{Path(rewritten).as_posix()}{match.group(3)}"

        return _IMAGE_PATTERN.sub(replace, text), missing
