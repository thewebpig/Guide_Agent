"""知识文档加载与文本切分。

本模块负责RAG流程中最靠前的两步：先按照Scene中的source_docs读取原始文档，
再把完整文档切成适合后续向量化和检索的小片段。Markdown 文档会按标题和
段落切分，使每条证据保持一个明确的语义边界；普通文本和超长段落仍使用有界
的固定宽度切分。
"""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from guide_agent.scene import Scene


class SourceDocument(BaseModel):
    """从一个来源文件读取的完整文档。

    source保存相对于场景目录的来源路径，text保存清理后的全文。二者一起保留，
    后续即使文本被切成很多块，也仍能追溯它来自哪个文件。
    """

    # Field(min_length=1)拒绝真正的空字符串；load_documents还会先strip，
    # 所以由加载函数创建的模型也不会包含只有空格、换行或制表符的文本。
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    """参与后续向量化与检索的单个文本片段。"""

    # chunk_id是每个片段的稳定标识，便于检索结果定位、引用和测试。
    chunk_id: str = Field(min_length=1)
    # text是实际参与Embedding和相似度检索的片段内容。
    text: str = Field(min_length=1)
    # source保留原始文件路径，让检索结果能够说明资料来源。
    source: str = Field(min_length=1)


class DocumentLoadError(ValueError):
    """知识文档读取失败时抛出的统一业务错误。

    调用者只需捕获这一种错误；原始的文件系统或编码异常仍通过__cause__保留，
    因而不会丢失排查问题所需的信息。
    """


def load_documents(
    scene: Scene,
    scene_dir: str | Path,
) -> list[SourceDocument]:
    """读取场景配置中指定的全部知识文档。

    scene_dir是scene.json所在的场景目录；scene.source_docs中的每一项都是相对
    这个目录的路径。函数保持配置中的顺序并返回SourceDocument列表。
    """

    # 同时接受字符串路径和Path对象，统一转为Path后再做路径运算。
    root = Path(scene_dir)

    # 先检查场景根目录，能够把“场景目录错误”和“某个来源文件缺失”区分开。
    if not root.is_dir():
        raise DocumentLoadError(
            f"scene directory not found: {root}"
        )

    documents: list[SourceDocument] = []

    # source_docs决定需要加载哪些资料；这里不把demo文件名写死在程序中。
    for source in scene.source_docs:
        # Path的/运算符用于拼接路径，例如scene_dir / "docs/visitor_guide.md"。
        document_path = root / source

        try:
            # 明确使用UTF-8，避免结果随着操作系统默认编码变化。
            text = document_path.read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeError) as error:
            # OSError包含文件不存在、无权限等读取问题；UnicodeError包含
            # UTF-8解码失败。raise ... from error把原始异常保存在__cause__中。
            raise DocumentLoadError(
                f"failed to load document {document_path}: {error}"
            ) from error

        # 删除全文首尾的空白，避免无意义空白进入后续切分。
        cleaned_text = text.strip()

        # 只有空格、换行或制表符的文件，strip后会成为空字符串。
        if not cleaned_text:
            raise DocumentLoadError(
                f"empty document: {document_path}"
            )

        # source保留配置里的相对路径，text保存已经通过检查的正文。
        documents.append(
            SourceDocument(
                source=source,
                text=cleaned_text,
            )
        )

    return documents


def split_documents(
    documents: list[SourceDocument],
    *,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """按 Markdown 语义边界或固定宽度切分文档。

    参数列表中的*表示chunk_size与overlap只能按关键字传入，例如
    split_documents(docs, chunk_size=500, overlap=100)，避免两个整数写反。
    """

    # chunk_size必须为正数，否则片段没有有效长度。
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    # overlap表示相邻片段重复的字符数，不能是负数。
    if overlap < 0:
        raise ValueError(
            "overlap must be greater than or equal to 0"
        )

    # 重叠量必须小于片段大小，才能保证每轮切分都向后推进。
    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks: list[DocumentChunk] = []

    for document in documents:
        # Markdown 的标题层级和段落是作者明确写出的语义边界。只有真的存在
        # ATX 标题时才使用这条路径；文件扩展名为 .md 但没有标题的文本仍按
        # 固定宽度处理，避免意外改变旧资料的切分语义。
        semantic_units = _markdown_paragraph_units(document.text)
        unit_texts = (
            _expand_markdown_units(semantic_units, chunk_size, overlap)
            if semantic_units is not None
            else _fixed_width_units(document.text, chunk_size, overlap)
        )

        for chunk_number, chunk_text in enumerate(unit_texts, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.source}::chunk-{chunk_number:04d}",
                    text=chunk_text,
                    source=document.source,
                )
            )

    return chunks


_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _markdown_paragraph_units(text: str) -> list[tuple[str, str]] | None:
    """Return ``(heading context, paragraph)`` units for headed Markdown.

    The context is included in the embedded text, so a paragraph such as an
    opening-hours rule retains both the venue title and its ``##`` section.
    Returning ``None`` deliberately chooses the legacy fixed-width fallback.
    """

    lines = text.splitlines()
    if not any(_MARKDOWN_HEADING.match(line) for line in lines):
        return None

    headings: list[str] = []
    body: list[str] = []
    units: list[tuple[str, str]] = []

    def flush_body() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        body = []
        if not content or not headings:
            return
        context = "\n".join(headings)
        # Blank lines mark paragraphs in the small source documents.  Keep
        # wrapped lines inside a paragraph intact rather than flattening them.
        for paragraph in re.split(r"\n\s*\n+", content):
            cleaned = paragraph.strip()
            if cleaned:
                units.append((context, cleaned))

    for line in lines:
        match = _MARKDOWN_HEADING.match(line)
        if match:
            flush_body()
            level = len(match.group(1))
            heading = f"{'#' * level} {match.group(2).strip()}"
            del headings[level - 1 :]
            headings.append(heading)
        else:
            body.append(line)
    flush_body()

    return units or None


def _expand_markdown_units(
    units: list[tuple[str, str]], chunk_size: int, overlap: int
) -> list[str]:
    """Format semantic units and apply a bounded fallback to oversized text."""

    chunks: list[str] = []
    for context, paragraph in units:
        prefix = f"{context}\n\n"
        combined = f"{prefix}{paragraph}"
        if len(combined) <= chunk_size:
            chunks.append(combined)
            continue

        # A long paragraph must not create an unbounded embedding input. Keep
        # the heading context on every fallback chunk, then window only body.
        body_size = max(1, chunk_size - len(prefix))
        for body_piece in _fixed_width_units(paragraph, body_size, overlap):
            chunks.append(f"{prefix}{body_piece}")
    return chunks


def _fixed_width_units(text: str, chunk_size: int, overlap: int) -> list[str]:
    """The bounded, overlapping fallback used for non-Markdown and long text."""

    effective_overlap = min(overlap, chunk_size - 1)
    step = chunk_size - effective_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(text):
            break
        start += step
    return chunks
