"""知识文档加载与文本切分。

本模块负责RAG流程中最靠前的两步：先按照Scene中的source_docs读取原始文档，
再把完整文档切成适合后续向量化和检索的小片段。这里暂时只做按字符数切分，
还不涉及Embedding、向量数据库或大模型。
"""

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
    """按照字符数量把完整文档切分为带重叠的片段。

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

    # 每次起点前进“片段大小-重叠量”。例如4和1得到步长3：
    # 第1块从0开始，第2块从3开始，因此两个片段重叠1个字符。
    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []

    for document in documents:
        # 每份文档都从第0个字符开始，片段编号也从1重新计数。
        # chunk_id还包含source，因此不同文档的0001不会发生冲突。
        start = 0
        chunk_number = 1

        while start < len(document.text):
            # 最后一块可能短于chunk_size；min防止end超过文本实际长度。
            end = min(
                start + chunk_size,
                len(document.text),
            )

            # Python切片左闭右开：包含start位置，不包含end位置。
            # 再清理片段首尾空白，避免生成仅包含边界空白的检索内容。
            chunk_text = document.text[
                start:end
            ].strip()

            # 某个切片若清理后为空，就不加入结果，也不消耗片段编号。
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        # :04d把编号补齐为4位，如0001，使ID稳定且便于排序。
                        chunk_id=(
                            f"{document.source}"
                            f"::chunk-{chunk_number:04d}"
                        ),
                        text=chunk_text,
                        source=document.source,
                    )
                )
                chunk_number += 1

            # end到达全文末尾时已经处理完最后一块。立即结束可避免再生成
            # 一个只由前一块重叠区域构成的多余尾块。
            if end == len(document.text):
                break

            # 未到末尾则按固定步长移动起点，形成相邻片段的字符重叠。
            start += step

    return chunks
