"""知识文档加载与切分测试。

测试分为两组：第一组验证真实场景文档的加载以及各种读取失败；第二组用一段
很短的可计算文本验证切分、重叠、稳定ID与参数边界。
"""

from pathlib import Path

import pytest

from guide_agent.documents import (
    DocumentChunk,
    DocumentLoadError,
    SourceDocument,
    load_documents,
    split_documents,
)
from guide_agent.scene import load_scene


# 从测试文件的位置推导项目根目录，不依赖pytest启动时的当前工作目录。
# parents[1]是guide_agent项目根目录，真实demo场景位于其scenes/demo目录。
PROJECT_ROOT = Path(__file__).parents[1]
SCENE_DIR = PROJECT_ROOT / "scenes" / "demo"
SCENE_PATH = SCENE_DIR / "scene.json"


def test_load_documents_reads_configured_sources() -> None:
    """正常情况下应按scene.json配置读取真实访客指南。"""

    scene = load_scene(SCENE_PATH)

    documents = load_documents(
        scene,
        SCENE_DIR,
    )

    # demo场景当前只配置了一份source_docs，因此应只产生一份完整文档。
    assert len(documents) == 1

    document = documents[0]
    # 同时固定返回模型类型、来源路径和几个关键正文内容。
    assert isinstance(document, SourceDocument)
    assert document.source == "docs/visitor_guide.md"
    assert document.text.startswith(
        "# 星河科技体验中心访客指南"
    )
    assert "人工智能实验室" in document.text
    assert "开放与访问规则" in document.text


def test_load_documents_rejects_missing_scene_directory(
    tmp_path: Path,
) -> None:
    """场景根目录本身不存在时，应给出明确的DocumentLoadError。"""

    scene = load_scene(SCENE_PATH)
    # pytest提供的tmp_path是真实临时目录；这个子路径没有被创建，故必定不存在。
    missing_scene_dir = tmp_path / "missing-scene"

    # match检查的不只是异常类型，也确认错误消息说明了正确的失败原因。
    with pytest.raises(
        DocumentLoadError,
        match="scene directory not found",
    ):
        load_documents(
            scene,
            missing_scene_dir,
        )


def test_load_documents_wraps_missing_source_file(
    tmp_path: Path,
) -> None:
    """场景目录存在但配置的来源文件缺失时，应包装底层文件异常。"""

    scene = load_scene(SCENE_PATH)
    # model_copy创建仅供本测试使用的场景变体，不改动磁盘上的scene.json。
    missing_source_scene = scene.model_copy(
        update={
            "source_docs": [
                "docs/missing.md",
            ],
        }
    )

    with pytest.raises(
        DocumentLoadError,
        match="failed to load document",
    ) as error_info:
        load_documents(
            missing_source_scene,
            tmp_path,
        )

    # 对外统一抛DocumentLoadError，同时__cause__仍保留真实FileNotFoundError。
    assert isinstance(
        error_info.value.__cause__,
        FileNotFoundError,
    )


def test_load_documents_rejects_empty_document(
    tmp_path: Path,
) -> None:
    """只有空白字符的UTF-8文件也应被当作空文档拒绝。"""

    # 在pytest临时目录中搭建最小docs结构，不污染项目里的真实场景资料。
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    empty_document_path = docs_dir / "empty.md"
    # 内容不是零字节，但strip后为空，可验证“空白文档”这一业务边界。
    empty_document_path.write_text(
        "  \n\t",
        encoding="utf-8",
    )

    scene = load_scene(SCENE_PATH)
    empty_document_scene = scene.model_copy(
        update={
            "source_docs": [
                "docs/empty.md",
            ],
        }
    )

    with pytest.raises(
        DocumentLoadError,
        match="empty document",
    ):
        load_documents(
            empty_document_scene,
            tmp_path,
        )


def test_load_documents_wraps_invalid_utf8(
    tmp_path: Path,
) -> None:
    """不能按UTF-8解码的来源文件应转换为统一加载错误。"""

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    invalid_document_path = docs_dir / "invalid.md"
    # 直接写入一组非法UTF-8字节，稳定触发UnicodeDecodeError。
    invalid_document_path.write_bytes(
        b"\xff\xfe\xfa"
    )

    scene = load_scene(SCENE_PATH)
    invalid_document_scene = scene.model_copy(
        update={
            "source_docs": [
                "docs/invalid.md",
            ],
        }
    )

    with pytest.raises(
        DocumentLoadError,
        match="failed to load document",
    ) as error_info:
        load_documents(
            invalid_document_scene,
            tmp_path,
        )

    # 统一异常没有掩盖根因，测试仍能通过__cause__确认是解码失败。
    assert isinstance(
        error_info.value.__cause__,
        UnicodeDecodeError,
    )


def test_split_documents_creates_overlap_and_stable_ids() -> None:
    """正常切分应产生预期重叠文本、顺序编号和来源信息。"""

    # 十个字符足以手工推导结果，便于准确验证切分算法而非只检查数量。
    document = SourceDocument(
        source="demo.md",
        text="abcdefghij",
    )

    # chunk_size=4、overlap=1时step=3，起点依次为0、3、6。
    chunks = split_documents(
        [document],
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) == 3
    # 所有返回项都应是结构明确的DocumentChunk，而不是裸字符串。
    assert all(
        isinstance(chunk, DocumentChunk)
        for chunk in chunks
    )
    # ID包含来源和补零序号，因而可追溯、稳定且按字符串排序也符合先后顺序。
    assert [
        chunk.chunk_id for chunk in chunks
    ] == [
        "demo.md::chunk-0001",
        "demo.md::chunk-0002",
        "demo.md::chunk-0003",
    ]
    # 三块分别为[0:4]、[3:7]、[6:10]，相邻块各重叠一个字符。
    assert [
        chunk.text for chunk in chunks
    ] == [
        "abcd",
        "defg",
        "ghij",
    ]
    # 每个片段都必须保留它所来自的原始文档。
    assert all(
        chunk.source == "demo.md"
        for chunk in chunks
    )


def test_split_documents_keeps_markdown_sections_and_paragraphs_independent() -> None:
    """Headed Markdown keeps context while avoiding cross-section evidence."""

    document = SourceDocument(
        source="guide.md",
        text=(
            "# 场馆指南\n\n"
            "## 场馆概览\n\n"
            "这里是概览。\n\n"
            "## 开放与访问规则\n\n"
            "常规开放时间为周二至周日9:30—17:30，16:30停止入场。\n\n"
            "请从主入口完成安检。"
        ),
    )

    chunks = split_documents([document], chunk_size=500, overlap=100)

    assert [chunk.chunk_id for chunk in chunks] == [
        "guide.md::chunk-0001",
        "guide.md::chunk-0002",
        "guide.md::chunk-0003",
    ]
    assert chunks[1].text == (
        "# 场馆指南\n## 开放与访问规则\n\n"
        "常规开放时间为周二至周日9:30—17:30，16:30停止入场。"
    )
    assert "这里是概览" not in chunks[1].text
    assert chunks[1].source == "guide.md"


def test_split_documents_uses_bounded_fallback_for_oversized_markdown_paragraph() -> None:
    """A long paragraph remains bounded and keeps its heading context."""

    document = SourceDocument(
        source="guide.md",
        text="# 指南\n\n## 规则\n\n" + "甲" * 30,
    )

    chunks = split_documents([document], chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all("## 规则" in chunk.text for chunk in chunks)
    # The heading itself can exceed a tiny synthetic chunk size, but each body
    # window remains bounded rather than allowing a full oversized paragraph.
    assert all(len(chunk.text) <= 20 for chunk in chunks)


# 参数化把三种独立的非法边界生成三条测试，避免复制三份测试函数。
@pytest.mark.parametrize(
    (
        "chunk_size",
        "overlap",
        "expected_message",
    ),
    [
        (
            0,
            0,
            "chunk_size must be greater than 0",
        ),
        (
            4,
            -1,
            "overlap must be greater than or equal to 0",
        ),
        (
            4,
            4,
            "overlap must be smaller than chunk_size",
        ),
    ],
)
def test_split_documents_rejects_invalid_parameters(
    chunk_size: int,
    overlap: int,
    expected_message: str,
) -> None:
    """非法大小或重叠量应在进入切分循环前立即报错。"""

    # 即使文档列表为空，参数规则也必须先验证；match固定具体错误原因。
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        split_documents(
            [],
            chunk_size=chunk_size,
            overlap=overlap,
        )
