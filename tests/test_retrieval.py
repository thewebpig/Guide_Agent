"""Embedding与FAISS索引构建测试。

单元测试使用FakeEmbedder固定向量，不依赖网络、模型缓存或模型效果；最后一条
集成测试读取真实demo场景与文档，但仍注入FakeEmbedder，从而稳定验证完整建库
流水线。真实模型另由scripts/build_index.py进行烟雾验证。
"""

from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

from guide_agent.documents import DocumentChunk
from guide_agent.retrieval import (
    KnowledgeIndex,
    build_knowledge_index,
    IndexBuildError,
    build_scene_knowledge_index,
    search_knowledge,
    KnowledgeSearchError,
)

# 测试路径从文件自身推导，不依赖pytest从哪个目录启动。
PROJECT_ROOT = Path(__file__).parents[1]
DEMO_SCENE_PATH = (
    PROJECT_ROOT
    / "scenes"
    / "demo"
    / "scene.json"
)


class FakeEmbedder:
    """测试专用的可控向量模型，不下载真实模型。

    vectors由每条测试预先指定；received_documents记录被测代码交给模型的正文，
    因而既能测试输出，也能确认输入文本和顺序正确。
    """

    def __init__(
        self,
        vectors: list[NDArray[np.float32]],
        query_vectors: (
            list[NDArray[np.float32]]
            | None
        ) = None,
    ) -> None:
        self.vectors = vectors
        # query_vectors与文档vectors分开，才能验证检索确实走query_embed。
        self.query_vectors = (
            query_vectors
            if query_vectors is not None
            else []
        )
        self.received_documents: list[str] | None = None
        self.received_queries: list[str] | None = None

    def embed(
        self,
        documents: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        # 返回迭代器以模拟FastEmbed的惰性返回形式。
        self.received_documents = documents
        return iter(self.vectors)

    def query_embed(
        self,
        queries: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        # 记录清理后的问题，并模拟FastEmbed返回查询向量迭代器。
        self.received_queries = queries
        return iter(self.query_vectors)


def test_build_knowledge_index_adds_normalized_vectors() -> None:
    """正常建库应保持映射关系，并把向量归一化后加入FAISS。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="主入口位于一层。",
            source="demo.md",
        ),
        DocumentChunk(
            chunk_id="demo.md::chunk-0002",
            text="科技展厅展示机器人。",
            source="demo.md",
        ),
    ]
    embedder = FakeEmbedder(
        [
            np.array(
                [3.0, 4.0],
                dtype=np.float32,
            ),
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
        ]
    )

    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    # 固定容器和FAISS具体索引类型，并检查维度、数量两个关键元数据。
    assert isinstance(
        knowledge_index,
        KnowledgeIndex,
    )
    assert isinstance(
        knowledge_index.index,
        faiss.IndexFlatIP,
    )
    assert knowledge_index.index.d == 2
    assert knowledge_index.index.ntotal == 2

    # Embedder只应收到chunk正文，且顺序必须与chunks一致。
    assert embedder.received_documents == [
        "主入口位于一层。",
        "科技展厅展示机器人。",
    ]
    # 内容相等但不是同一列表对象，证明构建函数保存了防修改副本。
    assert knowledge_index.chunks == chunks
    assert knowledge_index.chunks is not chunks
    assert knowledge_index.embedder is embedder

    # reconstruct_n从FAISS取回位置0开始的两个向量，直接验证[3,4]已经
    # 归一化为[0.6,0.8]，而单位向量[1,0]保持不变。
    stored_vectors = (
        knowledge_index.index.reconstruct_n(
            0,
            2,
        )
    )
    np.testing.assert_allclose(
        stored_vectors,
        np.array(
            [
                [0.6, 0.8],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_knowledge_index_rejects_empty_chunks() -> None:
    """空知识库应在执行Embedding前失败。"""

    embedder = FakeEmbedder([])

    with pytest.raises(
        IndexBuildError,
        match="cannot build index without chunks",
    ):
        build_knowledge_index(
            [],
            embedder,
        )

    # None证明FakeEmbedder.embed从未被调用，没有浪费模型计算。
    assert embedder.received_documents is None


def test_build_knowledge_index_rejects_duplicate_chunk_ids() -> None:
    """重复证据ID应在执行Embedding前失败。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="第一段文本。",
            source="demo.md",
        ),
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="第二段文本。",
            source="demo.md",
        ),
    ]
    embedder = FakeEmbedder([])

    with pytest.raises(
        IndexBuildError,
        match="duplicate chunk IDs are not allowed",
    ):
        build_knowledge_index(
            chunks,
            embedder,
        )

    assert embedder.received_documents is None


# 五组参数分别覆盖矩阵维数、数量、向量维度、有限值和零向量校验。
# pytest会把它们显示为五条独立用例，同时复用相同断言逻辑。
@pytest.mark.parametrize(
    (
        "vectors",
        "expected_message",
    ),
    [
        (
            [
                np.array(
                    1.0,
                    dtype=np.float32,
                )
            ],
            "embeddings must be a two-dimensional matrix",
        ),
        (
            [
                np.array(
                    [1.0, 0.0],
                    dtype=np.float32,
                ),
                np.array(
                    [0.0, 1.0],
                    dtype=np.float32,
                ),
            ],
            "embedding count does not match text count",
        ),
        (
            [
                np.array(
                    [],
                    dtype=np.float32,
                )
            ],
            "embedding dimension must be greater than 0",
        ),
        (
            [
                np.array(
                    [np.nan, 1.0],
                    dtype=np.float32,
                )
            ],
            "embeddings must contain only finite values",
        ),
        (
            [
                np.array(
                    [0.0, 0.0],
                    dtype=np.float32,
                )
            ],
            "embeddings must not contain zero vectors",
        ),
    ],
)
def test_build_knowledge_index_rejects_invalid_embeddings(
    vectors: list[NDArray[np.float32]],
    expected_message: str,
) -> None:
    """不合法的模型输出不能进入FAISS索引。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="用于测试的知识片段。",
            source="demo.md",
        )
    ]
    embedder = FakeEmbedder(vectors)

    with pytest.raises(
        IndexBuildError,
        match=expected_message,
    ):
        build_knowledge_index(
            chunks,
            embedder,
        )


def test_build_scene_knowledge_index_rebuilds_demo() -> None:
    """真实场景配置和知识文档应能从头重建索引。"""

    # chunk_size远大于当前指南，使真实文档稳定切成一块，只需提供一个假向量。
    embedder = FakeEmbedder(
        [
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ]
    )

    knowledge_index = build_scene_knowledge_index(
        DEMO_SCENE_PATH,
        embedder=embedder,
        chunk_size=10_000,
        overlap=0,
    )

    # 检查流水线最终索引、片段和注入的模型保持一致。
    assert knowledge_index.index.ntotal == 1
    assert len(knowledge_index.chunks) == 1
    assert knowledge_index.embedder is embedder

    # 固定真实来源的稳定ID、来源路径和关键正文，证明不是用假文档绕过加载。
    chunk = knowledge_index.chunks[0]
    assert (
        chunk.chunk_id
        == "docs/visitor_guide.md::chunk-0001"
    )
    assert chunk.source == "docs/visitor_guide.md"
    assert chunk.text.startswith(
        "# 星河科技体验中心访客指南"
    )
    assert "人工智能实验室" in chunk.text

    # 最终交给Embedding的文本就是加载、清理和切分后的真实chunk正文。
    assert embedder.received_documents == [
        chunk.text
    ]


def test_search_knowledge_returns_ranked_results() -> None:
    """正常检索应清理问题、按分数排序并返回完整证据字段。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="主入口位于场馆一层。",
            source="demo.md",
        ),
        DocumentChunk(
            chunk_id="demo.md::chunk-0002",
            text="科技展厅展示机器人。",
            source="demo.md",
        ),
        DocumentChunk(
            chunk_id="demo.md::chunk-0003",
            text="人工智能实验室需要预约。",
            source="demo.md",
        ),
    ]
    embedder = FakeEmbedder(
        vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            np.array(
                [0.0, 1.0],
                dtype=np.float32,
            ),
            np.array(
                [0.8, 0.6],
                dtype=np.float32,
            ),
        ],
        query_vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ],
    )
    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    results = search_knowledge(
        knowledge_index,
        "  主入口在哪里？  ",
        top_k=2,
    )

    # 首尾空格未进入模型，证明业务层使用的是cleaned_query。
    assert embedder.received_queries == [
        "主入口在哪里？"
    ]
    assert results == [
        {
            "chunk_id": "demo.md::chunk-0001",
            "text": "主入口位于场馆一层。",
            "source": "demo.md",
            "score": pytest.approx(1.0),
        },
        {
            "chunk_id": "demo.md::chunk-0003",
            "text": "人工智能实验室需要预约。",
            "source": "demo.md",
            "score": pytest.approx(0.8),
        },
    ]


# 非字符串、空字符串和仅含空白的字符串都属于无效问题。
@pytest.mark.parametrize(
    (
        "query",
        "expected_message",
    ),
    [
        (
            None,
            "query must be a string",
        ),
        (
            "",
            "query must not be blank",
        ),
        (
            "  \n\t",
            "query must not be blank",
        ),
    ],
)
def test_search_knowledge_rejects_invalid_query(
    query: object,
    expected_message: str,
) -> None:
    """无效问题应在查询Embedding之前失败。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="有效知识片段。",
            source="demo.md",
        )
    ]
    embedder = FakeEmbedder(
        vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ]
    )
    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    with pytest.raises(
        KnowledgeSearchError,
        match=expected_message,
    ):
        search_knowledge(
            knowledge_index,
            query,  # type: ignore[arg-type]
            top_k=1,
        )

    assert embedder.received_queries is None


# bool需在int之前判断；浮点数是类型错误，零和负数是范围错误。
@pytest.mark.parametrize(
    (
        "top_k",
        "expected_message",
    ),
    [
        (
            True,
            "top_k must be an integer",
        ),
        (
            1.5,
            "top_k must be an integer",
        ),
        (
            0,
            "top_k must be greater than 0",
        ),
        (
            -1,
            "top_k must be greater than 0",
        ),
    ],
)
def test_search_knowledge_rejects_invalid_top_k(
    top_k: object,
    expected_message: str,
) -> None:
    """非法Top-K应在查询Embedding之前失败。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="有效知识片段。",
            source="demo.md",
        )
    ]
    embedder = FakeEmbedder(
        vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ]
    )
    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    with pytest.raises(
        KnowledgeSearchError,
        match=expected_message,
    ):
        search_knowledge(
            knowledge_index,
            "有效问题",
            top_k=top_k,  # type: ignore[arg-type]
        )

    assert embedder.received_queries is None


def test_search_knowledge_limits_top_k_to_index_size() -> None:
    """请求数量超过索引规模时只返回真实存在的证据。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="第一段知识。",
            source="demo.md",
        ),
        DocumentChunk(
            chunk_id="demo.md::chunk-0002",
            text="第二段知识。",
            source="demo.md",
        ),
    ]
    embedder = FakeEmbedder(
        vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            np.array(
                [0.0, 1.0],
                dtype=np.float32,
            ),
        ],
        query_vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ],
    )
    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    results = search_knowledge(
        knowledge_index,
        "查询全部知识",
        top_k=10,
    )

    assert len(results) == 2
    assert [
        result["chunk_id"]
        for result in results
    ] == [
        "demo.md::chunk-0001",
        "demo.md::chunk-0002",
    ]


def test_search_knowledge_rejects_empty_index() -> None:
    """手工构造的空FAISS索引不能执行知识检索。"""

    embedder = FakeEmbedder(
        vectors=[],
        query_vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ],
    )
    knowledge_index = KnowledgeIndex(
        chunks=[],
        index=faiss.IndexFlatIP(2),
        embedder=embedder,
    )

    with pytest.raises(
        KnowledgeSearchError,
        match="knowledge index is empty",
    ):
        search_knowledge(
            knowledge_index,
            "有效问题",
        )

    assert embedder.received_queries is None


def test_search_knowledge_rejects_chunk_count_mismatch() -> None:
    """FAISS向量数与chunk数不一致时应拒绝错误位置映射。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="有效知识。",
            source="demo.md",
        )
    ]
    embedder = FakeEmbedder(
        vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ],
        query_vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ],
    )
    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    # 模拟建库后外部代码错误清空chunk列表，而FAISS仍保留一个向量。
    knowledge_index.chunks.clear()

    with pytest.raises(
        KnowledgeSearchError,
        match="FAISS vector count does not match chunk count",
    ):
        search_knowledge(
            knowledge_index,
            "有效问题",
        )

    assert embedder.received_queries is None


# 三组参数分别覆盖维度不匹配、非有限值和零向量。
@pytest.mark.parametrize(
    (
        "query_vectors",
        "expected_message",
    ),
    [
        (
            [
                np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float32,
                )
            ],
            "query embedding shape does not match index dimension",
        ),
        (
            [
                np.array(
                    [np.inf, 1.0],
                    dtype=np.float32,
                )
            ],
            "query embedding must contain only finite values",
        ),
        (
            [
                np.array(
                    [0.0, 0.0],
                    dtype=np.float32,
                )
            ],
            "query embedding must not be a zero vector",
        ),
    ],
)
def test_search_knowledge_rejects_invalid_query_embedding(
    query_vectors: list[NDArray[np.float32]],
    expected_message: str,
) -> None:
    """模型已被调用，但非法查询向量不能进入FAISS。"""

    chunks = [
        DocumentChunk(
            chunk_id="demo.md::chunk-0001",
            text="有效知识片段。",
            source="demo.md",
        )
    ]
    embedder = FakeEmbedder(
        vectors=[
            np.array(
                [1.0, 0.0],
                dtype=np.float32,
            )
        ],
        query_vectors=query_vectors,
    )
    knowledge_index = build_knowledge_index(
        chunks,
        embedder,
    )

    with pytest.raises(
        KnowledgeSearchError,
        match=expected_message,
    ):
        search_knowledge(
            knowledge_index,
            "有效问题",
            top_k=1,
        )

    assert embedder.received_queries == [
        "有效问题"
    ]
