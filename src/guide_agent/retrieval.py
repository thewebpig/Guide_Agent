"""知识片段的Embedding与FAISS索引构建。

本模块负责RAG检索链路的“建库”部分：把DocumentChunk正文转换成固定维度
的浮点向量，完成校验和L2归一化，再按原顺序加入FAISS索引。T3.4将在
此基础上把用户问题编码成查询向量，并把FAISS返回的位置映射回原始chunk。
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
from fastembed import TextEmbedding
import numpy as np
from numpy.typing import NDArray

from guide_agent.documents import (
    DocumentChunk,
    load_documents,
    split_documents,
)
from guide_agent.scene import load_scene


# 固定模型名称是可复现性的一部分。当前模型面向中文检索，输出512维向量，
# FastEmbed使用本地ONNX模型执行推理，不需要把API密钥写入项目。
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class Embedder(Protocol):
    """文本向量模型必须提供的最小接口。

    Protocol使用“结构化类型”：对象不必继承Embedder，只要实现这两个方法，
    就能被构建和检索代码使用。因此真实运行可传FastEmbedder，单元测试则可传
    返回固定小向量的FakeEmbedder，避免测试联网、下载模型或依赖模型效果。
    """

    def embed(
        self,
        documents: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        """把一组文本转换成一组向量。"""

    def query_embed(
        self,
        queries: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        """把检索问题转换成一组查询向量。"""


class FastEmbedder:
    """使用FastEmbed执行本地文本向量化。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        # 保存模型名便于构建脚本、日志和后续评测记录实际使用的方案。
        self.model_name = model_name
        # 第一次实例化会下载模型，之后通常从本机缓存加载。
        self._model = TextEmbedding(
            model_name=model_name,
        )

    def embed(
        self,
        documents: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        """生成知识文档片段的向量。"""

        # 文档片段使用passage_embed；BGE不会给passage添加查询指令。
        return self._model.passage_embed(
            documents
        )

    def query_embed(
        self,
        queries: list[str],
    ) -> Iterable[NDArray[np.float32]]:
        """生成用户检索问题的向量。"""

        # 查询使用query_embed，让FastEmbed执行模型对应的查询侧处理。
        return self._model.query_embed(
            queries
        )


@dataclass
class KnowledgeIndex:
    """把FAISS数字索引、原始片段和向量模型绑定在一起。

    FAISS只返回向量在索引中的数字位置，不保存chunk正文和来源。因此chunks的
    顺序必须始终与向量加入索引的顺序一致。embedder则供后续查询使用，确保问题
    和文档由同一模型编码到同一个向量空间。
    """

    chunks: list[DocumentChunk]
    index: faiss.Index
    embedder: Embedder


class IndexBuildError(ValueError):
    """片段或Embedding输出无法构成有效索引时抛出的业务错误。"""


class KnowledgeSearchError(ValueError):
    """知识检索输入或查询向量无效时抛出的错误。"""


def _embed_texts(
    embedder: Embedder,
    texts: list[str],
) -> NDArray[np.float32]:
    """生成经过校验和归一化的二维float32向量矩阵。"""

    # FastEmbed返回惰性生成器。list先执行全部推理，np.asarray再把结果转换为
    # FAISS要求的float32矩阵；矩阵形状应为“文本数量 × 向量维度”。
    vectors = np.asarray(
        list(embedder.embed(texts)),
        dtype=np.float32,
    )

    # 一条或多条文本的输出也必须是二维矩阵，而不是单个标量或一维数组。
    if vectors.ndim != 2:
        raise IndexBuildError(
            "embeddings must be a two-dimensional matrix"
        )

    # 每条输入文本必须恰好对应一行向量，否则位置无法映射回chunk。
    if vectors.shape[0] != len(texts):
        raise IndexBuildError(
            "embedding count does not match text count"
        )

    # 第二维是Embedding维度；零维向量没有任何可比较的信息。
    if vectors.shape[1] == 0:
        raise IndexBuildError(
            "embedding dimension must be greater than 0"
        )

    # NaN和正负无穷会污染相似度计算，因此在进入FAISS前明确拒绝。
    if not np.isfinite(vectors).all():
        raise IndexBuildError(
            "embeddings must contain only finite values"
        )

    # axis=1表示分别计算每一行向量的欧几里得长度，而非整张矩阵的长度。
    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    # 零向量没有方向，不能形成有效的余弦相似度。
    if np.any(norms == 0):
        raise IndexBuildError(
            "embeddings must not contain zero vectors"
        )

    # FAISS的Python接口读取连续内存中的float32数据；这里显式固定布局和类型。
    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )
    # 原地把每行长度归一化为1。此后两个向量的内积等价于余弦相似度。
    faiss.normalize_L2(vectors)

    return vectors


def _embed_query(
    embedder: Embedder,
    query: str,
    expected_dimension: int,
) -> NDArray[np.float32]:
    """生成经过校验和归一化的单个查询向量。

    expected_dimension来自已经建好的FAISS索引。查询必须由同一向量模型编码，
    并得到一行相同维度的向量，才能和索引内的文档向量计算相似度。
    """

    # query_embed接收列表并返回惰性迭代器。单次检索只传一个清理后的问题，
    # 因而正常形状应为(1, expected_dimension)。
    vectors = np.asarray(
        list(
            embedder.query_embed(
                [query]
            )
        ),
        dtype=np.float32,
    )

    expected_shape = (
        1,
        expected_dimension,
    )

    # 同时检查查询数量和向量维度，防止不同模型的输出进入当前索引。
    if vectors.shape != expected_shape:
        raise KnowledgeSearchError(
            "query embedding shape does not match index dimension"
        )

    # 查询侧同样拒绝NaN和无穷，否则FAISS得分不再可信。
    if not np.isfinite(vectors).all():
        raise KnowledgeSearchError(
            "query embedding must contain only finite values"
        )

    # 只有一行，所以直接计算vectors[0]的L2长度。
    query_norm = np.linalg.norm(
        vectors[0]
    )

    if query_norm == 0:
        raise KnowledgeSearchError(
            "query embedding must not be a zero vector"
        )

    # 固定为连续float32矩阵，并归一化为单位向量；这样查询与文档的
    # IndexFlatIP内积就是余弦相似度。
    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )
    faiss.normalize_L2(vectors)

    return vectors


def build_knowledge_index(
    chunks: list[DocumentChunk],
    embedder: Embedder,
) -> KnowledgeIndex:
    """为知识片段建立精确的内积相似度索引。"""

    # 没有片段就没有任何可检索证据，应在调用Embedding前失败。
    if not chunks:
        raise IndexBuildError(
            "cannot build index without chunks"
        )

    # chunk_id是检索结果的证据标识；重复ID会让引用无法唯一定位。
    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    if len(set(chunk_ids)) != len(chunk_ids):
        raise IndexBuildError(
            "duplicate chunk IDs are not allowed"
        )

    # 文本顺序与chunks顺序保持一致，这是FAISS位置映射回原文的关键不变量。
    texts = [
        chunk.text
        for chunk in chunks
    ]
    vectors = _embed_texts(
        embedder,
        texts,
    )

    # IndexFlatIP执行精确内积搜索，不需要训练，适合当前只有少量chunk的知识库。
    # dimension必须与每条向量的列数一致；当前真实BGE模型得到512。
    dimension = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dimension)
    # add按行依次加入向量，第一行对应位置0和chunks[0]，以此类推。
    index.add(vectors)

    # ntotal是FAISS实际保存的向量数，防御性检查保证没有静默遗漏。
    if index.ntotal != len(chunks):
        raise IndexBuildError(
            "FAISS vector count does not match chunk count"
        )

    return KnowledgeIndex(
        # 创建列表副本，避免调用者之后增删原列表导致位置对应关系被破坏。
        chunks=list(chunks),
        index=index,
        embedder=embedder,
    )


def build_scene_knowledge_index(
    scene_path: str | Path,
    *,
    embedder: Embedder | None = None,
    chunk_size: int = 500,
    overlap: int = 100,
) -> KnowledgeIndex:
    """根据一份场景配置重新构建完整知识索引。

    embedder为None时使用真实本地模型；测试可注入FakeEmbedder。chunk_size和
    overlap保持关键字参数，调用处不会混淆二者，并可在后续评测中调整。
    """

    scene_file = Path(scene_path)
    # load_scene负责JSON语法、字段与路线业务校验。
    scene = load_scene(scene_file)

    # scene.json中的source_docs路径相对于场景目录，所以传scene_file.parent。
    documents = load_documents(
        scene,
        scene_file.parent,
    )
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # 明确判断is not None，避免某个合法但布尔值为False的自定义对象被替换。
    active_embedder = (
        embedder
        if embedder is not None
        else FastEmbedder()
    )

    return build_knowledge_index(
        chunks,
        active_embedder,
    )


def search_knowledge(
    knowledge_index: KnowledgeIndex,
    query: str,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """返回与问题最相似的Top-K知识片段及其来源和得分。

    本函数只负责确定性检索，不调用生成式大模型回答问题。返回普通字典是为了
    让后续GuideService、LLM Tool或HTTP接口可以直接序列化和复用这些证据。
    """

    # 类型提示不会在运行时自动校验；显式检查可给外部调用者稳定业务错误。
    if not isinstance(query, str):
        raise KnowledgeSearchError(
            "query must be a string"
        )

    # 删除首尾空白，既避免把无意义字符交给模型，也统一测试和实际输入。
    cleaned_query = query.strip()

    if not cleaned_query:
        raise KnowledgeSearchError(
            "query must not be blank"
        )

    # bool是int的子类，但True不是清晰合法的Top-K数量，所以必须先单独拒绝。
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
    ):
        raise KnowledgeSearchError(
            "top_k must be an integer"
        )

    if top_k <= 0:
        raise KnowledgeSearchError(
            "top_k must be greater than 0"
        )

    # ntotal是FAISS中的真实向量数。索引状态错误应在运行查询模型前发现。
    vector_count = int(
        knowledge_index.index.ntotal
    )

    if vector_count == 0:
        raise KnowledgeSearchError(
            "knowledge index is empty"
        )

    # 数量不一致说明“位置→chunk”的映射已损坏，继续检索可能引用错误证据。
    if vector_count != len(
        knowledge_index.chunks
    ):
        raise KnowledgeSearchError(
            "FAISS vector count does not match chunk count"
        )

    # top_k大于索引规模时只搜索现有向量，避免FAISS用-1补齐不存在的位置。
    effective_top_k = min(
        top_k,
        vector_count,
    )

    query_vector = _embed_query(
        knowledge_index.embedder,
        cleaned_query,
        knowledge_index.index.d,
    )

    # search返回两个形状为(1, K)的矩阵：scores是内积分数，positions是
    # FAISS内部数字位置。IndexFlatIP按分数从高到低排列结果。
    scores, positions = (
        knowledge_index.index.search(
            query_vector,
            effective_top_k,
        )
    )

    results: list[dict[str, object]] = []

    # zip(strict=True)要求得分数和位置数完全相等，防止静默截断异常数据。
    for score, position in zip(
        scores[0],
        positions[0],
        strict=True,
    ):
        chunk_position = int(position)

        # 正常情况下前面限制K后不会出现越界；这里仍做防御性检查。
        if not (
            0
            <= chunk_position
            < len(knowledge_index.chunks)
        ):
            raise KnowledgeSearchError(
                "FAISS returned an invalid chunk position"
            )

        # FAISS位置与建库时chunks顺序相同，因此可取回完整证据模型。
        chunk = knowledge_index.chunks[
            chunk_position
        ]

        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                # NumPy float32转换为Python float，便于JSON序列化。
                "score": float(score),
            }
        )

    return results
