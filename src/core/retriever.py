"""
混合检索器
实现向量检索和BM25检索的融合
"""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from loguru import logger

from langchain.schema import Document


class HybridRetriever:
    """混合检索器：向量检索 + BM25检索"""
    
    def __init__(
        self,
        vector_store: Any,
        documents: List[Document] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_hybrid: bool = True
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量数据库
            documents: 文档列表（用于构建BM25索引）
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            use_hybrid: 是否使用混合检索
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.use_hybrid = use_hybrid
        
        # BM25索引
        self.bm25 = None
        self.bm25_documents = []
        
        if documents and use_hybrid:
            self._build_bm25_index(documents)
        
        logger.info(f"检索器初始化完成，混合检索: {use_hybrid}")
    
    def _build_bm25_index(self, documents: List[Document]):
        """构建BM25索引"""
        self.bm25_documents = documents
        
        # 分词（简单按字符分割，中文场景建议使用jieba）
        tokenized_docs = [
            list(doc.page_content) 
            for doc in documents
        ]
        
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25索引构建完成，文档数: {len(documents)}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回文档数量
            score_threshold: 分数阈值
            
        Returns:
            检索到的文档列表
        """
        if self.use_hybrid and self.bm25:
            # 混合检索
            results = self._hybrid_retrieve(query, top_k)
        else:
            # 仅向量检索
            results = self._vector_retrieve(query, top_k)
        
        # 过滤低分结果
        if score_threshold > 0:
            results = [
                doc for doc in results 
                if doc.metadata.get("score", 0) >= score_threshold
            ]
        
        return results[:top_k]
    
    def _vector_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Document]:
        """向量检索"""
        results = self.vector_store.similarity_search_with_score(
            query, 
            k=top_k
        )
        
        # 将分数添加到metadata
        documents = []
        for doc, score in results:
            doc.metadata["score"] = float(score)
            doc.metadata["retrieval_method"] = "vector"
            documents.append(doc)
        
        return documents
    
    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Document]:
        """
        混合检索
        
        使用RRF（Reciprocal Rank Fusion）融合向量检索和BM25检索结果
        """
        # 向量检索
        vector_results = self._vector_retrieve(query, top_k * 2)
        
        # BM25检索
        bm25_results = self._bm25_retrieve(query, top_k * 2)
        
        # RRF融合
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            k=60  # RRF参数
        )
        
        return fused_results
    
    def _bm25_retrieve(
        self,
        query: str,
        top_k: int
    ) -> List[Document]:
        """BM25检索"""
        # 分词
        tokenized_query = list(query)
        
        # 获取BM25分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取top_k索引
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 构建结果
        documents = []
        for idx in top_indices:
            doc = self.bm25_documents[idx]
            # 创建新的Document对象以避免修改原对象
            new_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "score": float(scores[idx]),
                    "retrieval_method": "bm25"
                }
            )
            documents.append(new_doc)
        
        return documents
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        k: int = 60
    ) -> List[Document]:
        """
        倒数排名融合（RRF）
        
        RRF公式: score(d) = Σ 1/(k + rank(d))
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果
            k: RRF参数
            
        Returns:
            融合后的文档列表
        """
        # 文档ID到RRF分数的映射
        rrf_scores: Dict[str, float] = {}
        document_map: Dict[str, Document] = {}
        
        # 处理向量检索结果
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            document_map[doc_id] = doc
        
        # 处理BM25检索结果
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in document_map:
                document_map[doc_id] = doc
        
        # 按RRF分数排序
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # 构建最终结果
        results = []
        for doc_id in sorted_doc_ids:
            doc = document_map[doc_id]
            # 更新分数为RRF分数
            doc.metadata["score"] = rrf_scores[doc_id]
            doc.metadata["retrieval_method"] = "hybrid"
            results.append(doc)
        
        return results
    
    def add_documents(self, documents: List[Document]):
        """
        添加新文档
        
        同时更新向量数据库和BM25索引
        """
        # 添加到向量数据库
        self.vector_store.add_documents(documents)
        
        # 更新BM25索引
        if self.use_hybrid:
            self.bm25_documents.extend(documents)
            self._build_bm25_index(self.bm25_documents)
        
        logger.info(f"添加 {len(documents)} 个文档到检索器")
