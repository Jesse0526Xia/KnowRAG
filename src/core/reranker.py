"""
重排序器
使用Cross-Encoder对检索结果进行重排序
"""

from typing import List, Tuple
from loguru import logger

from langchain.schema import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder重排序器"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cuda",
        top_k: int = 5
    ):
        """
        初始化重排序器
        
        Args:
            model_name: 模型名称
            device: 运行设备 (cuda/cpu)
            top_k: 重排序后保留的文档数量
        """
        self.model = CrossEncoder(model_name, device=device)
        self.top_k = top_k
        
        logger.info(f"重排序器初始化完成，模型: {model_name}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 重排序后保留的文档数量
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        top_k = top_k or self.top_k
        
        # 构建查询-文档对
        pairs = [(query, doc.page_content) for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 按分数排序
        scored_docs: List[Tuple[Document, float]] = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 构建重排序后的文档列表
        reranked_docs = []
        for doc, score in scored_docs[:top_k]:
            # 创建新的Document对象
            new_doc = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "rerank_score": float(score),
                    "score": float(score)  # 更新分数
                }
            )
            reranked_docs.append(new_doc)
        
        logger.info(f"重排序完成，从 {len(documents)} 个文档中选出 {len(reranked_docs)} 个")
        
        return reranked_docs
    
    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Document]],
        top_k: int = None
    ) -> List[List[Document]]:
        """
        批量重排序
        
        Args:
            queries: 查询列表
            documents_list: 文档列表的列表
            top_k: 每个查询保留的文档数量
            
        Returns:
            重排序后的文档列表的列表
        """
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        
        return results
