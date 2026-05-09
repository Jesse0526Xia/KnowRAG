"""
文本向量化器
使用Sentence-Transformers模型进行文本向量化
"""

from typing import List
from loguru import logger

from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """文本向量化器"""
    
    def __init__(
        self,
        model_name: str = "text2vec-base-chinese",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        初始化向量化器
        
        Args:
            model_name: 模型名称
            device: 运行设备 (cuda/cpu)
            batch_size: 批处理大小
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        
        # 获取向量维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"向量化器初始化完成，模型: {model_name}, 维度: {self.embedding_dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        单个文本向量化
        
        Args:
            text: 文本字符串
            
        Returns:
            向量（numpy数组）
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        批量文本向量化
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            向量矩阵（numpy数组）
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"批量向量化完成，数量: {len(texts)}")
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            相似度分数
        """
        # 归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 余弦相似度
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        return float(similarity)
    
    def similarity_batch(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        批量计算相似度
        
        Args:
            query_embedding: 查询向量
            doc_embeddings: 文档向量矩阵
            
        Returns:
            相似度分数数组
        """
        # 归一化
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        
        if query_norm == 0:
            return np.zeros(len(doc_embeddings))
        
        # 余弦相似度
        similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)
        
        return similarities
    
    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self.embedding_dim
