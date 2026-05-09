"""
Milvus向量数据库操作
"""

from typing import List, Tuple, Optional
from loguru import logger

from langchain.schema import Document
from langchain_community.vectorstores import Milvus
from langchain.embeddings import Embeddings

from ..embedding.embedder import Embedder


class MilvusVectorStore:
    """Milvus向量数据库封装"""
    
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "knowledge_base",
        host: str = "localhost",
        port: int = 19530,
        drop_old: bool = False
    ):
        """
        初始化Milvus向量数据库
        
        Args:
            embedding_function: 向量化函数
            collection_name: 集合名称
            host: Milvus服务地址
            port: Milvus服务端口
            drop_old: 是否删除已存在的集合
        """
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.host = host
        self.port = port
        
        # Milvus连接参数
        self.connection_args = {
            "host": host,
            "port": port
        }
        
        # 向量存储实例（延迟初始化）
        self.vector_store: Optional[Milvus] = None
        self.drop_old = drop_old
        
        logger.info(f"Milvus向量数据库初始化: {host}:{port}, 集合: {collection_name}")
    
    def _init_vector_store(self, documents: List[Document] = None):
        """
        初始化向量存储
        
        Args:
            documents: 初始文档列表
        """
        if documents:
            # 从文档创建向量存储
            self.vector_store = Milvus.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                drop_old=self.drop_old
            )
        else:
            # 创建空的向量存储
            self.vector_store = Milvus(
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
                connection_args=self.connection_args
            )
    
    def add_documents(self, documents: List[Document]):
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
        """
        if not documents:
            return
        
        if self.vector_store is None:
            self._init_vector_store(documents)
        else:
            # 为每个文档生成唯一ID
            ids = [str(hash(doc.page_content)) for doc in documents]
            
            self.vector_store.add_documents(
                documents=documents,
                ids=ids
            )
        
        logger.info(f"添加 {len(documents)} 个文档到向量数据库")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            相似文档列表
        """
        if self.vector_store is None:
            logger.warning("向量数据库为空")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        相似度搜索（带分数）
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            (文档, 分数) 元组列表
        """
        if self.vector_store is None:
            logger.warning("向量数据库为空")
            return []
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def delete_collection(self):
        """删除集合"""
        if self.vector_store:
            self.vector_store.col.drop()
            logger.info(f"集合 {self.collection_name} 已删除")
    
    def get_collection_stats(self) -> dict:
        """
        获取集合统计信息
        
        Returns:
            统计信息字典
        """
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        try:
            stats = self.vector_store.col.stats()
            return {
                "row_count": stats["row_count"],
                "collection_name": self.collection_name
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"status": "error", "message": str(e)}


class EmbeddingFunction(Embeddings):
    """
    LangChain Embeddings接口封装
    将自定义的Embedder包装为LangChain兼容的接口
    """
    
    def __init__(self, embedder: Embedder):
        """
        初始化
        
        Args:
            embedder: 自定义的Embedder实例
        """
        self.embedder = embedder
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档"""
        embeddings = self.embedder.embed_batch(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """向量化查询"""
        embedding = self.embedder.embed(text)
        return embedding.tolist()
