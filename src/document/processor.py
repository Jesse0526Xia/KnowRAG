"""
文档处理器
实现文档加载、切分、向量化的完整流水线
"""

from typing import List, Optional
from pathlib import Path
from loguru import logger

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .loader import DocumentLoader
from ..embedding.embedder import Embedder
from ..vectorstore.milvus_store import MilvusVectorStore


class DocumentProcessor:
    """文档处理流水线"""
    
    def __init__(
        self,
        loader: DocumentLoader,
        embedder: Embedder,
        vector_store: MilvusVectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化文档处理器
        
        Args:
            loader: 文档加载器
            embedder: 向量化器
            vector_store: 向量数据库
            chunk_size: 切分块大小
            chunk_overlap: 切分块重叠大小
        """
        self.loader = loader
        self.embedder = embedder
        self.vector_store = vector_store
        
        # 文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]
        )
        
        logger.info(f"文档处理器初始化完成，chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def process(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        add_to_vector_store: bool = True
    ) -> List[Document]:
        """
        处理单个文档
        
        Args:
            file_path: 文档路径
            metadata: 额外的元数据
            add_to_vector_store: 是否添加到向量数据库
            
        Returns:
            切分后的文档块列表
        """
        logger.info(f"开始处理文档: {file_path}")
        
        # 1. 加载文档
        documents = self.loader.load(file_path)
        logger.info(f"加载了 {len(documents)} 个文档")
        
        # 2. 添加元数据
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        # 3. 切分文档
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"切分为 {len(chunks)} 个文档块")
        
        # 4. 添加到向量数据库
        if add_to_vector_store:
            self.vector_store.add_documents(chunks)
            logger.info("文档块已添加到向量数据库")
        
        return chunks
    
    def process_batch(
        self,
        file_paths: List[str],
        metadata_list: Optional[List[dict]] = None,
        add_to_vector_store: bool = True
    ) -> List[Document]:
        """
        批量处理文档
        
        Args:
            file_paths: 文档路径列表
            metadata_list: 元数据列表
            add_to_vector_store: 是否添加到向量数据库
            
        Returns:
            所有切分后的文档块列表
        """
        all_chunks = []
        
        for i, file_path in enumerate(file_paths):
            metadata = metadata_list[i] if metadata_list else None
            chunks = self.process(file_path, metadata, add_to_vector_store=False)
            all_chunks.extend(chunks)
        
        # 批量添加到向量数据库
        if add_to_vector_store and all_chunks:
            self.vector_store.add_documents(all_chunks)
            logger.info(f"批量添加 {len(all_chunks)} 个文档块到向量数据库")
        
        return all_chunks
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: List[str] = None,
        add_to_vector_store: bool = True
    ) -> List[Document]:
        """
        处理目录下的所有文档
        
        Args:
            directory: 目录路径
            recursive: 是否递归处理子目录
            file_extensions: 文件扩展名过滤
            add_to_vector_store: 是否添加到向量数据库
            
        Returns:
            所有切分后的文档块列表
        """
        directory_path = Path(directory)
        
        # 默认支持的文件扩展名
        if file_extensions is None:
            file_extensions = [".pdf", ".docx", ".doc", ".txt", ".md"]
        
        # 查找所有文件
        if recursive:
            files = []
            for ext in file_extensions:
                files.extend(directory_path.rglob(f"*{ext}"))
        else:
            files = []
            for ext in file_extensions:
                files.extend(directory_path.glob(f"*{ext}"))
        
        file_paths = [str(f) for f in files]
        logger.info(f"在目录 {directory} 中找到 {len(file_paths)} 个文档")
        
        return self.process_batch(file_paths, add_to_vector_store=add_to_vector_store)
    
    def update_chunk_strategy(
        self,
        chunk_size: int,
        chunk_overlap: int
    ):
        """
        更新切分策略
        
        Args:
            chunk_size: 新的切分块大小
            chunk_overlap: 新的切分块重叠大小
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]
        )
        
        logger.info(f"更新切分策略: chunk_size={chunk_size}, overlap={chunk_overlap}")
