"""
Streamlit交互界面
提供用户友好的问答界面
"""

import streamlit as st
from typing import List, Dict
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.rag_engine import RAGEngine
from src.core.retriever import HybridRetriever
from src.core.generator import Generator
from src.core.reranker import Reranker
from src.document.processor import DocumentProcessor
from src.document.loader import DocumentLoader
from src.embedding.embedder import Embedder
from src.vectorstore.milvus_store import MilvusVectorStore, EmbeddingFunction
from loguru import logger


def init_rag_engine():
    """初始化RAG引擎"""
    # 这里需要根据实际配置初始化各个组件
    # 简化示例，实际使用时需要完整配置
    
    # 初始化向量化器
    embedder = Embedder(model_name="text2vec-base-chinese", device="cpu")
    embedding_function = EmbeddingFunction(embedder)
    
    # 初始化向量数据库
    vector_store = MilvusVectorStore(
        embedding_function=embedding_function,
        collection_name="knowledge_base"
    )
    
    # 初始化检索器
    retriever = HybridRetriever(
        vector_store=vector_store,
        use_hybrid=False  # 简化示例
    )
    
    # 初始化生成器
    generator = Generator()
    
    # 初始化RAG引擎
    engine = RAGEngine(
        retriever=retriever,
        generator=generator
    )
    
    return engine


def main():
    """主函数"""
    st.set_page_config(
        page_title="智能知识库问答系统",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 智能知识库问答系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("系统设置")
        
        # 文档上传
        st.subheader("文档管理")
        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("处理文档"):
                with st.spinner("正在处理文档..."):
                    # 这里添加文档处理逻辑
                    st.success(f"成功处理 {len(uploaded_files)} 个文档")
        
        st.markdown("---")
        
        # 检索参数
        st.subheader("检索参数")
        top_k = st.slider("检索文档数", 1, 20, 5)
        use_rerank = st.checkbox("启用重排序", value=True)
        
        st.markdown("---")
        
        # 对话管理
        st.subheader("对话管理")
        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.success("对话历史已清空")
    
    # 主界面
    # 初始化对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 如果是助手消息，显示来源
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("查看来源"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**来源 {i}**")
                        st.text(source["content"][:200] + "...")
                        st.caption(f"相关度: {source.get('score', 0):.3f}")
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题"):
        # 显示用户消息
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                # 这里应该调用RAG引擎
                # 简化示例，实际使用时需要完整实现
                answer = f"这是一个示例回答。您的问题是：{prompt}"
                sources = []
                
                # 显示回答
                st.markdown(answer)
                
                # 显示来源
                if sources:
                    with st.expander("查看来源"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**来源 {i}**")
                            st.text(source["content"][:200] + "...")
                            st.caption(f"相关度: {source.get('score', 0):.3f}")
                
                # 保存到对话历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })


if __name__ == "__main__":
    main()
