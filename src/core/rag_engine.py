"""
RAG核心引擎
实现检索增强生成的完整流程
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .retriever import HybridRetriever
from .generator import Generator
from .reranker import Reranker


class RAGEngine:
    """RAG问答引擎"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Generator,
        reranker: Optional[Reranker] = None,
        max_history: int = 5
    ):
        """
        初始化RAG引擎
        
        Args:
            retriever: 检索器
            generator: 生成器
            reranker: 重排序器（可选）
            max_history: 最大对话历史轮数
        """
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        
        logger.info("RAG引擎初始化完成")
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        use_history: bool = True,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        执行问答
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_history: 是否使用对话历史
            return_sources: 是否返回来源文档
            
        Returns:
            包含答案、来源、可信度等信息的字典
        """
        logger.info(f"处理问题: {question}")
        
        # 1. 查询改写（基于对话历史）
        rewritten_query = self._rewrite_query(question, use_history)
        logger.debug(f"改写后的查询: {rewritten_query}")
        
        # 2. 检索相关文档
        retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=top_k*2)
        logger.info(f"检索到 {len(retrieved_docs)} 个文档")
        
        # 3. 重排序（如果启用）
        if self.reranker:
            reranked_docs = self.reranker.rerank(
                rewritten_query, 
                retrieved_docs, 
                top_k=top_k
            )
            logger.info(f"重排序后保留 {len(reranked_docs)} 个文档")
        else:
            reranked_docs = retrieved_docs[:top_k]
        
        # 4. 构建上下文
        context = self._build_context(reranked_docs)
        
        # 5. 生成答案
        chat_history = self.conversation_history if use_history else []
        answer = self.generator.generate(
            question=question,
            context=context,
            chat_history=chat_history
        )
        
        # 6. 计算可信度
        confidence = self._calculate_confidence(answer, reranked_docs)
        
        # 7. 更新对话历史
        if use_history:
            self._update_history(question, answer)
        
        # 8. 构建返回结果
        result = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "rewritten_query": rewritten_query
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0)
                }
                for doc in reranked_docs
            ]
        
        logger.info(f"问答完成，可信度: {confidence:.2f}")
        return result
    
    def _rewrite_query(self, question: str, use_history: bool) -> str:
        """
        查询改写
        
        基于对话历史改写当前问题，使其更加完整和明确
        """
        if not use_history or not self.conversation_history:
            return question
        
        # 简单的查询改写逻辑（可以扩展为LLM改写）
        # 这里可以添加更复杂的改写逻辑
        return question
    
    def _build_context(self, documents: List[Any]) -> str:
        """
        构建上下文
        
        将检索到的文档拼接成上下文字符串
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[文档{i}]\n{doc.page_content}\n")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(
        self, 
        answer: str, 
        documents: List[Any]
    ) -> float:
        """
        计算答案可信度
        
        基于检索文档的相关性分数和答案长度计算可信度
        """
        if not documents:
            return 0.0
        
        # 平均检索分数
        avg_score = sum(
            doc.metadata.get("score", 0) for doc in documents
        ) / len(documents)
        
        # 答案长度因子（答案太短可能不可靠）
        length_factor = min(len(answer) / 100, 1.0)
        
        # 综合可信度
        confidence = avg_score * 0.7 + length_factor * 0.3
        
        return min(confidence, 1.0)
    
    def _update_history(self, question: str, answer: str):
        """更新对话历史"""
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": answer
        })
        
        # 保持历史记录在限制范围内
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history*2:]
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def stream_query(
        self,
        question: str,
        top_k: int = 10,
        use_history: bool = True
    ):
        """
        流式问答
        
        逐步返回生成的答案，提升用户体验
        """
        # 检索和重排序（与普通查询相同）
        rewritten_query = self._rewrite_query(question, use_history)
        retrieved_docs = self.retriever.retrieve(rewritten_query, top_k=top_k*2)
        
        if self.reranker:
            reranked_docs = self.reranker.rerank(
                rewritten_query, 
                retrieved_docs, 
                top_k=top_k
            )
        else:
            reranked_docs = retrieved_docs[:top_k]
        
        context = self._build_context(reranked_docs)
        chat_history = self.conversation_history if use_history else []
        
        # 流式生成
        for chunk in self.generator.stream_generate(
            question=question,
            context=context,
            chat_history=chat_history
        ):
            yield chunk
        
        # 更新历史（需要在流式生成完成后）
        if use_history:
            # 这里需要收集完整的答案来更新历史
            pass
