"""
答案生成器
基于LLM生成答案
"""

from typing import List, Dict, Any, Generator
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class Generator:
    """答案生成器"""
    
    def __init__(
        self,
        llm: Any = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        初始化生成器
        
        Args:
            llm: 语言模型实例
            system_prompt: 系统提示词
            temperature: 生成温度
            max_tokens: 最大生成token数
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info("生成器初始化完成")
    
    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是一个专业的知识库问答助手。请基于提供的上下文回答用户问题。

回答要求：
1. 答案必须基于提供的上下文信息，不要编造或臆造信息
2. 如果上下文中没有足够的信息回答问题，请明确告知用户
3. 回答要简洁、准确、有条理
4. 可以引用具体的文档来源来支持你的回答
5. 如果问题涉及多个方面，请分点回答"""
    
    def generate(
        self,
        question: str,
        context: str,
        chat_history: List[Dict] = None,
        temperature: float = None
    ) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史
            temperature: 生成温度（可选，覆盖默认值）
            
        Returns:
            生成的答案
        """
        # 构建消息列表
        messages = self._build_messages(question, context, chat_history)
        
        # 设置温度
        if temperature is not None:
            self.llm.temperature = temperature
        
        # 生成答案
        response = self.llm.invoke(messages)
        answer = response.content
        
        logger.debug(f"生成答案长度: {len(answer)}")
        return answer
    
    def stream_generate(
        self,
        question: str,
        context: str,
        chat_history: List[Dict] = None
    ) -> Generator[str, None, None]:
        """
        流式生成答案
        
        Args:
            question: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史
            
        Yields:
            答案的文本片段
        """
        # 构建消息列表
        messages = self._build_messages(question, context, chat_history)
        
        # 流式生成
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content
    
    def _build_messages(
        self,
        question: str,
        context: str,
        chat_history: List[Dict] = None
    ) -> List[Any]:
        """
        构建消息列表
        
        Args:
            question: 用户问题
            context: 上下文
            chat_history: 对话历史
            
        Returns:
            消息列表
        """
        messages = []
        
        # 系统提示词
        messages.append(SystemMessage(content=self.system_prompt))
        
        # 对话历史
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # 当前问题（带上下文）
        user_message = self._build_user_message(question, context)
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
    def _build_user_message(self, question: str, context: str) -> str:
        """
        构建用户消息
        
        将上下文和问题组合成完整的用户消息
        """
        prompt_template = """上下文信息：
{context}

用户问题：{question}

请基于上下文信息回答用户问题。如果上下文中没有相关信息，请明确说明。"""
        
        return prompt_template.format(
            context=context,
            question=question
        )
    
    def generate_with_sources(
        self,
        question: str,
        context: str,
        sources: List[Dict],
        chat_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        生成带来源标注的答案
        
        Args:
            question: 用户问题
            context: 上下文
            sources: 来源文档列表
            chat_history: 对话历史
            
        Returns:
            包含答案和来源的字典
        """
        # 生成答案
        answer = self.generate(question, context, chat_history)
        
        # 提取答案中引用的来源
        cited_sources = self._extract_cited_sources(answer, sources)
        
        return {
            "answer": answer,
            "cited_sources": cited_sources,
            "all_sources": sources
        }
    
    def _extract_cited_sources(
        self,
        answer: str,
        sources: List[Dict]
    ) -> List[Dict]:
        """
        提取答案中引用的来源
        
        简单实现：返回所有来源
        可以扩展为基于答案内容匹配来源
        """
        return sources
