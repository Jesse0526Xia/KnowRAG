# 智能文档知识库问答系统（RAG）

基于检索增强生成（RAG）技术构建的企业级知识库问答系统。

## 项目特点

- 支持多格式文档：PDF、Word、Markdown、TXT
- 混合检索策略：向量检索 + BM25关键词检索
- Cross-Encoder重排序提升检索精度
- 多轮对话记忆与上下文管理
- 答案溯源与可信度评估
- 流式输出支持

## 技术栈

- **LLM**: GPT-3.5/4, Claude, 通义千问
- **Embedding**: text2vec-base-chinese, bge-large-zh
- **向量数据库**: Milvus
- **RAG框架**: LangChain
- **Web框架**: FastAPI + Streamlit
- **文档处理**: PyMuPDF, python-docx

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制配置文件模板：
```bash
cp config/config.yaml.example config/config.yaml
```

修改配置文件，填入你的API密钥。

### 3. 启动服务

启动API服务：
```bash
python src/main.py --mode api
```

启动Streamlit界面：
```bash
streamlit run src/app/streamlit_app.py
```

## 项目结构

```
rag-knowledge-base/
├── config/              # 配置文件
├── src/
│   ├── app/            # 应用层（API、界面）
│   ├── core/           # RAG核心引擎
│   ├── document/       # 文档处理
│   ├── embedding/      # 向量化
│   ├── vectorstore/    # 向量数据库
│   └── utils/          # 工具函数
├── tests/              # 测试代码
├── data/               # 数据目录
└── docs/               # 文档
```

## 核心功能

### 文档上传与处理

```python
from src.document.processor import DocumentProcessor

processor = DocumentProcessor()
chunks = processor.process("path/to/document.pdf")
```

### 知识库问答

```python
from src.core.rag_engine import RAGEngine

engine = RAGEngine()
result = engine.query("什么是机器学习？")

print(result["answer"])        # 答案
print(result["sources"])       # 来源文档
print(result["confidence"])    # 可信度
```
