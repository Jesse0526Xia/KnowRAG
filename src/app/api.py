"""
FastAPI接口
提供RESTful API服务
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from loguru import logger

from ..core.rag_engine import RAGEngine


# 请求模型
class QueryRequest(BaseModel):
    """查询请求"""
    question: str
    top_k: Optional[int] = 10
    use_history: Optional[bool] = True
    return_sources: Optional[bool] = True


class QueryResponse(BaseModel):
    """查询响应"""
    question: str
    answer: str
    confidence: float
    sources: Optional[List[Dict[str, Any]]] = None
    rewritten_query: Optional[str] = None


# 创建FastAPI应用
app = FastAPI(
    title="RAG知识库问答系统API",
    description="基于检索增强生成的智能问答系统",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局RAG引擎实例（需要初始化）
rag_engine: RAGEngine = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global rag_engine
    # 这里需要初始化RAG引擎
    # rag_engine = init_rag_engine()
    logger.info("API服务启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    logger.info("API服务关闭")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG知识库问答系统API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    问答接口
    
    Args:
        request: 查询请求
        
    Returns:
        查询响应
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG引擎未初始化")
    
    try:
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            use_history=request.use_history,
            return_sources=request.return_sources
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式问答接口
    
    Args:
        request: 查询请求
        
    Returns:
        流式响应
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG引擎未初始化")
    
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        for chunk in rag_engine.stream_query(
            question=request.question,
            top_k=request.top_k,
            use_history=request.use_history
        ):
            yield json.dumps({"chunk": chunk}) + "\n"
    
    return StreamingResponse(generate(), media_type="application/json")


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    文档上传接口
    
    Args:
        file: 上传的文件
        
    Returns:
        上传结果
    """
    try:
        # 保存文件
        file_path = f"data/documents/{file.filename}"
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 处理文档（需要实现）
        # processor.process(file_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "文档上传成功"
        }
    
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch-upload")
async def batch_upload_documents(files: List[UploadFile] = File(...)):
    """
    批量文档上传接口
    
    Args:
        files: 上传的文件列表
        
    Returns:
        上传结果
    """
    results = []
    
    for file in files:
        try:
            file_path = f"data/documents/{file.filename}"
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            results.append({
                "filename": file.filename,
                "status": "success"
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "results": results
    }


@app.delete("/history")
async def clear_history():
    """清空对话历史"""
    if rag_engine:
        rag_engine.clear_history()
    
    return {"status": "success", "message": "对话历史已清空"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "rag_engine": "initialized" if rag_engine else "not_initialized"
    }


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """运行API服务"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()
