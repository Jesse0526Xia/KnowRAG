"""
主入口文件
"""

import argparse
from loguru import logger

from .app.api import run_api


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG知识库问答系统")
    parser.add_argument(
        "--mode",
        type=str,
        default="api",
        choices=["api", "streamlit"],
        help="运行模式: api 或 streamlit"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API服务地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API服务端口"
    )
    
    args = parser.parse_args()
    
    logger.info(f"启动模式: {args.mode}")
    
    if args.mode == "api":
        # 启动API服务
        run_api(host=args.host, port=args.port)
    
    elif args.mode == "streamlit":
        # 启动Streamlit（需要通过命令行启动）
        logger.info("请使用以下命令启动Streamlit:")
        logger.info("streamlit run src/app/streamlit_app.py")


if __name__ == "__main__":
    main()
