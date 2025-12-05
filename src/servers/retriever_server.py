from fastapi import FastAPI
from langserve import add_routes
from src.utils.retriever import create_documents, Retriever, RetrieverBM25
from src.utils.base_models import RetrieverArgs
import toml
from typing import Dict, Any, Optional
import uvicorn
import argparse
from langchain_core.retrievers import BaseRetriever
from argparse import Namespace


def load_config() -> Dict[str, Any]:
    """Load configuration from TOML file located in root folder."""
    return toml.load("src/config.toml")


def initialize_components(
    config: Dict[str, Any], bm25: bool, db_load_folder: Optional[str] = None
) -> Retriever:
    args = RetrieverArgs(**config["retriever"])
    if db_load_folder is None or args.ensemble or bm25:
        documents = create_documents(**config["documents"])
    else:
        documents = None

    if bm25:
        return RetrieverBM25(documents)
    else:
        return Retriever(documents=documents, args=args, db_load_folder=db_load_folder)


def create_app(retriever: BaseRetriever) -> FastAPI:
    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="Spin up a simple api server using Langchain's Runnable interfaces",
    )
    add_routes(app, retriever)
    return app


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Run the LangChain server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host for the FastAPI server"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the FastAPI server"
    )
    parser.add_argument(
        "--db-load-folder", type=str, default=None, help="Folder to load FAISS DB from"
    )
    parser.add_argument(
        "--db-save-folder", type=str, default=None, help="Folder to save FAISS DB to"
    )
    parser.add_argument(
        "--bm25",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use BM25 instead of FAISS for retrieval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    retriever = initialize_components(config, args.bm25, args.db_load_folder)

    if args.db_save_folder is not None:
        retriever.save_db(args.db_save_folder)

    app = create_app(retriever.retriever)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
