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
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def load_config() -> Dict[str, Any]:
    """Load configuration from TOML file located in root folder."""
    return toml.load("src/config.toml")


def initialize_components(
    config: Dict[str, Any], bm25: bool, db_load_folder: Optional[str] = None
) -> Retriever:
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Initializing components...")
    # Convert empty string to None for optional fields
    retriever_config = config["retriever"].copy()
    if (
        "qdrant_api_key" in retriever_config
        and retriever_config["qdrant_api_key"] == ""
    ):
        retriever_config["qdrant_api_key"] = None
    if "rerank_model" in retriever_config and retriever_config["rerank_model"] == "":
        retriever_config["rerank_model"] = None

    args = RetrieverArgs(**retriever_config)
    logger.info(f"RetrieverArgs: {args}")

    if db_load_folder is None or args.ensemble or bm25:
        logger.info("Creating documents...")
        documents = create_documents(**config["documents"])
        logger.info(f"Documents created: {len(documents)} documents")
    else:
        logger.info("Skipping document creation, will load from existing DB")
        documents = None

    if bm25:
        logger.info("Creating BM25 retriever...")
        return RetrieverBM25(documents)
    else:
        logger.info("Creating Qdrant retriever...")
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
        "--db-load-folder",
        type=str,
        default=None,
        help="Collection name to load Qdrant DB from",
    )
    parser.add_argument(
        "--db-save-folder",
        type=str,
        default=None,
        help="Collection name to save Qdrant DB to (data is persisted on server)",
    )
    parser.add_argument(
        "--bm25",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use BM25 instead of Qdrant for retrieval",
    )
    return parser.parse_args()


def main() -> None:
    import logging

    logger = logging.getLogger(__name__)

    logger.info("Starting retriever server...")
    args = parse_args()
    logger.info(f"Arguments: {args}")

    logger.info("Loading config...")
    config = load_config()
    logger.info(f"Config loaded: {config}")

    logger.info("Initializing components...")
    retriever = initialize_components(config, args.bm25, args.db_load_folder)
    logger.info("Components initialized successfully")

    if args.db_save_folder is not None:
        logger.info(f"Saving DB to {args.db_save_folder}")
        retriever.save_db(args.db_save_folder)

    logger.info("Creating FastAPI app...")
    app = create_app(retriever.retriever)
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
