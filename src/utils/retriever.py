#!/usr/bin/env python
import logging
import os
from typing import List, Any, Optional, Callable

import pandas as pd
from langchain_community.document_loaders import (
    DataFrameLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain.docstore.document import Document
from src.utils.base_models import RetrieverArgs
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.bm25 import BM25Retriever, default_preprocessing_func
from langchain_core.retrievers import BaseRetriever

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

LOADER_MAPPING = {
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


class Retriever:
    def __init__(
        self,
        documents: Optional[List[Document]],
        args: RetrieverArgs,
        db_load_folder: Optional[str] = None,
        k: int = 4,
    ):
        self.args = args
        logger.info("Retriever init")
        self._create_embedding_model(
            model_name=self.args.embedding_model,
            normalize_embeddings=self.args.normalize_embeddings,
        )

        logger.info(f"Qdrant init with URL: {self.args.qdrant_url}")
        try:
            # Convert empty string to None for api_key
            api_key = self.args.qdrant_api_key if self.args.qdrant_api_key else None
            self.qdrant_client = QdrantClient(
                url=self.args.qdrant_url,
                api_key=api_key,
                timeout=30,  # Add timeout
            )
            logger.info("Qdrant client initialized")
            # Test connection
            logger.info("Testing Qdrant connection...")
            collections = self.qdrant_client.get_collections()
            logger.info(
                f"Qdrant connection successful. Existing collections: {[c.name for c in collections.collections]}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
            logger.error("Make sure Qdrant server is running and accessible")
            raise

        logger.info(
            f"db_load_folder: {db_load_folder}, documents: {documents is not None}"
        )
        if db_load_folder is None:
            logger.info("Creating new database...")
            self.create_db(documents)
        else:
            logger.info(f"Loading database from folder: {db_load_folder}")
            self.load_db(db_load_folder)

        self.retriever = self.db.as_retriever(search_kwargs={"k": k})

        if self.args.ensemble:
            bm25_retriever = self._create_bm25(documents, k)
            self.retriever = self._create_ensemble(
                self.retriever, bm25_retriever, weights=[0.7, 0.2]
            )
            logger.info(f"Ensemble retriever created {self.retriever}")

        if self.args.rerank_model is not None:
            self.base_retriever = self.retriever
            self.retriever = self._add_rerank(self.args.rerank_model)
            logger.info(f"Reranking added {self.retriever}")

    def _create_bm25(
        self,
        documents: Optional[List[Document]],
        k: int,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
    ) -> BM25Retriever:
        bm25_retriever = BM25Retriever.from_documents(
            documents, preprocess_func=preprocess_func
        )
        bm25_retriever.k = k
        logger.info(f"BM25 retriever created {bm25_retriever}")
        return bm25_retriever

    def _create_ensemble(
        self,
        retriever_1: BaseRetriever,
        retriever_2: BaseRetriever,
        weights: List[float] = [0.5, 0.5],
    ) -> EnsembleRetriever:
        return EnsembleRetriever(retrievers=[retriever_1, retriever_2], weights=weights)

    def _create_embedding_model(
        self, model_name: str, normalize_embeddings: bool = False
    ) -> HuggingFaceEmbeddings:
        logger.info(f"Creating embedding model: {model_name}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
            show_progress=True,
            cache_folder="/qdrant_indexes",
        )
        logger.info(f"Embedding model created: {self.embedding_model}")

    def load_db(self, folder: str) -> None:
        collection_name = folder if folder else self.args.qdrant_collection_name
        self.db = Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding_model,
        )
        logger.info(f"Db loaded from collection {collection_name}")

    def create_db(self, documents: Optional[List[Document]]) -> None:
        logger.info(f"Creating db with {len(documents) if documents else 0} documents")
        if documents is None or len(documents) == 0:
            logger.warning("No documents provided for database creation!")
            raise ValueError("Cannot create database without documents")

        try:
            logger.info(
                f"Calling Qdrant.from_documents with collection: {self.args.qdrant_collection_name}"
            )
            logger.info(
                f"Processing {len(documents)} documents - this may take a while as embeddings are being created..."
            )

            # Check if collection already exists
            try:
                collection_info = self.qdrant_client.get_collection(
                    self.args.qdrant_collection_name
                )
                logger.warning(
                    f"Collection {self.args.qdrant_collection_name} already exists with {collection_info.points_count} points"
                )
                logger.info("Will recreate collection...")
            except Exception as e:
                logger.info(
                    f"Collection {self.args.qdrant_collection_name} does not exist, will create new one (error: {e})"
                )

            logger.info("Starting Qdrant.from_documents call...")
            import time

            start_time = time.time()

            # Convert empty string to None for api_key
            api_key = self.args.qdrant_api_key if self.args.qdrant_api_key else None
            self.db = Qdrant.from_documents(
                documents,
                self.embedding_model,
                url=self.args.qdrant_url,
                api_key=api_key,
                collection_name=self.args.qdrant_collection_name,
            )

            elapsed_time = time.time() - start_time
            logger.info(
                f"Db created successfully with collection {self.args.qdrant_collection_name} in {elapsed_time:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to create database: {e}", exc_info=True)
            raise

    def save_db(self, folder: str) -> None:
        logger.info(
            f"Qdrant data is persisted on server. Collection: {self.args.qdrant_collection_name}"
        )

    def _add_rerank(
        self, model_name: str, top_n: int = 4
    ) -> ContextualCompressionRetriever:
        model = HuggingFaceCrossEncoder(model_name=model_name)
        compressor = CrossEncoderReranker(model=model, top_n=top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.base_retriever
        )


def preprocess(text: str) -> List[str]:
    return text.lower().split()


class RetrieverBM25(Retriever):
    def __init__(self, documents: Optional[List[Document]], k: int = 4):
        self.retriever = self._create_bm25(documents, k, preprocess)
        logger.info(f"Retriever created {self.retriever}")


def create_documents(
    folder: str,
    target_column: str = "text",
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> Any:
    logger.info(f"Starting document creation from folder: {folder}")
    raw_docs = []
    logger.info(
        f"folder: {folder} target_column: {target_column} chunk_size: {chunk_size} chunk_overlap: {chunk_overlap}"
    )

    if not os.path.exists(folder):
        logger.error(f"Folder does not exist: {folder}")
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    files = os.listdir(folder)
    logger.info(f"Found {len(files)} items in folder")

    for filename in files:
        full_path = os.path.join(folder, filename)
        if not os.path.isfile(full_path):
            logger.debug(f"Skipping non-file: {filename}")
            continue

        logger.info(f"Processing file {filename}...")
        ext = "." + filename.rsplit(".", 1)[-1]

        try:
            if ext == ".csv":
                logger.debug(f"Loading CSV file: {filename}")
                data = pd.read_csv(
                    full_path,
                    sep=",",
                    quotechar='"',
                )
                loader = DataFrameLoader(data, page_content_column=target_column)
            elif ext in LOADER_MAPPING:
                logger.debug(f"Loading file with {ext} loader: {filename}")
                loader_class, loader_args = LOADER_MAPPING[ext]
                loader = loader_class(full_path, **loader_args)
            else:
                logger.warning(f"Unsupported file type {ext}, skipping...")
                continue

            logger.debug(f"Loading documents from {filename}...")
            loaded_docs = loader.load()
            logger.info(f"Loaded {len(loaded_docs)} documents from {filename}")
            raw_docs.extend(loaded_docs)
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            continue

    logger.info(f"Total documents before splitting: {len(raw_docs)}")
    if len(raw_docs) == 0:
        logger.warning("No documents were loaded!")
        return []

    logger.info("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        length_function=len,
        is_separator_regex=False,
        chunk_size=chunk_size,  # min: 50, max: 2000
        chunk_overlap=chunk_overlap,  # min: 0, max: 500,
    )
    documents = text_splitter.split_documents(raw_docs)
    logger.info(f"Total documents after splitting: {len(documents)}")

    return documents
