from typing import List, Optional
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings

from src.ingest import ingest_pipeline
from utils.logger import setup_logger


# ******************************* CONSTANT **************************************
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "hr_policies"

logger = setup_logger(__name__)

# ******************************* GLOBAL CACHE ***********************************
_embedding_model = None


# ******************************* LOAD EMBEDDING MODEL ***************************
def load_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading embedding model (only once)...")

        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

    return _embedding_model


# **************************** CREATE VECTOR STORE **********************************
def create_vector_store(
    documents: List[Document],
    persist_dir: str = PERSIST_DIR
):
    embedding_model = load_embedding_model()

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        client_settings=Settings(anonymized_telemetry=False)
    )

    vectordb.add_documents(documents)

    logger.info("Vector DB created and persisted")

    return vectordb


# ******************************** LOAD EXISTING VECTOR STORE ******************************
def load_vector_store(persist_dir: str = PERSIST_DIR):
    embedding_model = load_embedding_model()

    logger.info("Loading existing vector DB...")

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME
    )


