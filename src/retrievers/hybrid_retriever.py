from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_chroma import Chroma

from src.retrievers.mmr_retriever import get_base_retriever
from src.retrievers.multiquery_retriever import get_multiquery_retriever

from utils.logger import setup_logger

logger = setup_logger(__name__)


# ******************************* HYBRID RETRIEVER ****************************
def retrieve_documents(
    vectordb: Chroma,
    query: str,
    llm: Optional[BaseLanguageModel] = None,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 5
) -> List[Document]:

    #  PRE-FILTER (best for strict queries)
    if filters:
        docs = vectordb.similarity_search(
            query=query,
            k=20,
            filter=filters
        )

        if docs:
            logger.info(f"Pre-filtered docs: {len(docs)}")
            return docs[:k]

        logger.warning("No docs found with pre-filter, falling back...")

    # RETRIEVER SELECTION
    if llm and not filters:
        retriever = get_multiquery_retriever(vectordb, llm, k)
    else:
        retriever = get_base_retriever(vectordb, k)

    docs = retriever.invoke(query)

    # POST-FILTER (refinement)
    if filters:
        filtered_docs = [
            doc for doc in docs
            if all(
                str(doc.metadata.get(key, "")).lower() == str(value).lower()
                for key, value in filters.items()
            )
        ]

        if filtered_docs:
            logger.info(f"Post-filtered docs: {len(filtered_docs)}")
            return filtered_docs

    return docs