from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_chroma import Chroma

from src.retrievers.mmr_retriever import get_base_retriever
from src.retrievers.multiquery_retriever import get_multiquery_retriever
from src.retrievers.chroma_metadata_filter import build_chroma_filter
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

    # 1. Convert filters → Chroma format
    chroma_filter = build_chroma_filter(filters)
    
    # 2. PRE-FILTER ATTEMPT (Strict matching)
    if chroma_filter:
        logger.info(f"Attempting pre-filter retrieval with: {chroma_filter}")
        docs = vectordb.similarity_search(
            query=query,
            k=k,
            filter=chroma_filter   
        )

        if docs:
            logger.info(f"Successfully retrieved {len(docs)} docs with pre-filter.")
            return docs

        logger.warning("No docs found with strict pre-filter. Falling back to semantic search...")

    # 3. FALLBACK: SEMANTIC SEARCH (Retriever Selection)
    # If filters failed or were not provided, we rely on text similarity.
    if llm and not filters:
        retriever = get_multiquery_retriever(vectordb, llm, k)
    else:
        # We use the base retriever (MMR) to get a diverse set of results
        retriever = get_base_retriever(vectordb, k)

    # Invoke retriever (this ignores the strict chroma_filter metadata)
    docs = retriever.invoke(query)

    # 4. SMART POST-FILTER (Refinement)
    if filters and docs:
        # We try to match as many filters as possible, but we don't 
        # return an empty list if one filter (like a specific year) fails.
        filtered_docs = []
        for doc in docs:
            match_count = 0
            for key, value in filters.items():
                doc_val = str(doc.metadata.get(key, "")).lower()
                target_val = str(value).lower()
                if target_val in doc_val or doc_val in target_val:
                    match_count += 1
            
            # If at least one major filter matches (e.g., department), prioritize it
            if match_count > 0:
                filtered_docs.append(doc)

        if filtered_docs:
            logger.info(f"Post-filtered matches found: {len(filtered_docs)}")
            return filtered_docs

    # 5. FINAL FALLBACK
    # If even post-filtering finds nothing, return the raw semantic results 
    # so the LLM can at least try to find the answer in the text.
    if not docs:
        logger.error("Total retrieval failure: No documents found even in fallback.")
    
    return docs