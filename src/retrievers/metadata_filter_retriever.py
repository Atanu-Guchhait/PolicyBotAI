from typing import Dict, Any, Optional, List
from langchain_core.documents import Document
from langchain_chroma import Chroma


# *************************** PRE-FILTER ***************************
def retrieve_with_filter(
    vectordb: Chroma,
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 5
) -> List[Document]:
    """
    Retrieve documents using metadata filtering (pre-filtering)
    """

    return vectordb.similarity_search(
        query=query,
        k=k,
        filter=filters if filters else None
    )