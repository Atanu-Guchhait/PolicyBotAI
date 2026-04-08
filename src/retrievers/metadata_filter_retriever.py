from typing import Dict, Any, Optional, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.retrievers.chroma_metadata_filter import build_chroma_filter

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
    
    #  Use the Chroma filter right here to translate the dictionary!
    chroma_formatted_filter = build_chroma_filter(filters) if filters else None

    #  Pass the translated filter to the database
    return vectordb.similarity_search(
        query=query,
        k=k,
        filter=chroma_formatted_filter 
    )