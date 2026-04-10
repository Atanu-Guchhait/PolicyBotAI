from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_core.language_models.base import BaseLanguageModel
from langchain_chroma import Chroma

from src.retrievers.mmr_retriever import get_base_retriever
from src.retrievers.multiquery_retriever import get_multiquery_retriever
from src.retrievers.metadata_filter_retriever import retrieve_with_filter 
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
    """
    Orchestrates multi-stage retrieval:
    1. Pre-filter (Strict DB check)
    2. Semantic Search (Broad Fallback)
    3. Intelligent Post-Filter (Ranking by metadata relevance, version, and keywords)
    """

    # --- STAGE 1: PRE-FILTER (Strict Database Query) ---
    if filters:
        logger.info(f"Attempting pre-filter retrieval with: {filters}")
        try:
            docs = retrieve_with_filter(
                vectordb=vectordb,
                query=query,
                filters=filters,
                k=k
            )
            if docs:
                logger.info(f"Successfully retrieved {len(docs)} docs with strict pre-filter.")
                return docs
        except Exception as e:
            logger.error(f"Pre-filter search error: {e}")

        logger.warning("No docs found with strict pre-filter. Falling back to semantic search...")

    # --- STAGE 2: SEMANTIC SEARCH ---
    # We retrieve k*3 candidates to allow the Post-Filter to promote relevant general docs
    retriever = get_multiquery_retriever(vectordb, llm, k * 3) if llm else get_base_retriever(vectordb, k * 3)
    docs = retriever.invoke(query)

    # --- STAGE 3: SMART POST-FILTER (Re-ranking) ---
    if filters and docs:
        scored_docs = []
        subject_keys = {"subcategory", "policy_name"}
        structural_keys = {"department", "category"}
        
        for doc in docs:
            score = 0.0
            doc_metadata = doc.metadata or {}
            
            # --- A. VERSION TIE-BREAKER ---
            # Kept very small (0.01) to avoid overriding subject matches
            v = doc_metadata.get("version", "1.0")
            try:
                score += (float(str(v).replace('v','').strip()) * 0.01)
            except:
                score += 0.001

            # --- B. METADATA MATCH SCORING ---
            for key, target in filters.items():
                actual = doc_metadata.get(key)
                if not actual: continue
                
                # Exact matches for Policy/Subcategory are weighted heavily
                if str(actual).lower() == str(target).lower():
                    if key in subject_keys:
                        score += 5.0
                    elif key in structural_keys:
                        score += 2.0

            # --- C. KEYWORD RELEVANCE BONUS (The Fix for 'I don't know') ---
            # If a document contains the specific keywords from the user's question,
            # we give it a massive boost. This allows a 'General Leave' document 
            # to be promoted if it contains 'carry forward' or 'notice period'.
            query_words = set(query.lower().split())
            content_lower = doc.page_content.lower()
            
            # Filter for meaningful words to avoid boosting on 'the', 'is', 'for'
            important_words = [w for w in query_words if len(w) > 3]
            match_count = sum(1 for w in important_words if w in content_lower)
            
            if match_count > 0:
                # Each matching keyword adds significant weight
                score += (match_count * 1.5)

            scored_docs.append((score, doc))
        
        # Sort by total score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Post-filtered matches found: {len(scored_docs)}. Best score: {scored_docs[0][0]}")
        return [d[1] for d in scored_docs[:k]]
    
    return docs[:k]