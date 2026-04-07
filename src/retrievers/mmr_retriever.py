from langchain_chroma import Chroma


# *************************** MMR RETRIEVER ***************************
def get_base_retriever(vectordb: Chroma, k: int = 5):
    """
    Base retriever using MMR (diversity-aware retrieval)
    """

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20  # fetch more → then diversify
        }
    )