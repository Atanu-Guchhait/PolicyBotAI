from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.language_models.base import BaseLanguageModel
from langchain_chroma import Chroma


# *************************** MULTI QUERY ***************************
def get_multiquery_retriever(
    vectordb: Chroma,
    llm: BaseLanguageModel,
    k: int = 5
):
    """
    MultiQueryRetriever generates multiple query variations
    """

    base_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20
        }
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )