from typing import Callable, Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from src.prompt import get_prompt
from utils.logger import setup_logger

logger = setup_logger(__name__)


#  Format retrieved documents
def format_docs(docs, max_docs=5):
    docs = docs[:max_docs]
    return "\n\n".join(doc.page_content for doc in docs)


#  Build chain
def build_chain(retriever_fn: Callable, llm):
    prompt = get_prompt()

    # 🔹 Step 1: Retrieve + prepare inputs
    def prepare_inputs(inputs: Dict[str, Any]):
        query = inputs["question"]
        chat_history = inputs.get("chat_history", [])
        filters = inputs.get("filters", None)

        docs = retriever_fn(
            query=query,
            filters=filters
        )

        logger.info(f"Retrieved {len(docs)} documents")

        #  Handle empty retrieval
        if not docs:
            context = "No relevant policy found."
        else:
            context = format_docs(docs)

        return {
            "context": context,
            "question": query,
            "chat_history": chat_history
        }

    chain = (
        RunnableLambda(prepare_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain