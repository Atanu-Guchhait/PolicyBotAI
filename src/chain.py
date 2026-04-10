from typing import Callable, Dict, Any, List
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from utils.logger import setup_logger

logger = setup_logger(__name__)

# **************************** FORMAT DOCS ****************************
def format_docs(docs: List, max_docs: int = 5) -> str:
    if not docs:
        return "No relevant policy documents found."
    
    formatted = []
    for i, doc in enumerate(docs[:max_docs]):
        # Extract metadata - using .get() with defaults is good practice
        policy_name = doc.metadata.get('policy_name', 'Unknown Policy')
        version = doc.metadata.get('version', 'Unknown')
        year = doc.metadata.get('year', 'Unknown')
        cycle = doc.metadata.get('review_cycle', 'As Needed')
        
        # --- IMPROVEMENT: Clearer Header for the LLM Auditor ---
        # We wrap the metadata in a clear 'METADATA' tag so the LLM knows 
        # these are the facts it must use for the Temporal Audit.
        header = (f"--- DOCUMENT {i+1} METADATA ---\n"
                  f"Policy: {policy_name}\n"
                  f"Version: {version}\n"
                  f"Last Updated Year: {year}\n"
                  f"Review Cycle: {cycle}\n"
                  f"--- CONTENT ---")
        
        formatted.append(f"{header}\n{doc.page_content}")
        
    return "\n\n".join(formatted)

# **************************** BUILD CHAIN ****************************
def build_chain(retriever_fn: Callable, llm, vectordb, prompt):
    
    # Retrieve + prepare inputs
    def prepare_inputs(inputs: Dict[str, Any]):
        question = inputs["question"]
        standalone_question = inputs["standalone_question"]
        chat_history = inputs.get("chat_history", [])
        active_filters = inputs.get("active_filters", {})
        current_date = inputs.get("current_date", "Unknown Date")
        # ADDED: current_year for the prompt audit logic
        current_year = inputs.get("current_year", 2026) 

        # Build the user profile string for the prompt
        if active_filters:
            profile_text = "\n".join([f"- {k.capitalize()}: {v}" for k, v in active_filters.items()])
        else:
            profile_text = "No profile details known yet."

        # Execute the Hybrid Retriever
        docs = retriever_fn(
            vectordb=vectordb,
            query=standalone_question,
            llm=llm,
            filters=active_filters
        )

        logger.info(f"LCEL Chain Retrieved {len(docs)} documents")
        context = format_docs(docs)

        # Return the dictionary expected by your prompt.py
        return {
            "context": context,
            "question": question, 
            "chat_history": chat_history,
            "user_profile": profile_text,
            "current_date": current_date,
            "current_year": current_year 
        }

    # The LCEL Pipeline
    chain = (
        RunnableLambda(prepare_inputs)
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain