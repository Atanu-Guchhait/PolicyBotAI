import json
import re
import logging
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage

# Setup logger for production monitoring
logger = logging.getLogger(__name__)

# **************************** SAFE JSON PARSER ****************************
def safe_json_parse(text: str) -> Dict[str, Any]:
    """Extracts and parses JSON from LLM responses, handling markdown blocks."""
    try:
        # Remove potential markdown code blocks
        clean_text = re.sub(r"```json|```", "", text).strip()
        if not clean_text:
            return {}
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}. Raw text: {text}")
        return {}

# **************************** NORMALIZE FILTERS ****************************
def normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans and standardizes filter values for ChromaDB compatibility."""
    return {
        str(k).lower().strip(): str(v).lower().strip()
        for k, v in filters.items()
        if v not in [None, "", "null", "none", "unknown"]
    }

# **************************** CONTEXTUAL QUERY ****************************
def build_contextual_query(llm, chat_history: List[BaseMessage], question: str) -> str:
    """Rewrites a dependent question into a standalone query using conversation history."""
    prompt = f"""
Given the following conversation and a follow-up question, rephrase the follow-up 
into a standalone question that includes all necessary context (like department, 
role, or specific policy mentioned previously).

Chat History:
{chat_history}

User Input: 
{question}

Standalone Question:
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error in build_contextual_query: {e}")
        return question

# **************************** SMART REWRITE CHECK ****************************
def should_rewrite(chat_history: List[BaseMessage]) -> bool:
    """Rewrite if there is a history to ensure follow-up context is captured."""
    return len(chat_history) > 0

# **************************** FILTER EXTRACTION ****************************
# **************************** FILTER EXTRACTION ****************************
def extract_filters(llm, chat_history: List[BaseMessage], question: str) -> Dict[str, Any]:
    """Extracts metadata filters using both current question and previous context."""
    full_input = f"Chat History: {chat_history}\nLatest Input: {question}"

    prompt = f"""
You are an HR Metadata Extractor. Based on the conversation history and latest input, 
identify the applicable metadata filters. 

Available fields:
- category (e.g., leave, recruitment, compliance)
- department (e.g., admin, finance, engineering)
- applicable_to (e.g., male, female, contract employee, all employees)
- role (e.g., senior, junior, manager, staff)

Rules:
1. ONLY return JSON.
2. If a department, category, or role was mentioned earlier in the history and still applies, include it.
3. Use generic values for 'applicable_to' (e.g., use 'male' if they say 'I am a man').
4. If the user specifies a seniority or job level, assign it to the 'role' field.

Input:
{full_input}
"""
    try:
        response = llm.invoke(prompt)
        filters = safe_json_parse(response.content)
        return normalize_filters(filters)
    except Exception as e:
        logger.error(f"Error in extract_filters: {e}")
        return {}

# **************************** FORMAT DOCS ****************************
def format_docs(docs: List, max_docs: int = 5) -> str:
    """Joins document contents into a single string for LLM context."""
    if not docs:
        return "No relevant policy documents found."
    return "\n\n".join(doc.page_content for doc in docs[:max_docs])

# **************************** MAIN CHAT FUNCTION ****************************
def chat(memory, llm, prompt, retrieve_documents, vectordb, question: str) -> str:
    """
    Main orchestration function for the HR Bot.
    Handles memory, query rewriting, metadata filtering, and RAG.
    """
    try:
        # 1. Get Conversation History
        chat_history = memory.get_messages()

        # 2. Standalone Query Generation (Contextualize)
        if should_rewrite(chat_history):
            standalone_question = build_contextual_query(llm, chat_history, question)
        else:
            standalone_question = question

        # 3. Extract and Persist Filters
        # We extract from history + standalone question to ensure we don't lose the 'Admin' department
        new_filters = extract_filters(llm, chat_history, standalone_question)
        memory.update_filters(new_filters)
        
        # 4. Final Filter Assembly
        active_filters = memory.get_filters()

        # 5. Retrieve Documents (Hybrid Search)
        # Note: Your retrieve_documents should handle the fallback if filters return 0 results
        docs = retrieve_documents(
            vectordb=vectordb,
            query=standalone_question,
            llm=llm,
            filters=active_filters
        )

        context = format_docs(docs)

        # 6. Generate Response
        # Format the active filters into a readable string for the LLM
        if active_filters:
            profile_text = "\n".join([f"- {k.capitalize()}: {v.title()}" for k, v in active_filters.items()])
        else:
            profile_text = "No profile details known yet."

        chain_input = {
            "chat_history": chat_history,
            "context": context,
            "question": question,
            "user_profile": profile_text  # <-- This is the new integration
        }
        
        response = llm.invoke(prompt.format_messages(**chain_input))
        answer = response.content

        # 7. Update Memory
        memory.add_user_message(question)
        memory.add_ai_message(answer)

        return answer

    except Exception as e:
        logger.error(f"Critical error in chat flow: {e}", exc_info=True)
        return "I encountered an internal error. Please try rephrasing your question."