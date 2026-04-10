import logging
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import BaseMessage
from src.pydantic_schema.schema import HRMetadata, extract_metadata_from_query
from src.chain import build_chain 
from utils.logger import setup_logger

logger = setup_logger(__name__)

def build_contextual_query(llm, chat_history: List[BaseMessage], question: str) -> str:
    prompt = f"Rephrase this follow-up into a standalone question based on history: {chat_history[-5:]}\nQuestion: {question}"
    try:
        return llm.invoke(prompt).content.strip()
    except:
        return question

def chat(memory, llm, prompt, retrieve_documents, vectordb, question: str) -> str:
    try:
        chat_history = memory.get_messages()
        active_filters = memory.get_filters()
        verified_dept = active_filters.get("department")
        now = datetime.now()
        current_date, current_year = now.strftime("%B %Y"), now.year 

        #  EXPANDED TRIGGERS: Added words that imply a general rule search
        broad_search_triggers = [
            "carry forward", "rules", "apply", "limit", 
            "process", "procedure", "notice", "advance", "period"
        ]
        needs_broader_context = any(t in question.lower() for t in broad_search_triggers)

        #  METADATA EXTRACTION
        new_metadata = extract_metadata_from_query(llm, question)
        requested_dept = new_metadata.get("department")

        # --- TOPIC GUARD REFINEMENT ---
        # REMOVED "policy" from triggers to prevent locking when changing categories
        context_triggers = ["it", "this", "that", "forward", "carry", "apply", "rules", "limit", "notice", "period"]
        previous_policy = active_filters.get("policy_name")
        
        # We only lock if there is NO new policy detected by the metadata extractor
        is_topic_shift = "policy_name" in new_metadata or "category" in new_metadata
        is_follow_up = previous_policy and any(word in question.lower() for word in context_triggers) and not is_topic_shift

        if is_follow_up:
            new_metadata.update({
                "policy_name": previous_policy,
                "subcategory": active_filters.get("subcategory"),
                "category": active_filters.get("category")
            })
            standalone_question = question 
            logger.info(f"Topic Guard Active: Locked to {previous_policy}")
        else:
            standalone_question = build_contextual_query(llm, chat_history, question) if chat_history else question

        # --- DEPARTMENT LOCK LOGIC ---
        if not verified_dept:
            if requested_dept:
                memory.update_filters({"department": requested_dept})
                return f"Successfully logged into **{requested_dept.upper()}**. How can I help?"
            return "Please let me know your **department** to proceed."

        # BLOCK department shifts unless explicitly requested
        # Find this block in src/chat.py and update it:
        if requested_dept and requested_dept.lower() != verified_dept.lower():
            # Only block if the user is explicitly asking for "IT department" or "Finance department"
            if any(kw in question.lower() for kw in ["department", "dept", "team"]):
                return f"Access Denied. Your session is locked to the **{verified_dept.upper()}** department."
            else:
                # If it's a general question, ignore the 'requested_dept' and force the search 
                # to stay within the user's verified department.
                new_metadata["department"] = verified_dept

        #  FILTER STRATEGY
        if new_metadata: 
            memory.update_filters(new_metadata)
        
        search_filters = memory.get_filters().copy()
        
        # RELAX FILTERS: If asking about general rules (carry forward/notice), 
        # pop the specific policy so the retriever finds the "General Policy" doc.
        if needs_broader_context:
            search_filters.pop("policy_name", None)
            search_filters.pop("subcategory", None)
            logger.info("Broadening search scope for general rules.")

        # 4. RUN RAG CHAIN
        chain = build_chain(retriever_fn=retrieve_documents, llm=llm, vectordb=vectordb, prompt=prompt)
        
        return chain.invoke({
            "question": question,
            "standalone_question": standalone_question,
            "chat_history": chat_history[-10:],
            "active_filters": search_filters,
            "current_date": current_date,
            "current_year": current_year 
        })
        
    except Exception as e:
        logger.error(f"Chat Error: {e}", exc_info=True)
        return "I encountered an error. Please try again."