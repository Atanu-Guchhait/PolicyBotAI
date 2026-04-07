from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class WindowMemory:
    """Use Short-term-memory to store converastion and also update filters"""
    def __init__(self, k=30):
        self.chat_history = InMemoryChatMessageHistory()
        self.k = k

        # structured memory (user profile)
        self.filters = {}

    def add_user_message(self, message: str):
        self.chat_history.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.chat_history.add_message(AIMessage(content=message))

    def get_messages(self):
        return self.chat_history.messages[-self.k:] if self.chat_history.messages else []

    # update filters
    def update_filters(self, new_filters: dict):
        #  Define terms that should NEVER overwrite a specific, known trait
        generic_terms = {"all employees", "all", "any", "unknown", "n/a", "general", "none", "not specified"}

        for key, value in new_filters.items():
            if value:
                val_clean = str(value).lower().strip()
                
                #  Dynamic check: Does this key already have a specific, non-generic value?
                existing_value = self.filters.get(key)
                
                if existing_value and existing_value not in generic_terms:
                    # If we already know the user is 'admin', don't let 'all' overwrite it
                    if val_clean in generic_terms:
                        continue  
                
                # Otherwise, update or set the new value
                self.filters[key] = val_clean

    # get user profile
    def get_filters(self):
        return self.filters

    # reset filters
    def clear_filters(self):
        self.filters = {}


#  multi-user support
memory_store = {}

def get_memory(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = WindowMemory(k=30)
    return memory_store[session_id]