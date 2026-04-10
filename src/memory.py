from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class WindowMemory:
    """Stores conversation history and manages the persistent User Profile (filters)."""
    def __init__(self, k=20):
        self.chat_history = InMemoryChatMessageHistory()
        self.k = k
        # Structured memory (Persistent Session Metadata)
        self.filters = {}

    def add_user_message(self, message: str):
        self.chat_history.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.chat_history.add_message(AIMessage(content=message))

    def get_messages(self):
        """Returns the last k messages."""
        return self.chat_history.messages[-self.k:] if self.chat_history.messages else []

    def update_filters(self, new_filters: dict):
        """
        Merges new findings into existing profile. 
        Protects specific values from being overwritten by generic ones.
        """
        generic_terms = {"all employees", "all", "any", "unknown", "n/a", "general", "none", "not specified"}

        for key, value in new_filters.items():
            if value is not None:
                val_clean = str(value).lower().strip()
                
                # --- SPECIAL HANDLING FOR BOOLEANS ---
                if key == "is_mandatory":
                    self.filters[key] = val_clean == "true"
                    continue

                # --- SPECIAL HANDLING FOR DEPARTMENT (SECURITY) ---
                existing_value = self.filters.get(key)
                
                if key == "department":
                    # If we already have a specific department, don't let 'all' overwrite the security lock
                    if existing_value and existing_value not in generic_terms:
                        if val_clean in generic_terms:
                            continue 
                
                # --- GENERAL OVERWRITE PROTECTION ---
                # Don't let a generic term (like 'all') overwrite a specific term (like 'male')
                if existing_value and existing_value not in generic_terms:
                    if val_clean in generic_terms:
                        continue  
                
                self.filters[key] = val_clean

    def get_filters(self):
        """Retrieves the full profile of metadata filters."""
        return self.filters

    def clear_filters(self):
        """Wipes the current user profile."""
        self.filters = {}


# --- Session Store Management ---
memory_store = {}

def get_memory(session_id: str):
    """Retrieves existing memory object or initializes a new one for a user session."""
    if session_id not in memory_store:
        memory_store[session_id] = WindowMemory(k=20)
    return memory_store[session_id]