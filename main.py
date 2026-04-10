from src.embeddings import load_vector_store
from src.llm import get_llm
from src.prompt import get_prompt
from src.retrievers.hybrid_retriever import retrieve_documents
from src.memory import get_memory
from src.chat import chat

def main():
    print("🚀 HR Policy Chatbot Ready! (type 'exit' to quit)\n")

    # 1. Load components
    vectordb = load_vector_store()
    llm = get_llm()
    prompt = get_prompt()

    # 2. Create memory (single user for now)
    memory = get_memory(session_id="user_1")

    # --- PROACTIVE GREETING ---
    # Check if a department is already set in memory filters
    if not memory.get_filters().get("department"):
        welcome_msg = "🤖 Bot: Welcome! Before we begin, please let me know which **department** you belong to?"
        print(f"{welcome_msg}\n")
        # Record the bot's proactive question in memory so the LLM has context for the user's reply
        memory.add_ai_message(welcome_msg)

    # 3. Chat loop
    while True:
        question = input("👤 You: ")

        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            # The chat function now handles the logic:
            # - If user provides dept, it locks it and asks "How can I help?"
            # - If user asks a question, it uses the locked dept for RAG.
            answer = chat(
                memory=memory,
                llm=llm,
                prompt=prompt,
                retrieve_documents=retrieve_documents,
                vectordb=vectordb,
                question=question
            )

            print(f"\n🤖 Bot: {answer}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
