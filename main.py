from src.embeddings import load_vector_store
from src.llm import get_llm
from src.prompt import get_prompt
from src.retrievers.hybrid_retriever import retrieve_documents
from src.memory import get_memory
from src.chat import chat


def main():
    print("🚀 HR Policy Chatbot Ready! (type 'exit' to quit)\n")

    #  Load components
    vectordb = load_vector_store()
    llm = get_llm()
    prompt = get_prompt()

    #  Create memory (single user for now)
    memory = get_memory(session_id="user_1")

    #  Chat loop
    while True:
        question = input("👤 You: ")

        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            answer = chat(
                memory=memory,
                llm=llm,
                prompt=prompt,
                retrieve_documents=retrieve_documents,
                vectordb=vectordb,
                question=question,
        
            )

            print(f"\n🤖 Bot: {answer}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main()