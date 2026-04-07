from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024,
        top_p=0.9,
        timeout=30,
        max_retries=2
    )