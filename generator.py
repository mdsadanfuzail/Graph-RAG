from config import llm

def generate_answer(question: str, context: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    return llm.invoke(prompt).content
