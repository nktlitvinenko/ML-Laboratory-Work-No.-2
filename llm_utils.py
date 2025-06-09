import requests
from config import Config

def build_prompt(context: list[str], question: str) -> str:
    return f"""You are an assistant. Use the context below to answer the question.

        Please do not rely on your own knowledge or make assumptions beyond the context.
        If the answer is not explicitly available, say: "I'm sorry, I couldn't find enough information to answer that."

        Context:
        {chr(10).join(context)}

        Question: {question}
        Answer:"""

def ask_llm(prompt: str) -> str:
    response = requests.post(Config.get("LLM_URL"), json={
        "model": Config.get("LLM_MODEL_NAME"),
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]
