# test_model_compare.py
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

candidates = ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "openai/gpt-oss-120b"]

print("--- Testing Available Models ---")
for model in candidates:
    try:
        llm = ChatGroq(model=model, temperature=0.2, streaming=False)
        start = time.time()
        resp = llm.invoke("Say hello in exactly 5 words.")
        print(f"{model}: OK ({time.time()-start:.2f}s) -> {resp.content}")
    except Exception as e:
        print(f"{model}: FAILED -> {e}")
