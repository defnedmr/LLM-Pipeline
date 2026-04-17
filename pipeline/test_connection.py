"""
Connection test — verifies Groq and Together.ai API keys work.
Run this before starting the full pipeline.
"""

import os
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def test_groq():
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Türkçe bir cümle yaz."}],
        temperature=0,
    )
    print("[GROQ OK]", response.choices[0].message.content[:80])


def test_together():
    client = OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B",
        messages=[{"role": "user", "content": "Hello, respond in one sentence."}],
        temperature=0,
        max_tokens=64,
    )
    print("[TOGETHER OK]", response.choices[0].message.content[:80])


if __name__ == "__main__":
    test_groq()
    test_together()
