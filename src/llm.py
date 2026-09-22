import os
import requests

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    HF_MODEL
)

load_dotenv(override=True)

def generate_answer_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]


def generate_answer_hf(prompt):
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN was not found.")

    client = InferenceClient(
        provider="auto",
        api_key=token
    )

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.1
    )

    return response.choices[0].message.content


def generate_answer(prompt, provider="ollama"):
    if provider == "huggingface":
        return generate_answer_hf(prompt)

    return generate_answer_ollama(prompt)