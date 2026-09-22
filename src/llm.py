import os
import requests

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_TEMPERATURE,
    HF_MODEL,
)

load_dotenv(override=True)


def generate_answer_ollama(prompt):
    """Generate an answer using local Ollama."""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE
            }
        }
    )

    response.raise_for_status()

    return response.json()["response"]


def generate_answer_hf(prompt):
    """Generate an answer using Hugging Face."""

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
    """Choose which LLM provider to use."""

    if provider == "huggingface":
        return generate_answer_hf(prompt)

    return generate_answer_ollama(prompt)