import os
from pathlib import Path
from dotenv import load_dotenv


# Load values from .env
load_dotenv(override=True)


# -------------------------
# Project Paths
# -------------------------

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
VECTOR_DB_DIR = DATA_DIR / "vector_db"


# Create folders automatically
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Embedding Model
# -------------------------

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DEVICE = "cpu"


# -------------------------
# Text Splitting
# -------------------------

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# -------------------------
# Retrieval
# -------------------------

RETRIEVER_K = 3


# -------------------------
# Ollama - Local LLM
# -------------------------

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

OLLAMA_TEMPERATURE = float(
    os.getenv("OLLAMA_TEMPERATURE", "0.1")
)


# -------------------------
# Hugging Face - Hosted LLM
# -------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = os.getenv(
    "HF_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct"
)


# -------------------------
# LLM Provider
# -------------------------

# "huggingface" is the deployed default; set
# LLM_PROVIDER=ollama in .env for local testing.
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "huggingface"
)
