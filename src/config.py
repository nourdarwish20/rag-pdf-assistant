EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

TOP_K = 3

# Local LLM
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Hosted LLM
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"