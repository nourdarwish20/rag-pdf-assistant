# 📄 RAG PDF Assistant

### 🚀 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/no2004/rag-pdf-assistant)

This repository contains a **Retrieval Augmented Generation (RAG)** system built in **Python** that lets you ask questions about your own PDF files.
It uses **FAISS** for vector search, **Hugging Face Inference** (`meta-llama/Llama-3.1-8B-Instruct`) for generation, and a **Gradio** web interface.

Answers are taken from the uploaded document and shown with the page they came from. The PDF is always the primary source; when the document genuinely cannot answer a question, **Exa AI** performs a fast web search and that answer is clearly labelled as coming from the web.

It is designed as a **learning project** for understanding:
- RAG pipelines (Ingestion → Embedding → Retrieval → Generation)
- Sentence-transformer embeddings
- Vector databases (FAISS)
- LLM integration (Hugging Face Inference)
- Building interactive AI UIs with Gradio
- Deploying a working app to Hugging Face Spaces

---

## 📂 Project Structure

```
.
├─ README.md
├─ requirements.txt
├─ .env.example
├─ app.py                 # Gradio UI + API endpoints
├─ .github/workflows/
│ └─ sync-to-hf.yml       # Auto-deploy to Hugging Face
└─ src/
  ├─ __init__.py
  ├─ config.py            # Settings (models, chunking, keys)
  ├─ ingest.py            # PDF loader & text splitter
  ├─ vector_store.py      # FAISS store & embeddings
  ├─ llm.py               # Hugging Face connector
  ├─ rag.py               # Retrieval, prompt, answering
  └─ web_search.py        # Exa web search fallback
```

---

## ⚙️ Requirements

- Python 3.10+
- A [Hugging Face token](https://huggingface.co/settings/tokens) with inference access
- An [Exa API key](https://exa.ai) — optional, only for the web fallback

Install dependencies:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> Without an Exa key the app still runs. Questions the PDF cannot answer simply return *"I don't know based on the provided document."*

---

## 🚀 Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/nourdarwish20/rag-pdf-assistant.git
cd rag-pdf-assistant
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure your environment

Copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Required | Default |
| --- | --- | --- |
| `HF_TOKEN` | Yes | – |
| `HF_MODEL` | No | `meta-llama/Llama-3.1-8B-Instruct` |
| `LLM_PROVIDER` | No | `huggingface` |
| `EXA_API_KEY` | No | – |
| `EXA_SEARCH_TYPE` | No | `fast` |
| `EXA_NUM_RESULTS` | No | `3` |

> [!IMPORTANT]
> `HF_TOKEN` and `EXA_API_KEY` are secrets. Never commit `.env`. On Hugging Face Spaces, add them under **Settings → Secrets**.

### 5️⃣ Run the app

```bash
python app.py
```

This opens a local URL (e.g. `http://127.0.0.1:7860`).

---

## 🐳 Run with Docker

Build the image:

```bash
docker build -t rag-pdf-assistant .
```

Run the container, passing your secrets from `.env` at runtime:

```bash
docker run -p 7860:7860 --env-file .env rag-pdf-assistant
```

Then open `http://localhost:7860`.

> [!NOTE]
> The image installs a CPU-only build of PyTorch to keep it small. `.env` is excluded from the image, so your keys are never baked in.

---

## 🛠️ Implementation Notes

### 🧩 Components

- **LLM**: Hugging Face Inference, `meta-llama/Llama-3.1-8B-Instruct`.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, running on CPU.
- **Vector Store**: FAISS, held in memory per session.
- **Orchestration**: LangChain (loaders, splitter, retriever).
- **UI & API**: Gradio.
- **Web Fallback**: Exa AI, fast search mode.

The flow is: a PDF is split into 700-character chunks, each chunk is embedded and indexed in FAISS, and a question retrieves the 3 closest chunks. Those chunks become the context for the model, which is instructed to answer from them alone. If the model reports the answer is not in the document, Exa searches the web instead.

### ✨ Key Features

- **PDF-First Answering**: The document is always searched before the web, so normal questions stay fast and grounded.
- **Source Citations**: Every answer lists the file and page numbers it came from.
- **Private Session Store**: Each visitor gets their own in-memory FAISS index. Visitors never see each other's documents.
- **Smart Deduplication**: Uploading the same PDF twice in a session does not create duplicate chunks.
- **Follow-Up Handling**: Short messages like *"explain more"* or *"can I put lemon in it?"* are understood using the last turn of the conversation. This is not full conversational memory — nothing is stored between requests.
- **Labelled Web Answers**: Exa-backed answers are marked *"Answered from the web"* and cite their links, so they are never mistaken for document content.

---

## 🔌 API

The Gradio app exposes two endpoints. They share one session, so upload before asking.

```python
from gradio_client import Client, handle_file

client = Client("no2004/rag-pdf-assistant")

client.predict([handle_file("document.pdf")], api_name="/upload")

result = client.predict("What is the password policy?", api_name="/ask")
print(result["answer"])
```

`/ask` returns `answer`, `sources`, `from_web` and `error`.

---

## ⚡ Performance

A measured test against the deployed Space returned a PDF-backed answer from `/ask` in **2.25 seconds**.

This is one test result on the live deployment, not a universal benchmark. Web-fallback questions take longer, since they add a search step and a second model call.

---

## 🌐 Deployment

The app runs publicly on [Hugging Face Spaces](https://huggingface.co/spaces/no2004/rag-pdf-assistant).

Deployment is automatic: a GitHub Actions workflow (`.github/workflows/sync-to-hf.yml`) syncs every push to `main` across to the Space, so the live demo always matches this repository.

---

## ⚠️ Limitations

- Uploaded PDFs are temporary and are lost when the session ends or the Space restarts.
- Follow-up questions use only the last turn, not full conversation history.
- The Exa fallback is slower than a normal PDF answer.
