---
title: RAG PDF Assistant
emoji: 📄
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# RAG PDF Assistant

[Live Demo on Hugging Face Spaces](ADD_LINK_AFTER_DEPLOYMENT)

Ask questions about your own PDF files and get answers taken from the
document itself, with the page numbers they came from.

You upload one or more PDFs. The text is split into small chunks, each
chunk is turned into an embedding, and the embeddings are stored in a
FAISS index. When you ask a question, FAISS finds the most relevant
chunks and a Hugging Face model writes the answer using only those
chunks. The answer is shown with its source pages.

The PDF is always the primary source. If the document genuinely does
not contain the answer, the app can fall back to a simple, fast web
search with Exa, and that answer is clearly labelled as coming from the
web.

## What this project demonstrates

- An end-to-end RAG pipeline wired from scratch
- PDF ingestion and chunking
- Sentence-transformer embeddings
- FAISS vector retrieval
- Hugging Face LLM integration
- A Gradio user interface
- API access to the same pipeline
- A simple web-search fallback with Exa

## Project structure

```
rag-pdf-assistant/
├─ app.py               Gradio interface and the API endpoints
├─ requirements.txt     Pinned dependencies
├─ README.md            This file
├─ .env.example         Template for your local .env
└─ src/
   ├─ config.py         All settings, read from environment variables
   ├─ ingest.py         PDF loading and text splitting
   ├─ vector_store.py   Embeddings and the FAISS store
   ├─ llm.py            Hugging Face and Ollama calls
   ├─ rag.py            Retrieval, prompt building, answering
   └─ web_search.py     Exa web search fallback
```

## How the RAG pipeline works

```
PDF
 → text loading
 → chunks (700 characters, 100 overlap)
 → all-MiniLM-L6-v2 embeddings
 → FAISS index
 → top 3 relevant chunks
 → prompt
 → Hugging Face LLM
 → answer + PDF sources
```

If the PDF cannot answer the question:

```
 → Exa fast web search
 → answer clearly labelled as coming from the web
```

## Main technologies

| Technology | Role |
| --- | --- |
| Python | Application language |
| Gradio | Web interface and API |
| LangChain | PDF loading, splitting, retriever |
| FAISS | Vector similarity search |
| sentence-transformers | Embedding model backend |
| `all-MiniLM-L6-v2` | The embedding model |
| Hugging Face Inference | Runs `meta-llama/Llama-3.1-8B-Instruct` |
| Exa AI | Fast web search fallback |

## Key features

- Upload one or more PDF files
- Semantic retrieval over the document, not keyword matching
- Answers cite their source file and page number
- The same PDF is never indexed twice in one session
- Each visitor gets a private, in-memory FAISS store
- Simple follow-up handling, such as "explain more"
- `/upload` and `/ask` API endpoints
- Optional Exa web search, used only when the PDF cannot answer
- The PDF is always tried before the web

Follow-up handling uses only the most recent turn of the conversation
so that short messages like "explain more" are understood. This is not
full conversational memory: nothing is stored between requests.

## Installation

```bash
git clone https://github.com/nourdarwish20/rag-pdf-assistant.git
cd rag-pdf-assistant

python -m venv .venv
```

Activate the environment.

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS and Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and fill in your own values:

```bash
cp .env.example .env
```

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `HF_TOKEN` | Yes | – | Hugging Face access token |
| `HF_MODEL` | No | `meta-llama/Llama-3.1-8B-Instruct` | Model that writes answers |
| `LLM_PROVIDER` | No | `huggingface` | Set to `ollama` for local testing |
| `EXA_API_KEY` | No | – | Enables the web search fallback |
| `EXA_SEARCH_TYPE` | No | `fast` | Exa search mode |
| `EXA_NUM_RESULTS` | No | `3` | Web results per search |

`HF_TOKEN` and `EXA_API_KEY` are secrets. Never commit your `.env`
file; it is already listed in `.gitignore`. On Hugging Face Spaces, add
both under **Settings → Secrets**.

Without `EXA_API_KEY` the app still runs. Questions the PDF cannot
answer simply return "I don't know based on the provided document."

## Run locally

```bash
python app.py
```

The app normally opens at `http://127.0.0.1:7860`.

## API

Both endpoints are reachable with `gradio_client`. They share one
session, so upload a PDF before asking about it.

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")

client.predict(files=[handle_file("document.pdf")], api_name="/upload")

result = client.predict(question="What is the password policy?", api_name="/ask")
print(result["answer"])
```

`/upload` takes a list of PDF paths and returns a status string.

`/ask` takes a question and returns JSON:

```json
{
  "answer": "Company passwords must never be shared with other people.",
  "sources": ["📄 sample.pdf · Page 2", "📄 sample.pdf · Page 1"],
  "from_web": false,
  "error": null
}
```

`from_web` is `true` when the answer came from Exa instead of the PDF.
`error` holds a message when the question is empty or no PDF has been
uploaded in that session.

## Session and privacy

Each visitor gets their own FAISS store, held in memory only. Visitors
never see each other's documents, and nothing is written to disk.

This means an uploaded PDF disappears when the session ends or when the
Space restarts. That is intentional for a public prototype: it keeps
one visitor's documents private from the next.

## Performance

Local testing was successful, with PDF-backed answers returning in a
few seconds and the Exa fallback taking longer because it adds a web
search and a second model call.

**Live Hugging Face API latency: TO BE MEASURED AFTER DEPLOYMENT**

## Limitations

- Uploaded documents are temporary and are lost on restart
- Only the last turn is used for follow-up questions, not full memory
- The Exa fallback is slower than a normal PDF answer, because it adds
  a web search step on top of the usual pipeline

## Purpose

This was built as a standalone RAG prototype, to practise wiring an
end-to-end retrieval pipeline and deploying it on Hugging Face Spaces.
