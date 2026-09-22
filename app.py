import os
import gradio as gr

from src.ingest import ingest_file
from src.vector_store import update_session_store
from src.rag import answer_question


# ------------------------------------------------
# PDF Upload
# ------------------------------------------------

def process_upload(files, session_store):
    """Save uploaded PDFs into this session's own store."""

    if not files:
        return "Please select at least one PDF.", session_store

    if not isinstance(files, list):
        files = [files]

    all_documents = []
    status_messages = []

    try:
        for file in files:

            file_path = (
                file
                if isinstance(file, str)
                else file.name
            )

            filename = os.path.basename(file_path)

            # PDF -> LangChain chunks
            documents = ingest_file(file_path)

            all_documents.extend(documents)

            status_messages.append(
                f"✓ {filename}: {len(documents)} chunks"
            )

        # Add documents to this session's FAISS store
        session_store = update_session_store(
            session_store,
            all_documents
        )

        status_messages.append(
            f"\n✓ Knowledge base ready"
            f"\n{session_store.index.ntotal} vectors stored"
        )

        return "\n".join(status_messages), session_store

    except Exception as e:
        return f"Error: {str(e)}", session_store


# ------------------------------------------------
# Chat
# ------------------------------------------------

def format_sources(documents):
    """Turn retrieved documents into readable source lines."""

    source_lines = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get("page")

        source_name = os.path.basename(source)

        if page is not None:
            source_line = (
                f"📄 {source_name} · "
                f"Page {page + 1}"
            )
        else:
            source_line = f"📄 {source_name}"

        if source_line not in source_lines:
            source_lines.append(source_line)

    return source_lines


def ask(question, session_store):
    """Answer one question. This is the API endpoint."""

    if not question or not question.strip():
        return {
            "answer": None,
            "sources": [],
            "error": "Question is empty."
        }

    if session_store is None:
        return {
            "answer": None,
            "sources": [],
            "error": "No PDF has been uploaded in this session yet."
        }

    try:
        result = answer_question(session_store, question)

        return {
            "answer": result["answer"],
            "sources": format_sources(result["sources"]),
            "error": None
        }

    except Exception as e:
        return {
            "answer": None,
            "sources": [],
            "error": str(e)
        }


def chat(message, history, session_store):
    """Run the RAG pipeline and update chat history."""

    if not message.strip():
        return history, ""

    result = ask(message, session_store)

    if result["error"]:
        assistant_message = result["error"]

    elif result["sources"]:
        sources = "\n".join(result["sources"])
        assistant_message = (
            f"{result['answer']}\n\n"
            f"**Sources**\n{sources}"
        )

    else:
        assistant_message = result["answer"]

    history.append(
        {
            "role": "user",
            "content": message
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": assistant_message
        }
    )

    return history, ""


def clear_chat():
    return []


# ------------------------------------------------
# Custom Style
# ------------------------------------------------

custom_css = """

:root {
    --background: #F7F4EE;
    --panel: #FFFFFF;
    --accent: #66756B;
    --accent-hover: #55645B;
    --text: #2E332F;
    --muted: #777B76;
    --border: #E4DED3;
}


/* Whole page */

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    background: var(--background) !important;
    color: var(--text) !important;
}


/* Header */

#header {
    text-align: center;
    padding: 25px 10px 20px 10px;
}

#header h1 {
    color: var(--text);
    margin-bottom: 5px;
}

#header p {
    color: var(--muted);
    font-size: 16px;
}


/* Cards */

.panel-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
}


/* Buttons */

.primary-btn {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
}

.primary-btn:hover {
    background: var(--accent-hover) !important;
}


/* Chat */

#chatbot {
    border-radius: 18px !important;
    border: 1px solid var(--border) !important;
    background: white !important;
}


/* Text box */

#message-box textarea {
    border-radius: 14px !important;
}


/* Footer */

#footer {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding-top: 20px;
}
"""


# ------------------------------------------------
# Gradio UI
# ------------------------------------------------

with gr.Blocks(
    title="RAG PDF Assistant"
) as demo:

    # One private, in-memory vector store per visitor.
    # Nothing is shared between sessions.
    session_store = gr.State(None)

    # Header

    gr.Markdown(
        """
        # RAG PDF Assistant
        Ask questions and get answers directly from your documents.
        """,
        elem_id="header"
    )

    # Main Layout

    with gr.Row():

        # --------------------------------
        # Left Side - Knowledge Base
        # --------------------------------

        with gr.Column(
            scale=1,
            min_width=280,
            elem_classes="panel-card"
        ):

            gr.Markdown(
                """
                ### Knowledge Base

                Upload your PDF files and prepare
                them for the assistant.
                """
            )

            pdf_files = gr.File(
                label="PDF Documents",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath"
            )

            process_button = gr.Button(
                "Process Documents",
                elem_classes="primary-btn"
            )

            status = gr.Textbox(
                label="Status",
                lines=6,
                interactive=False,
                placeholder=(
                    "Your document status "
                    "will appear here..."
                )
            )

        # --------------------------------
        # Right Side - Chat
        # --------------------------------

        with gr.Column(
            scale=3,
            elem_classes="panel-card"
        ):

            gr.Markdown(
                "### Chat with your documents"
            )

            chatbot = gr.Chatbot(
                value=[],
                height=500,
                elem_id="chatbot",
                placeholder="Upload a PDF, then ask anything about it.",
                layout="panel"
            )

            with gr.Row():

                message = gr.Textbox(
                    placeholder=(
                        "Ask something about "
                        "your documents..."
                    ),
                    show_label=False,
                    lines=1,
                    scale=8,
                    elem_id="message-box"
                )

                send_button = gr.Button(
                    "Send",
                    scale=1,
                    elem_classes="primary-btn"
                )

            clear_button = gr.Button(
                "Clear chat",
                size="sm"
            )

    # Events

    process_button.click(
        fn=process_upload,
        inputs=[
            pdf_files,
            session_store
        ],
        outputs=[
            status,
            session_store
        ],
        api_name="upload"
    )

    send_button.click(
        fn=chat,
        inputs=[
            message,
            chatbot,
            session_store
        ],
        outputs=[
            chatbot,
            message
        ],
        api_name=False
    )

    message.submit(
        fn=chat,
        inputs=[
            message,
            chatbot,
            session_store
        ],
        outputs=[
            chatbot,
            message
        ],
        api_name=False
    )

    clear_button.click(
        fn=clear_chat,
        outputs=chatbot,
        api_name=False
    )

    # ----------------------------------------
    # API endpoint: /ask
    # Hidden from the UI, callable from code.
    # ----------------------------------------

    api_question = gr.Textbox(visible=False)
    api_answer = gr.JSON(visible=False)
    api_trigger = gr.Button(visible=False)

    api_trigger.click(
        fn=ask,
        inputs=[
            api_question,
            session_store
        ],
        outputs=api_answer,
        api_name="ask"
    )

    # Footer

    gr.Markdown(
        """
        Built with LangChain · FAISS ·
        Hugging Face · Gradio
        """,
        elem_id="footer"
    )

if __name__ == "__main__":
    demo.launch(
        css=custom_css
    )