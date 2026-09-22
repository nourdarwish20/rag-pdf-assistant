import os
import shutil
import gradio as gr

from src.config import INPUTS_DIR
from src.ingest import ingest_file
from src.vector_store import (
    update_vector_store,
    load_vector_store,
)
from src.rag import answer_question


# Hugging Face is used for the deployed app
APP_PROVIDER = "huggingface"


# ------------------------------------------------
# PDF Upload
# ------------------------------------------------

def process_upload(files):
    """Save uploaded PDFs and add them to FAISS."""

    if not files:
        return "Please select at least one PDF."

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

            save_path = INPUTS_DIR / filename

            # Save PDF
            shutil.copy2(
                file_path,
                save_path
            )

            # PDF → LangChain chunks
            documents = ingest_file(
                str(save_path)
            )

            all_documents.extend(documents)

            status_messages.append(
                f"✓ {filename}: {len(documents)} chunks"
            )

        # Add documents to FAISS
        vectorstore = update_vector_store(
            all_documents
        )

        status_messages.append(
            f"\n✓ Knowledge base ready"
            f"\n{vectorstore.index.ntotal} vectors stored"
        )

        return "\n".join(status_messages)

    except Exception as e:
        return f"Error: {str(e)}"


# ------------------------------------------------
# Chat
# ------------------------------------------------

def chat(message, history):
    """Run the RAG pipeline and update chat history."""

    if not message.strip():
        return history, ""

    try:
        vectorstore = load_vector_store()

        if vectorstore is None:

            assistant_message = (
                "Please upload and process a PDF first."
            )

        else:

            result = answer_question(
                vectorstore,
                message,
                provider=APP_PROVIDER
            )

            answer = result["answer"]

            # Build source list
            source_lines = []

            for document in result["sources"]:

                source = document.metadata.get(
                    "source",
                    "Unknown"
                )

                page = document.metadata.get(
                    "page"
                )

                source_name = os.path.basename(
                    source
                )

                if page is not None:
                    source_line = (
                        f"📄 {source_name} · "
                        f"Page {page + 1}"
                    )
                else:
                    source_line = (
                        f"📄 {source_name}"
                    )

                if source_line not in source_lines:
                    source_lines.append(
                        source_line
                    )

            if source_lines:

                sources = "\n".join(
                    source_lines
                )

                assistant_message = (
                    f"{answer}\n\n"
                    f"**Sources**\n{sources}"
                )

            else:
                assistant_message = answer

    except Exception as e:

        assistant_message = (
            f"Something went wrong: {str(e)}"
        )

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
        inputs=pdf_files,
        outputs=status
    )

    send_button.click(
        fn=chat,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot,
            message
        ]
    )

    message.submit(
        fn=chat,
        inputs=[
            message,
            chatbot
        ],
        outputs=[
            chatbot,
            message
        ]
    )

    clear_button.click(
        fn=clear_chat,
        outputs=chatbot
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