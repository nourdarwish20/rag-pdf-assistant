import gradio as gr

from src.ingest import ingest_pdf
from src.vector_store import create_vector_store
from src.rag import answer_question


chunks = []
index = None


def process_pdf(file_path):
    global chunks, index

    if not file_path:
        return "Please upload a PDF."

    chunks = ingest_pdf(file_path)
    index = create_vector_store(chunks)

    return f"PDF ready. {len(chunks)} chunks created."


def ask_question(question):
    if index is None:
        return "Please upload and process a PDF first."

    if not question.strip():
        return "Please enter a question."

    return answer_question(index, chunks, question)


with gr.Blocks(title="RAG PDF Assistant") as demo:

    gr.Markdown("# RAG PDF Assistant")
    gr.Markdown("Upload a PDF, then ask questions about it.")

    pdf_file = gr.File(
        label="Upload PDF",
        file_types=[".pdf"],
        type="filepath"
    )

    process_button = gr.Button("Process PDF")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    process_button.click(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=status
    )

    question = gr.Textbox(
        label="Question",
        placeholder="Ask something about the PDF..."
    )

    ask_button = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        interactive=False
    )

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=answer
    )


if __name__ == "__main__":
    demo.launch()