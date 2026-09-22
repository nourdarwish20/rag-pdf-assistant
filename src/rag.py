from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from src.config import RETRIEVER_K, LLM_PROVIDER
from src.llm import generate_answer


def get_retriever(vectorstore) -> BaseRetriever:
    """Create a retriever from the FAISS vector store."""

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVER_K
        }
    )


def build_prompt(
    question: str,
    documents: list[Document]
) -> str:
    """Build the prompt using retrieved documents."""

    context_parts = []

    for document in documents:
        context_parts.append(
            document.page_content
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
Use only the context below to answer the question.

If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def answer_question(
    vectorstore,
    question: str,
    provider: str = LLM_PROVIDER
):
    """Retrieve relevant documents and generate an answer."""

    retriever = get_retriever(vectorstore)

    relevant_documents = retriever.invoke(
        question
    )

    prompt = build_prompt(
        question,
        relevant_documents
    )

    answer = generate_answer(
        prompt,
        provider=provider
    )

    return {
        "answer": answer,
        "sources": relevant_documents
    }