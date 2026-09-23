import os
from functools import lru_cache
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    VECTOR_DB_DIR,
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it for every upload."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def create_vector_store(
    documents: List[Document]
) -> FAISS:
    """Create a new FAISS vector store."""

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore


def save_vector_store(
    vectorstore: FAISS
) -> None:
    """Save FAISS locally."""

    vectorstore.save_local(
        str(VECTOR_DB_DIR)
    )


def load_vector_store() -> Optional[FAISS]:
    """Load an existing FAISS vector store."""

    index_file = VECTOR_DB_DIR / "index.faiss"

    if not index_file.exists():
        return None

    embeddings = get_embeddings()

    return FAISS.load_local(
        str(VECTOR_DB_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_existing_sources(
    vectorstore: FAISS
) -> set:
    """Get PDF files already stored."""

    sources = set()

    for document in vectorstore.docstore._dict.values():

        source = document.metadata.get("source")

        if source:
            source = os.path.basename(source)
            sources.add(source)

    return sources


def update_vector_store(
    documents: List[Document]
) -> FAISS:
    """Create or update the vector store."""

    vectorstore = load_vector_store()

    if vectorstore is None:

        vectorstore = create_vector_store(
            documents
        )

    else:

        existing_sources = get_existing_sources(
            vectorstore
        )

        new_documents = []

        for document in documents:

            source = document.metadata.get(
                "source"
            )

            if source:
                source = os.path.basename(source)

                if source in existing_sources:
                    continue

            new_documents.append(document)

        if new_documents:
            vectorstore.add_documents(
                new_documents
            )

    save_vector_store(vectorstore)

    return vectorstore


def update_session_store(
    vectorstore: Optional[FAISS],
    documents: List[Document]
) -> FAISS:
    """Create or extend one user's private in-memory store.

    Nothing is written to disk, so two visitors never share
    their PDFs. Same-file dedup still applies inside a session.
    """

    if vectorstore is None:
        return create_vector_store(documents)

    existing_sources = get_existing_sources(vectorstore)

    new_documents = []

    for document in documents:

        source = document.metadata.get("source")

        if source:
            source = os.path.basename(source)

            if source in existing_sources:
                continue

        new_documents.append(document)

    if new_documents:
        vectorstore.add_documents(new_documents)

    return vectorstore
