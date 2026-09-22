import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> List[Document]:
    """Load one PDF file."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)

    return loader.load()


def load_pdfs_from_directory(directory_path: str) -> List[Document]:
    """Load all PDF files from a folder."""

    if not os.path.exists(directory_path):
        raise FileNotFoundError(
            f"Directory not found: {directory_path}"
        )

    loader = PyPDFDirectoryLoader(directory_path)

    return loader.load()


def split_documents(
    documents: List[Document]
) -> List[Document]:
    """Split documents into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return text_splitter.split_documents(documents)


def ingest_file(file_path: str) -> List[Document]:
    """Load and split one PDF."""

    documents = load_pdf(file_path)
    return split_documents(documents)


def ingest_directory(
    directory_path: str
) -> List[Document]:
    """Load and split all PDFs inside a folder."""

    documents = load_pdfs_from_directory(
        directory_path
    )

    return split_documents(documents)