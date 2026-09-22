from pypdf import PdfReader
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def read_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text):
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end]

        chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest_pdf(file_path):
    text = read_pdf(file_path)
    chunks = split_text(text)

    return chunks