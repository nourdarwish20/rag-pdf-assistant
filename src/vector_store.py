import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, TOP_K


# Load the embedding model once
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def create_vector_store(chunks):
    """
    Convert chunks into embeddings and store them in FAISS.
    """

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_vector_store(index, chunks, query, top_k=TOP_K):
    """
    Search FAISS and return the most relevant chunks.
    """

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])

    return results