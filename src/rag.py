from src.vector_store import search_vector_store
from src.llm import generate_answer


def build_prompt(question, context_chunks):
    context = "\n\n".join(context_chunks)

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


def answer_question(index, chunks, question):
    relevant_chunks = search_vector_store(
        index,
        chunks,
        question
    )

    prompt = build_prompt(
        question,
        relevant_chunks
    )

    answer = generate_answer(prompt)

    return answer