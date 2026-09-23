import re

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from src.config import RETRIEVER_K, LLM_PROVIDER
from src.llm import generate_answer
from src.web_search import search_web


# The exact sentence the model must use when the PDF has no answer.
# Seeing it is what triggers the web fallback.
NOT_IN_DOCUMENT = "I don't know based on the provided document."

# Whole messages that only make sense after a previous answer.
FOLLOW_UP_MESSAGES = {
    "explain more",
    "tell me more",
    "more",
    "why",
    "elaborate",
}

# Openings that point back at the previous answer
# ("what about that?", "i meant in the cake", "can i add lemon?").
FOLLOW_UP_OPENINGS = (
    "what about",
    "how about",
    "i mean",
    "i meant",
    "can i add",
    "can i put",
    "can i replace",
)


def get_retriever(vectorstore) -> BaseRetriever:
    """Create a retriever from the FAISS vector store."""

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": RETRIEVER_K
        }
    )


def is_follow_up(question: str) -> bool:
    """True only for obvious follow-ups like "explain more" or "why?".

    Everything else, including short questions like "What is RAG?",
    is treated as a new, standalone question.
    """

    text = " ".join(re.findall(r"[a-z']+", question.lower()))

    if text in FOLLOW_UP_MESSAGES:
        return True

    return any(
        text == opening or text.startswith(opening + " ")
        for opening in FOLLOW_UP_OPENINGS
    )


def build_search_query(
    question: str,
    previous_question: str = None
) -> str:
    """Pick the text used to search FAISS.

    "explain more" alone matches nothing, so a follow-up is
    searched together with the question it follows.
    """

    if previous_question and is_follow_up(question):
        return f"{previous_question} {question}"

    return question


def build_prompt(
    question: str,
    documents: list[Document],
    previous_question: str = None,
    previous_answer: str = None
) -> str:
    """Build the prompt using retrieved documents."""

    context_parts = []

    for document in documents:
        context_parts.append(
            document.page_content
        )

    context = "\n\n".join(context_parts)

    previous_turn = ""

    if previous_question and previous_answer:
        previous_turn = (
            f"\nPrevious question:\n{previous_question}\n"
            f"\nPrevious answer:\n{previous_answer}\n"
        )

    prompt = f"""
You are an assistant that answers questions about one PDF document.

Rules:
- Use only the context below. Never add facts from outside it.
- If the user asks you to explain more, expand on the previous
  answer using the same context. Do not invent new facts.
- Stay on the topic of the document and the question asked.
- If the answer is not in the context, reply with exactly:
"{NOT_IN_DOCUMENT}"
{previous_turn}
Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def build_web_prompt(
    question: str,
    results: list[dict],
    topic: str = None
) -> str:
    """Build a prompt from web snippets, used only as a fallback."""

    context_parts = []

    for result in results:
        context_parts.append(
            f"{result['title']}\n{result['snippet']}"
        )

    context = "\n\n".join(context_parts)

    topic_line = ""

    if topic and topic.strip() != question.strip():
        topic_line = (
            f"\nThe conversation is about: {topic}\n"
            "Read the question in that light: words like "
            '"it" or "that" refer to this topic.\n'
        )

    prompt = f"""
The user's PDF did not contain the answer, so here are web results.

Rules:
- Use only the web results below.
- The question may contain typos (e.g. "recope" means "recipe").
  Answer what the user clearly meant, not the literal spelling.
- Answer only the question asked. Stay on its topic.
- If any result answers it, use that result, even if it is only
  one example (e.g. one recipe when the user asked for "a recipe").
- Keep the answer short: at most four sentences, or a short list
  when the answer is steps or ingredients.
- Only if no result is relevant, say you could not find it.
{topic_line}
Web results:
{context}

Question:
{question}

Answer:
"""

    return prompt


def answer_question(
    vectorstore,
    question: str,
    provider: str = LLM_PROVIDER,
    previous_question: str = None,
    previous_answer: str = None
):
    """Retrieve relevant documents and generate an answer.

    The PDF is always tried first. The web is used only when the
    PDF cannot answer.
    """

    search_query = build_search_query(
        question,
        previous_question
    )

    retriever = get_retriever(vectorstore)

    relevant_documents = retriever.invoke(
        search_query
    )

    prompt = build_prompt(
        question,
        relevant_documents,
        previous_question,
        previous_answer
    )

    answer = generate_answer(
        prompt,
        provider=provider
    )

    # PDF first. Only fall back when it truly has no answer.
    if "don't know based on" in answer.lower():

        web_results = search_web(search_query)

        if web_results:

            web_answer = generate_answer(
                build_web_prompt(
                    question,
                    web_results,
                    previous_question
                ),
                provider=provider
            )

            return {
                "answer": web_answer,
                "sources": [],
                "web_sources": web_results,
                "from_web": True,
            }

    return {
        "answer": answer,
        "sources": relevant_documents,
        "web_sources": [],
        "from_web": False,
    }
