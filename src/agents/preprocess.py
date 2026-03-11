# Preprocess agent — cache lookup and input sanitization before the main pipeline.
from src.data.vectorstore import SessionVectorStore


def sanitize_input(query: str) -> str | None:
    """Validate and sanitize user input.

    Returns an error message string if input is invalid, or None if OK.
    """
    stripped = query.strip()
    if not stripped:
        return "Please enter a question."
    if len(stripped) < 3:
        return "Your question is too short. Please provide more detail."
    if len(stripped) > 2000:
        return "Your question is too long. Please keep it under 2,000 characters."
    return None


def check_cache(
    vectorstore: SessionVectorStore, query: str, threshold: float
) -> dict | None:
    """Search Q&A pairs for a semantically similar past question.

    Returns a dict with keys {answer, sql, question} if cache hit, else None.
    Uses search_qa_cache which embeds by question only, so distances
    are accurate for cache-hit comparison.
    """
    results = vectorstore.search_qa_cache(query, top_k=1)
    if not results:
        return None

    best = results[0]
    if best["distance"] < threshold:
        return {
            "answer": best["answer"],
            "sql": best["sql"],
            "question": best["question"],
        }

    return None
