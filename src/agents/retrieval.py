# Retrieval agent — searches vector store for relevant context to enrich responses.
from src.config import RAG_SIMILARITY_THRESHOLD
from src.data.vectorstore import SessionVectorStore


def retrieve_context(vectorstore: SessionVectorStore, query: str, top_k: int = 3) -> str:
    """Search all collections for context relevant to the query.

    Returns formatted string of relevant context, or empty string if nothing relevant.
    """
    results = vectorstore.search_all(query, top_k=top_k)

    relevant = [r for r in results if r["distance"] < RAG_SIMILARITY_THRESHOLD]

    if not relevant:
        return ""

    lines = []
    for r in relevant:
        source = r.get("collection", "unknown")
        label = {
            "summaries": "Past Summary",
            "documents": "Business Document",
            "qa_pairs": "Past Q&A",
        }.get(source, source)
        # qa_pairs stores question as doc, full details in metadata
        if source == "qa_pairs":
            meta = r.get("metadata", {})
            text = f"Q: {meta.get('question', r['text'])}\nA: {meta.get('answer', '')}"
        else:
            text = r["text"]
        lines.append(f"[{label}] {text}")

    return "\n\n".join(lines)


def retrieve_similar_qa(vectorstore: SessionVectorStore, query: str, top_k: int = 3) -> str:
    """Search specifically for similar past Q&A pairs (used in fallback).

    Uses search_qa_cache for accurate question-to-question matching.
    """
    results = vectorstore.search_qa_cache(query, top_k=top_k)
    relevant = [r for r in results if r["distance"] < RAG_SIMILARITY_THRESHOLD]

    if not relevant:
        return ""

    lines = []
    for r in relevant:
        lines.append(f"Q: {r['question']}\nSQL: {r['sql']}\nA: {r['answer']}")
    return "\n\n---\n\n".join(lines)
