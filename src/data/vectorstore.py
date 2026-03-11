# Session-scoped ChromaDB vector store for RAG.
import chromadb
from datetime import datetime


class SessionVectorStore:
    """ChromaDB vector store scoped to a single analysis session."""

    def __init__(self, chromadb_path: str):
        self.client = chromadb.PersistentClient(path=chromadb_path)

    def _get_collection(self, name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(name=name)

    def add_summary(self, text: str, metadata: dict | None = None) -> None:
        """Store a generated executive summary."""
        collection = self._get_collection("summaries")
        meta = {"type": "summary", "created_at": datetime.now().isoformat()}
        if metadata:
            meta.update(metadata)
        doc_id = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        collection.add(documents=[text], metadatas=[meta], ids=[doc_id])

    def add_document(self, text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> int:
        """Chunk and store an uploaded document. Returns number of chunks stored."""
        collection = self._get_collection("documents")
        chunks = _chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        ids = [f"doc_{source}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "type": "document",
                "source": source,
                "chunk_index": i,
                "created_at": datetime.now().isoformat(),
            }
            for i in range(len(chunks))
        ]
        collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def add_qa_pair(self, question: str, sql: str, answer: str) -> None:
        """Store a successful question-answer pair.

        The *document* (what gets embedded) is the question alone, so cache
        lookups compare query-to-question distance directly.  The SQL,
        answer, and reasoning are kept in metadata for retrieval.
        """
        collection = self._get_collection("qa_pairs")
        doc_id = f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        meta = {
            "type": "qa_pair",
            "question": question[:500],
            "sql": sql[:2000],
            "answer": answer[:5000],
            "created_at": datetime.now().isoformat(),
        }
        collection.add(documents=[question], metadatas=[meta], ids=[doc_id])

    def search_qa_cache(self, query: str, top_k: int = 1) -> list[dict]:
        """Search Q&A pairs by question similarity.

        Returns list of dicts with keys: question, sql, answer, distance.
        Because the embedded document is the question only, distances are
        meaningful for cache-hit thresholds.
        """
        collection = self._get_collection("qa_pairs")
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query], n_results=min(top_k, collection.count())
        )

        output = []
        for i in range(len(results["documents"][0])):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            output.append({
                "question": meta.get("question", ""),
                "sql": meta.get("sql", ""),
                "answer": meta.get("answer", ""),
                "distance": results["distances"][0][i] if results["distances"] else 1.0,
            })
        return output

    def index_table_schema(self, table_name: str, description: str) -> None:
        """Index a table's compact description for semantic table retrieval."""
        collection = self._get_collection("table_schemas")
        doc_id = f"schema_{table_name}"
        meta = {
            "type": "table_schema",
            "table_name": table_name,
            "created_at": datetime.now().isoformat(),
        }
        collection.upsert(documents=[description], metadatas=[meta], ids=[doc_id])

    def remove_table_schema(self, table_name: str) -> None:
        """Remove a table's schema from the index."""
        collection = self._get_collection("table_schemas")
        try:
            collection.delete(ids=[f"schema_{table_name}"])
        except Exception:
            pass

    def search_tables(self, query: str, top_k: int = 3) -> list[str]:
        """Find the most relevant table names for a query via semantic search."""
        collection = self._get_collection("table_schemas")
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query], n_results=min(top_k, collection.count())
        )

        table_names = []
        for meta in results["metadatas"][0]:
            name = meta.get("table_name")
            if name and name not in table_names:
                table_names.append(name)
        return table_names

    def search(self, query: str, collection_name: str, top_k: int = 3) -> list[dict]:
        """Search a collection for relevant documents."""
        collection = self._get_collection(collection_name)
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query], n_results=min(top_k, collection.count())
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 1.0,
            })
        return output

    def search_all(self, query: str, top_k: int = 3) -> list[dict]:
        """Search across all collections and merge results by relevance."""
        all_results = []
        for collection_name in ["summaries", "documents", "qa_pairs"]:
            results = self.search(query, collection_name, top_k=top_k)
            for r in results:
                r["collection"] = collection_name
            all_results.extend(results)

        all_results.sort(key=lambda x: x["distance"])
        return all_results[:top_k]

    def get_collection_counts(self) -> dict[str, int]:
        """Get document counts for all collections."""
        counts = {}
        for name in ["summaries", "documents", "qa_pairs", "table_schemas"]:
            try:
                counts[name] = self._get_collection(name).count()
            except Exception:
                counts[name] = 0
        return counts


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
