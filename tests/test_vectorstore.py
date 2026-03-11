import pytest
from src.data.vectorstore import SessionVectorStore, _chunk_text


@pytest.fixture
def vs(tmp_path):
    return SessionVectorStore(str(tmp_path / "chromadb"))


class TestChunking:
    def test_short_text_no_chunking(self):
        chunks = _chunk_text("Hello world", 500, 100)
        assert chunks == ["Hello world"]

    def test_empty_text(self):
        chunks = _chunk_text("", 500, 100)
        assert chunks == []

    def test_whitespace_only(self):
        chunks = _chunk_text("   ", 500, 100)
        assert chunks == []

    def test_long_text_chunked(self):
        text = "A" * 1000
        chunks = _chunk_text(text, 300, 50)
        assert len(chunks) >= 3
        assert all(len(c) > 0 for c in chunks)

    def test_overlap(self):
        text = "0123456789" * 10  # 100 chars
        chunks = _chunk_text(text, 30, 10)
        assert len(chunks) >= 4


class TestAddAndSearch:
    def test_add_summary(self, vs):
        vs.add_summary("Revenue was $1M in Q1")
        counts = vs.get_collection_counts()
        assert counts["summaries"] == 1

    def test_add_document(self, vs):
        count = vs.add_document("Some business doc content", source="test.txt")
        assert count >= 1
        counts = vs.get_collection_counts()
        assert counts["documents"] >= 1

    def test_add_qa_pair(self, vs):
        vs.add_qa_pair(
            question="Top category?",
            sql="SELECT category FROM sales ORDER BY revenue DESC LIMIT 1",
            answer="Shoes is the top category",
        )
        counts = vs.get_collection_counts()
        assert counts["qa_pairs"] == 1

    def test_search_summaries(self, vs):
        vs.add_summary("Total revenue was 71M INR from Amazon India sales")
        results = vs.search("revenue total", "summaries", top_k=1)
        assert len(results) == 1
        assert "revenue" in results[0]["text"].lower()
        assert "distance" in results[0]

    def test_search_empty_collection(self, vs):
        results = vs.search("anything", "summaries", top_k=3)
        assert results == []

    def test_search_all(self, vs):
        vs.add_summary("Summary about shoes sales")
        vs.add_qa_pair("shoe question", "SELECT...", "shoes answer")
        results = vs.search_all("shoes", top_k=5)
        assert len(results) >= 2

    def test_get_collection_counts_empty(self, vs):
        counts = vs.get_collection_counts()
        assert counts["summaries"] == 0
        assert counts["documents"] == 0
        assert counts["qa_pairs"] == 0

    def test_document_reuploads_via_upsert(self, vs):
        vs.add_document("Version 1", source="notes.txt")
        count1 = vs.get_collection_counts()["documents"]
        vs.add_document("Version 2", source="notes.txt")
        count2 = vs.get_collection_counts()["documents"]
        assert count1 == count2


class TestTableSchemas:
    def test_index_and_search_table_schema(self, vs):
        vs.index_table_schema("sales", "Table: sales (1000 rows). Columns: id, product, revenue")
        vs.index_table_schema("returns", "Table: returns (200 rows). Columns: id, reason, amount")
        vs.index_table_schema("inventory", "Table: inventory (500 rows). Columns: sku, quantity, warehouse")

        results = vs.search_tables("revenue by product", top_k=2)
        assert "sales" in results
        assert len(results) <= 2

    def test_search_tables_empty(self, vs):
        results = vs.search_tables("anything", top_k=3)
        assert results == []

    def test_remove_table_schema(self, vs):
        vs.index_table_schema("temp_table", "Table: temp_table. Columns: x, y")
        assert vs.get_collection_counts()["table_schemas"] == 1

        vs.remove_table_schema("temp_table")
        assert vs.get_collection_counts()["table_schemas"] == 0

    def test_upsert_updates_existing(self, vs):
        vs.index_table_schema("t1", "Version 1")
        vs.index_table_schema("t1", "Version 2 with more columns")
        assert vs.get_collection_counts()["table_schemas"] == 1

    def test_collection_counts_includes_table_schemas(self, vs):
        counts = vs.get_collection_counts()
        assert "table_schemas" in counts
        assert counts["table_schemas"] == 0


class TestQACache:
    def test_search_qa_cache_empty(self, vs):
        results = vs.search_qa_cache("anything", top_k=1)
        assert results == []

    def test_search_qa_cache_returns_metadata(self, vs):
        vs.add_qa_pair(
            question="What is total revenue?",
            sql="SELECT SUM(revenue) FROM sales",
            answer="The total revenue is 71M INR.",
        )
        results = vs.search_qa_cache("What is total revenue?", top_k=1)
        assert len(results) == 1
        assert results[0]["question"] == "What is total revenue?"
        assert "SUM(revenue)" in results[0]["sql"]
        assert "71M" in results[0]["answer"]
        assert results[0]["distance"] < 0.1  # near-exact question match

    def test_qa_embeds_question_only(self, vs):
        """Verify that the embedded document is the question, not the full Q/SQL/A blob."""
        vs.add_qa_pair(
            question="Top selling category",
            sql="SELECT category FROM sales ORDER BY revenue DESC LIMIT 1",
            answer="Electronics is the top selling category.",
        )
        # Generic search should return the question as the document text
        results = vs.search("Top selling category", "qa_pairs", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "Top selling category"
        # The full answer is in metadata, not in the document
        assert results[0]["metadata"]["answer"] == "Electronics is the top selling category."

    def test_qa_cache_distance_is_low_for_similar_questions(self, vs):
        vs.add_qa_pair("monthly revenue trends", "SELECT ...", "Revenue grew 15%.")
        results = vs.search_qa_cache("revenue trends by month", top_k=1)
        assert len(results) == 1
        # Similar questions should have low distance (question-to-question)
        assert results[0]["distance"] < 0.5
