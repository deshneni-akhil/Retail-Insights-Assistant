import pytest
from src.agents.preprocess import sanitize_input, check_cache
from src.data.vectorstore import SessionVectorStore


# --- sanitize_input tests (pure Python) ---

class TestSanitizeInput:
    def test_empty_string(self):
        assert sanitize_input("") is not None
        assert "enter a question" in sanitize_input("").lower()

    def test_whitespace_only(self):
        assert sanitize_input("   ") is not None

    def test_too_short(self):
        result = sanitize_input("hi")
        assert result is not None
        assert "too short" in result.lower()

    def test_exactly_three_chars_valid(self):
        assert sanitize_input("why") is None

    def test_normal_question(self):
        assert sanitize_input("What is the total revenue?") is None

    def test_too_long(self):
        result = sanitize_input("x" * 2001)
        assert result is not None
        assert "too long" in result.lower()

    def test_exactly_2000_chars_valid(self):
        assert sanitize_input("x" * 2000) is None


# --- check_cache tests (requires ChromaDB) ---

@pytest.fixture
def vs(tmp_path):
    return SessionVectorStore(str(tmp_path / "chromadb"))


class TestCheckCache:
    def test_empty_collection(self, vs):
        result = check_cache(vs, "anything", threshold=0.15)
        assert result is None

    def test_exact_match_returns_dict(self, vs):
        vs.add_qa_pair(
            question="What is total revenue?",
            sql="SELECT SUM(revenue) FROM sales",
            answer="The total revenue is 71M INR.",
        )
        # Question-to-question distance should be very low for exact match
        result = check_cache(vs, "What is total revenue?", threshold=0.15)
        assert result is not None
        assert "71M" in result["answer"]
        assert "SUM(revenue)" in result["sql"]
        assert result["question"] == "What is total revenue?"

    def test_unrelated_query_no_hit(self, vs):
        vs.add_qa_pair(
            question="What is total revenue?",
            sql="SELECT SUM(revenue) FROM sales",
            answer="The total revenue is 71M INR.",
        )
        result = check_cache(vs, "How many warehouses exist in Asia?", threshold=0.15)
        assert result is None

    def test_answer_contains_full_response(self, vs):
        vs.add_qa_pair(
            question="Top category?",
            sql="SELECT category FROM sales ORDER BY revenue DESC LIMIT 1",
            answer="Shoes is the top category with 45K revenue.",
        )
        result = check_cache(vs, "Top category?", threshold=0.15)
        assert result is not None
        assert "Shoes" in result["answer"]
        assert "SELECT category" in result["sql"]

    def test_threshold_boundary(self, vs):
        vs.add_qa_pair(
            question="Monthly trends",
            sql="SELECT ...",
            answer="Revenue increased 15% MoM.",
        )
        # With threshold=0.0, nothing should match (distance must be < 0)
        result = check_cache(vs, "Monthly trends", threshold=0.0)
        assert result is None

    def test_similar_question_within_threshold(self, vs):
        vs.add_qa_pair(
            question="What is the total revenue for all products?",
            sql="SELECT SUM(revenue) FROM sales",
            answer="Total revenue is 50M.",
        )
        # Semantically similar question should hit within a reasonable threshold
        result = check_cache(vs, "total revenue across all products", threshold=0.5)
        assert result is not None
        assert "50M" in result["answer"]
