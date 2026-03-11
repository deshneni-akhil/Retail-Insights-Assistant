import pytest
import pandas as pd
from src.data.db import Database
from src.data.vectorstore import SessionVectorStore
from src.agents.schema_selector import select_schema


@pytest.fixture
def db():
    return Database(None)  # in-memory


@pytest.fixture
def vs(tmp_path):
    return SessionVectorStore(str(tmp_path / "chromadb"))


def _add_tables(db, vs, count):
    """Helper: create N tables and index them in the vectorstore."""
    for i in range(count):
        name = f"table_{i}"
        db.ingest_dataframe(name, pd.DataFrame({f"col_{i}": range(5)}))
        description = db.get_compact_description(name)
        vs.index_table_schema(name, description)


class TestSchemaSelector:
    def test_small_schema_passthrough(self, db, vs):
        """<=5 tables should return all schemas without selection."""
        _add_tables(db, vs, 3)
        result = select_schema(db, vs, "show me data")
        assert "table_0" in result
        assert "table_1" in result
        assert "table_2" in result
        # Should NOT have the TABLE DIRECTORY header (fast path)
        assert "TABLE DIRECTORY" not in result

    def test_exact_threshold_passthrough(self, db, vs):
        """Exactly 5 tables should still use fast path."""
        _add_tables(db, vs, 5)
        result = select_schema(db, vs, "anything")
        assert "TABLE DIRECTORY" not in result

    def test_table_selection_triggered(self, db, vs):
        """6+ tables should trigger semantic selection."""
        _add_tables(db, vs, 7)
        result = select_schema(db, vs, "show table_0 data")
        # Should have the directory header
        assert "TABLE DIRECTORY" in result
        # Should have "DETAILED SCHEMA" section
        assert "DETAILED SCHEMA" in result
        # Should mention selected count
        assert "of 7 tables" in result

    def test_table_directory_lists_all(self, db, vs):
        """Directory should list every table even when only some have full schemas."""
        _add_tables(db, vs, 8)
        result = select_schema(db, vs, "query")
        for i in range(8):
            assert f"table_{i}" in result

    def test_no_tables(self, db, vs):
        """Empty database should return helpful message."""
        result = select_schema(db, vs, "anything")
        assert "No tables" in result

    def test_empty_vectorstore_fallback(self, db, vs):
        """If vectorstore has no indexed schemas, fall back to all schemas."""
        for i in range(7):
            db.ingest_dataframe(f"t{i}", pd.DataFrame({"x": [1]}))
        # Don't index in vectorstore
        result = select_schema(db, vs, "query")
        # Falls back to all schemas
        assert "t0" in result
        assert "t6" in result
