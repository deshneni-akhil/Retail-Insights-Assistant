import pytest
import json
from pathlib import Path
from src.session import SessionManager


@pytest.fixture
def sm(tmp_path):
    return SessionManager(tmp_path / "sessions")


class TestSessionManager:
    def test_create_session(self, sm):
        sm.create_session("test_session")
        assert sm.active_session == "test_session"
        assert sm.db is not None
        assert sm.vectorstore is not None

        session_dir = sm.sessions_dir / "test_session"
        assert session_dir.exists()
        assert (session_dir / "data.duckdb").exists()
        assert (session_dir / "chromadb").exists()
        assert (session_dir / "meta.json").exists()

    def test_create_session_sanitizes_name(self, sm):
        sm.create_session("My Test Session!")
        assert sm.active_session == "my_test_session"

    def test_create_duplicate_raises(self, sm):
        sm.create_session("demo")
        sm.close()
        with pytest.raises(ValueError, match="already exists"):
            sm.create_session("demo")

    def test_list_sessions(self, sm):
        sm.create_session("alpha")
        sm.close()
        sm.create_session("beta")
        sm.close()

        sessions = sm.list_sessions()
        names = [s["name"] for s in sessions]
        assert "alpha" in names
        assert "beta" in names

    def test_list_sessions_empty(self, sm):
        assert sm.list_sessions() == []

    def test_load_session(self, sm):
        sm.create_session("persist_test")
        # Register some data
        import pandas as pd
        sm.db.ingest_dataframe("tbl", pd.DataFrame({"a": [1, 2, 3]}))
        sm.close()

        # Reload
        sm.load_session("persist_test")
        assert sm.active_session == "persist_test"
        tables = sm.db.list_tables()
        assert "tbl" in tables

    def test_load_nonexistent_raises(self, sm):
        with pytest.raises(ValueError, match="not found"):
            sm.load_session("nonexistent")

    def test_delete_session(self, sm):
        sm.create_session("to_delete")
        sm.close()
        sm.delete_session("to_delete")
        assert not (sm.sessions_dir / "to_delete").exists()
        assert sm.list_sessions() == []

    def test_delete_active_session(self, sm):
        sm.create_session("active_del")
        sm.delete_session("active_del")
        assert sm.active_session is None
        assert sm.db is None

    def test_update_meta(self, sm):
        sm.create_session("meta_test")
        sm.update_meta("sales", "sales.csv", 1000)

        meta_path = sm.sessions_dir / "meta_test" / "meta.json"
        meta = json.loads(meta_path.read_text())
        assert len(meta["tables"]) == 1
        assert meta["tables"][0]["name"] == "sales"
        assert meta["tables"][0]["row_count"] == 1000

    def test_update_meta_replaces_existing(self, sm):
        sm.create_session("replace_test")
        sm.update_meta("sales", "sales_v1.csv", 100)
        sm.update_meta("sales", "sales_v2.csv", 200)

        meta_path = sm.sessions_dir / "replace_test" / "meta.json"
        meta = json.loads(meta_path.read_text())
        assert len(meta["tables"]) == 1
        assert meta["tables"][0]["row_count"] == 200

    def test_get_dynamic_schema(self, sm):
        sm.create_session("schema_test")
        import pandas as pd
        sm.db.ingest_dataframe("orders", pd.DataFrame({
            "id": [1, 2],
            "product": ["Shoes", "Bags"],
            "amount": [100.0, 200.0],
        }))

        schema = sm.get_dynamic_schema()
        assert "orders" in schema
        assert "id" in schema
        assert "product" in schema
        assert "amount" in schema

    def test_get_dynamic_schema_no_session(self, sm):
        schema = sm.get_dynamic_schema()
        assert "No active session" in schema

    def test_close(self, sm):
        sm.create_session("close_test")
        sm.close()
        assert sm.active_session is None
        assert sm.db is None
        assert sm.vectorstore is None

    def test_switching_sessions(self, sm):
        sm.create_session("session_a")
        import pandas as pd
        sm.db.ingest_dataframe("tbl_a", pd.DataFrame({"a": [1]}))

        sm.create_session("session_b")
        sm.db.ingest_dataframe("tbl_b", pd.DataFrame({"b": [2]}))

        # Session B should only have tbl_b
        assert "tbl_b" in sm.db.list_tables()
        assert "tbl_a" not in sm.db.list_tables()

        # Switch back to A
        sm.load_session("session_a")
        assert "tbl_a" in sm.db.list_tables()
        assert "tbl_b" not in sm.db.list_tables()

    def test_get_dynamic_schema_with_query(self, sm):
        """With a user query and vectorstore, schema selection is used."""
        sm.create_session("query_schema")
        import pandas as pd
        sm.db.ingest_dataframe("sales", pd.DataFrame({
            "id": [1, 2], "revenue": [100, 200],
        }))
        sm.db.ingest_dataframe("returns", pd.DataFrame({
            "id": [1], "reason": ["defective"],
        }))
        # Index tables
        for t in sm.db.list_tables():
            desc = sm.db.get_compact_description(t)
            sm.vectorstore.index_table_schema(t, desc)

        # With only 2 tables, fast path — query param shouldn't change result
        schema = sm.get_dynamic_schema("revenue by product")
        assert "sales" in schema
        assert "returns" in schema

    def test_session_load_indexes_tables(self, sm):
        """Loading a session should index existing tables in ChromaDB."""
        sm.create_session("index_test")
        import pandas as pd
        sm.db.ingest_dataframe("orders", pd.DataFrame({"x": [1, 2]}))
        sm.close()

        # Reload — _activate should call _index_existing_tables
        sm.load_session("index_test")
        counts = sm.vectorstore.get_collection_counts()
        assert counts["table_schemas"] == 1
