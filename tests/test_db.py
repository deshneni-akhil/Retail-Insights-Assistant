import pytest
import pandas as pd
from src.data.db import Database


@pytest.fixture
def db():
    d = Database()
    yield d
    d.close()


@pytest.fixture
def db_with_data(db):
    df = pd.DataFrame({
        "order_id": ["001", "002", "003"],
        "category": ["SHOES", "BAGS", "SHOES"],
        "amount": [100.0, 200.0, 150.0],
        "qty": [1, 2, 1],
    })
    db.ingest_dataframe("test_orders", df)
    return db


class TestDatabase:
    def test_ingest_dataframe_and_query(self, db_with_data):
        results = db_with_data.execute("SELECT * FROM test_orders")
        assert len(results) == 3
        assert results[0]["order_id"] == "001"

    def test_execute_returns_list_of_dicts(self, db_with_data):
        results = db_with_data.execute(
            "SELECT category, SUM(amount) as total FROM test_orders GROUP BY category"
        )
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        categories = {r["category"] for r in results}
        assert categories == {"SHOES", "BAGS"}

    def test_execute_aggregate(self, db_with_data):
        results = db_with_data.execute("SELECT SUM(amount) as total FROM test_orders")
        assert results[0]["total"] == 450.0

    def test_execute_invalid_sql_raises(self, db):
        with pytest.raises(RuntimeError, match="SQL execution error"):
            db.execute("SELECT * FROM nonexistent_table")

    def test_explain_valid_sql(self, db_with_data):
        err = db_with_data.explain("SELECT * FROM test_orders")
        assert err == ""

    def test_explain_invalid_sql(self, db_with_data):
        err = db_with_data.explain("SELECT * FROM nonexistent")
        assert err != ""

    def test_get_table_info(self, db_with_data):
        info = db_with_data.get_table_info("test_orders")
        assert len(info) == 4
        col_names = [c["column_name"] for c in info]
        assert "order_id" in col_names
        assert "amount" in col_names

    def test_get_table_info_nonexistent(self, db):
        info = db.get_table_info("nonexistent")
        assert info == []

    def test_list_tables(self, db_with_data):
        tables = db_with_data.list_tables()
        assert "test_orders" in tables

    def test_list_tables_empty(self, db):
        assert db.list_tables() == []

    def test_get_schema_text(self, db_with_data):
        schema = db_with_data.get_schema_text("test_orders")
        assert "test_orders" in schema
        assert "order_id" in schema
        assert "amount" in schema
        assert "3 rows" in schema

    def test_get_all_schemas_text(self, db_with_data):
        schema = db_with_data.get_all_schemas_text()
        assert "test_orders" in schema

    def test_get_all_schemas_text_empty(self, db):
        schema = db.get_all_schemas_text()
        assert "No tables" in schema

    def test_drop_table(self, db_with_data):
        db_with_data.drop_table("test_orders")
        assert "test_orders" not in db_with_data.list_tables()

    def test_ingest_file_csv(self, db, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,value\nAlice,10\nBob,20")
        count = db.ingest_file("csv_table", str(csv_path), "csv")
        assert count == 2
        results = db.execute("SELECT * FROM csv_table")
        assert len(results) == 2

    def test_ingest_file_json(self, db, tmp_path):
        json_path = tmp_path / "test.json"
        json_path.write_text('[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]')
        count = db.ingest_file("json_table", str(json_path), "json")
        assert count == 2
        results = db.execute("SELECT * FROM json_table")
        assert len(results) == 2
