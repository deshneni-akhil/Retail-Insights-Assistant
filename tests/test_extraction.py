import pytest
import pandas as pd
from src.data.db import Database
from src.agents.extraction import execute_query


@pytest.fixture
def db():
    d = Database()
    df = pd.DataFrame({
        "order_id": ["001", "002", "003"],
        "category": ["SHOES", "BAGS", "SHOES"],
        "amount": [100.0, 200.0, 150.0],
    })
    d.ingest_dataframe("test_orders", df)
    yield d
    d.close()


class TestExtraction:
    def test_valid_select(self, db):
        result = execute_query(db, "SELECT * FROM test_orders")
        assert result["success"] is True
        assert result["row_count"] == 3
        assert len(result["results"]) == 3
        assert result["error"] == ""

    def test_with_cte(self, db):
        sql = "WITH t AS (SELECT * FROM test_orders) SELECT * FROM t"
        result = execute_query(db, sql)
        assert result["success"] is True
        assert result["row_count"] == 3

    def test_block_non_select(self, db):
        result = execute_query(db, "INSERT INTO test_orders VALUES ('004', 'HAT', 50.0)")
        assert result["success"] is False
        assert "Only SELECT" in result["error"]

    def test_block_drop(self, db):
        result = execute_query(db, "SELECT 1; DROP TABLE test_orders")
        assert result["success"] is False
        assert "DROP" in result["error"]

    def test_block_delete(self, db):
        result = execute_query(db, "SELECT * FROM test_orders; DELETE FROM test_orders")
        assert result["success"] is False
        assert "DELETE" in result["error"]

    def test_block_update(self, db):
        result = execute_query(db, "SELECT 1; UPDATE test_orders SET amount=0")
        assert result["success"] is False
        assert "UPDATE" in result["error"]

    def test_block_alter(self, db):
        result = execute_query(db, "SELECT 1; ALTER TABLE test_orders ADD COLUMN x INT")
        assert result["success"] is False
        assert "ALTER" in result["error"]

    def test_block_truncate(self, db):
        result = execute_query(db, "SELECT 1; TRUNCATE test_orders")
        assert result["success"] is False
        assert "TRUNCATE" in result["error"]

    def test_sql_error_returns_failure(self, db):
        result = execute_query(db, "SELECT * FROM nonexistent")
        assert result["success"] is False
        assert result["row_count"] == 0

    def test_aggregate_query(self, db):
        result = execute_query(db, "SELECT SUM(amount) as total FROM test_orders")
        assert result["success"] is True
        assert result["results"][0]["total"] == 450.0
