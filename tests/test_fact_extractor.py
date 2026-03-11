import pytest
from src.agents.fact_extractor import extract_facts, _format_value


class TestFormatValue:
    def test_none_returns_none(self):
        assert _format_value(None) is None

    def test_integer(self):
        assert _format_value(42) == "42"

    def test_float(self):
        assert _format_value(3.14) == "3.14"

    def test_string_quoted(self):
        assert _format_value("Shoes") == "'Shoes'"

    def test_bool(self):
        assert _format_value(True) == "True"


class TestExtractFactsEmpty:
    def test_failed_query(self):
        result = {"success": False, "results": [], "error": "bad sql", "row_count": 0}
        assert extract_facts(result) == ""

    def test_empty_results(self):
        result = {"success": True, "results": [], "error": "", "row_count": 0}
        assert extract_facts(result) == ""

    def test_missing_success(self):
        result = {"results": [{"a": 1}]}
        assert extract_facts(result) == ""


class TestSingleRow:
    def test_aggregation_result(self):
        result = {
            "success": True,
            "results": [{"total_revenue": 71000000, "order_count": 15420}],
            "row_count": 1,
        }
        facts = extract_facts(result)
        assert "VERIFIED FACTS FROM DATA:" in facts
        assert "total_revenue = 71000000" in facts
        assert "order_count = 15420" in facts

    def test_single_count(self):
        result = {"success": True, "results": [{"n": 42}], "row_count": 1}
        facts = extract_facts(result)
        assert "n = 42" in facts

    def test_none_values_skipped(self):
        result = {"success": True, "results": [{"a": 1, "b": None}], "row_count": 1}
        facts = extract_facts(result)
        assert "a = 1" in facts
        assert "b" not in facts


class TestMultiRow:
    def test_basic_multi_row(self):
        result = {
            "success": True,
            "results": [
                {"category": "Shoes", "revenue": 45000},
                {"category": "Bags", "revenue": 32000},
            ],
            "row_count": 2,
        }
        facts = extract_facts(result)
        assert "2 rows" in facts
        assert "Row 1:" in facts
        assert "category = 'Shoes'" in facts
        assert "revenue = 45000" in facts
        assert "Row 2:" in facts
        assert "category = 'Bags'" in facts

    def test_numeric_aggregates(self):
        result = {
            "success": True,
            "results": [
                {"name": "A", "value": 10},
                {"name": "B", "value": 30},
                {"name": "C", "value": 20},
            ],
            "row_count": 3,
        }
        facts = extract_facts(result)
        assert "Numeric summaries:" in facts
        assert "value: min = 10, max = 30, total = 60" in facts

    def test_truncation_at_20_rows(self):
        rows = [{"id": i, "val": i * 10} for i in range(50)]
        result = {"success": True, "results": rows, "row_count": 50}
        facts = extract_facts(result)
        assert "50 rows" in facts
        assert "showing first 20" in facts
        assert "Row 20:" in facts
        assert "Row 21:" not in facts
        # Aggregates should still use ALL 50 rows
        assert f"total = {sum(r['val'] for r in rows)}" in facts

    def test_mixed_types(self):
        result = {
            "success": True,
            "results": [
                {"name": "X", "count": 5, "flag": True, "note": None},
            ],
            "row_count": 1,
        }
        facts = extract_facts(result)
        assert "name = 'X'" in facts
        assert "count = 5" in facts
        assert "flag = True" in facts
        assert "note" not in facts
