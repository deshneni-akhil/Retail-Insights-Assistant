import pytest
from src.agents.validation import compute_data_quality, format_quality_report


class TestComputeDataQuality:
    def test_empty_results(self):
        metrics = compute_data_quality([])
        assert metrics["row_count"] == 0
        assert "0 rows" in metrics["issues"][0]

    def test_clean_data(self):
        results = [
            {"id": 1, "name": "Alice", "amount": 100.0},
            {"id": 2, "name": "Bob", "amount": 200.0},
            {"id": 3, "name": "Carol", "amount": 150.0},
        ]
        metrics = compute_data_quality(results)
        assert metrics["row_count"] == 3
        assert all(r == 0.0 for r in metrics["null_ratios"].values())
        assert metrics["negative_ratios"] == {}
        assert metrics["duplicate_ratio"] == 0.0
        assert metrics["issues"] == []

    def test_null_detection(self):
        results = [
            {"id": 1, "val": None},
            {"id": 2, "val": None},
            {"id": 3, "val": 10},
        ]
        metrics = compute_data_quality(results)
        assert metrics["null_ratios"]["val"] == pytest.approx(0.667, abs=0.01)
        assert any("val" in issue and "NULL" in issue for issue in metrics["issues"])

    def test_all_null_column(self):
        results = [
            {"id": 1, "empty_col": None},
            {"id": 2, "empty_col": None},
        ]
        metrics = compute_data_quality(results)
        assert metrics["null_ratios"]["empty_col"] == 1.0
        assert any("entirely NULL" in issue for issue in metrics["issues"])

    def test_negative_values(self):
        results = [
            {"id": 1, "amount": -50.0},
            {"id": 2, "amount": -30.0},
            {"id": 3, "amount": 100.0},
        ]
        metrics = compute_data_quality(results)
        assert "amount" in metrics["negative_ratios"]
        assert metrics["negative_ratios"]["amount"] == pytest.approx(0.667, abs=0.01)
        assert any("negative" in issue.lower() for issue in metrics["issues"])

    def test_negative_values_low_ratio_no_issue(self):
        results = [
            {"amount": -10.0},
            {"amount": 100.0},
            {"amount": 200.0},
            {"amount": 300.0},
        ]
        metrics = compute_data_quality(results)
        assert "amount" in metrics["negative_ratios"]
        # ratio 0.25 < 0.5, so no issue reported
        assert not any("negative" in issue.lower() for issue in metrics["issues"])

    def test_duplicate_detection(self):
        results = [
            {"id": 1, "val": "A"},
            {"id": 1, "val": "A"},
            {"id": 1, "val": "A"},
            {"id": 2, "val": "B"},
        ]
        metrics = compute_data_quality(results)
        assert metrics["duplicate_ratio"] == 0.5

    def test_high_duplicate_ratio_reports_issue(self):
        results = [
            {"id": 1, "val": "A"},
            {"id": 1, "val": "A"},
            {"id": 1, "val": "A"},
        ]
        metrics = compute_data_quality(results)
        assert metrics["duplicate_ratio"] == pytest.approx(0.667, abs=0.01)
        assert any("duplicate" in issue.lower() for issue in metrics["issues"])

    def test_no_duplicates(self):
        results = [
            {"id": 1, "val": "A"},
            {"id": 2, "val": "B"},
            {"id": 3, "val": "C"},
        ]
        metrics = compute_data_quality(results)
        assert metrics["duplicate_ratio"] == 0.0

    def test_outlier_detection(self):
        # 9 normal values + 1 extreme outlier
        results = [{"val": float(i)} for i in range(1, 10)]
        results.append({"val": 10000.0})
        metrics = compute_data_quality(results)
        assert "val" in metrics["outlier_columns"]

    def test_no_outliers_uniform_data(self):
        results = [{"val": float(i)} for i in range(10)]
        metrics = compute_data_quality(results)
        assert metrics["outlier_columns"] == []

    def test_outlier_needs_minimum_values(self):
        # Only 3 values — IQR check skipped
        results = [{"val": 1.0}, {"val": 2.0}, {"val": 1000.0}]
        metrics = compute_data_quality(results)
        assert metrics["outlier_columns"] == []

    def test_mixed_types_skip_non_numeric(self):
        results = [
            {"name": "Alice", "amount": 100.0},
            {"name": "Bob", "amount": 200.0},
        ]
        metrics = compute_data_quality(results)
        # "name" should not appear in negative_ratios or outlier_columns
        assert "name" not in metrics["negative_ratios"]
        assert "name" not in metrics["outlier_columns"]

    def test_single_row(self):
        results = [{"id": 1, "total": 500.0}]
        metrics = compute_data_quality(results)
        assert metrics["row_count"] == 1
        assert metrics["duplicate_ratio"] == 0.0


class TestFormatQualityReport:
    def test_clean_report(self):
        metrics = compute_data_quality([
            {"id": 1, "val": 10},
            {"id": 2, "val": 20},
        ])
        report = format_quality_report(metrics)
        assert "Rows returned: 2" in report
        assert "NULL" not in report
        assert "Negative" not in report
        assert "Outlier" not in report

    def test_report_with_nulls(self):
        metrics = {
            "row_count": 3,
            "null_ratios": {"col_a": 0.333, "col_b": 0.0},
            "negative_ratios": {},
            "duplicate_ratio": 0.0,
            "outlier_columns": [],
            "issues": [],
        }
        report = format_quality_report(metrics)
        assert "col_a" in report
        assert "col_b" not in report.split("NULL")[1] if "NULL" in report else True

    def test_report_with_issues(self):
        metrics = {
            "row_count": 0,
            "null_ratios": {},
            "negative_ratios": {},
            "duplicate_ratio": 0.0,
            "outlier_columns": [],
            "issues": ["Query returned 0 rows."],
        }
        report = format_quality_report(metrics)
        assert "0 rows" in report

    def test_report_with_all_findings(self):
        metrics = {
            "row_count": 10,
            "null_ratios": {"price": 0.3},
            "negative_ratios": {"qty": 0.2},
            "duplicate_ratio": 0.1,
            "outlier_columns": ["revenue"],
            "issues": ["Column 'price' has 30% NULL values."],
        }
        report = format_quality_report(metrics)
        assert "price: 30%" in report
        assert "qty: 20%" in report
        assert "revenue" in report
        assert "10%" in report  # duplicate ratio
        assert "Issues:" in report
