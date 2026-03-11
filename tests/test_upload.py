import pytest
import json
from src.data.db import Database
from src.upload import sanitize_table_name, parse_and_register, extract_text, is_tabular


@pytest.fixture
def db():
    d = Database()
    yield d
    d.close()


class TestSanitizeTableName:
    def test_basic_csv(self):
        assert sanitize_table_name("sales_data.csv") == "sales_data"

    def test_spaces_and_hyphens(self):
        assert sanitize_table_name("My Sales - Report.xlsx") == "my_sales_report"

    def test_special_chars(self):
        result = sanitize_table_name("data@2024!.json")
        assert result == "data2024"

    def test_starts_with_number(self):
        result = sanitize_table_name("2024_sales.csv")
        assert result.startswith("t_")

    def test_empty_after_sanitize(self):
        result = sanitize_table_name("!!!.csv")
        assert result.startswith("t_")

    def test_truncation(self):
        long_name = "a" * 100 + ".csv"
        result = sanitize_table_name(long_name)
        assert len(result) <= 60

    def test_amazon_sale_report(self):
        assert sanitize_table_name("Amazon Sale Report.csv") == "amazon_sale_report"


class TestIsTabular:
    def test_csv(self):
        assert is_tabular("data.csv") is True

    def test_xlsx(self):
        assert is_tabular("data.xlsx") is True

    def test_json(self):
        assert is_tabular("data.json") is True

    def test_txt(self):
        assert is_tabular("notes.txt") is False

    def test_pdf(self):
        assert is_tabular("report.pdf") is False


class TestParseAndRegister:
    def test_csv(self, db):
        csv_bytes = b"Name,Age,Score\nAlice,30,85.5\nBob,25,92.0"
        result = parse_and_register(db, csv_bytes, "students.csv")
        assert result["table_name"] == "students"
        assert result["row_count"] == 2
        assert result["col_count"] == 3
        assert "students" in result["schema_text"]

        # Verify queryable
        rows = db.execute("SELECT * FROM students")
        assert len(rows) == 2

    def test_json(self, db):
        data = [{"x": 1, "y": "hello"}, {"x": 2, "y": "world"}]
        json_bytes = json.dumps(data).encode()
        result = parse_and_register(db, json_bytes, "test_data.json")
        assert result["table_name"] == "test_data"
        assert result["row_count"] == 2

        rows = db.execute("SELECT * FROM test_data")
        assert len(rows) == 2

    def test_xlsx(self, db):
        import pandas as pd
        from io import BytesIO

        df = pd.DataFrame({"product": ["A", "B"], "price": [10, 20]})
        buf = BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        xlsx_bytes = buf.getvalue()

        result = parse_and_register(db, xlsx_bytes, "products.xlsx")
        assert result["table_name"] == "products"
        assert result["row_count"] == 2

        rows = db.execute("SELECT * FROM products")
        assert len(rows) == 2

    def test_unsupported_extension(self, db):
        with pytest.raises(ValueError, match="Not a tabular file"):
            parse_and_register(db, b"data", "file.pdf")


class TestExtractText:
    def test_txt(self):
        text = extract_text(b"Hello world", "notes.txt")
        assert text == "Hello world"

    def test_md(self):
        text = extract_text(b"# Title\nBody", "doc.md")
        assert "Title" in text

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Cannot extract text"):
            extract_text(b"data", "file.csv")
