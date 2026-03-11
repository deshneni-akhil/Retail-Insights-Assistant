# File upload handler — parses CSV/Excel/JSON/PDF/text and registers in DuckDB or RAG.
import re
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.data.db import Database


TABULAR_EXTENSIONS = {"csv", "xlsx", "xls", "json"}
TEXT_EXTENSIONS = {"txt", "md", "pdf"}


def get_extension(filename: str) -> str:
    """Extract lowercase file extension without dot."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def is_tabular(filename: str) -> bool:
    """Returns True for CSV/Excel/JSON files that go into DuckDB."""
    return get_extension(filename) in TABULAR_EXTENSIONS


def sanitize_table_name(filename: str) -> str:
    """Convert a filename into a safe SQL table name.

    Example: 'Amazon Sale Report.csv' → 'amazon_sale_report'
    """
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    name = name.lower().strip()
    name = re.sub(r"[\s\-\.]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name or not name[0].isalpha():
        name = "t_" + name
    return name[:60]


def parse_and_register(db: Database, file_bytes: bytes, filename: str) -> dict:
    """Parse a tabular file and register it as a DuckDB table.

    Returns:
        dict with keys: table_name, row_count, col_count, schema_text
    """
    ext = get_extension(filename)
    table_name = sanitize_table_name(filename)

    if ext == "csv":
        # Write to temp file so DuckDB can use read_csv_auto
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            row_count = db.ingest_file(table_name, tmp_path, "csv")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    elif ext in ("xlsx", "xls"):
        # DuckDB can't read Excel natively — use pandas + openpyxl
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
        df = _normalize_columns(df)
        df = _drop_junk_columns(df)
        row_count = db.ingest_dataframe(table_name, df)

    elif ext == "json":
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            row_count = db.ingest_file(table_name, tmp_path, "json")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    else:
        raise ValueError(f"Not a tabular file: .{ext}")

    schema_text = db.get_schema_text(table_name)
    col_count = len(db.get_table_info(table_name))

    return {
        "table_name": table_name,
        "row_count": row_count,
        "col_count": col_count,
        "schema_text": schema_text,
    }


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text content from PDF, TXT, or MD files for RAG ingestion."""
    ext = get_extension(filename)

    if ext == "pdf":
        return _extract_pdf_text(file_bytes)
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Cannot extract text from .{ext} files")


def _extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n\n".join(pages)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: lowercase, underscores, strip whitespace."""
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def _drop_junk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed/index columns from CSV/Excel exports."""
    drop_cols = [c for c in df.columns if c.startswith("unnamed") or c == "index"]
    return df.drop(columns=drop_cols, errors="ignore")
