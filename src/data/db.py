# DuckDB connection management — file-backed or in-memory, with native file ingestion.
import duckdb
import pandas as pd


class Database:
    """Manages a DuckDB database with dynamic table registration and schema inference."""

    def __init__(self, db_path: str | None = None):
        """Create a DuckDB connection.

        Args:
            db_path: Path to a persistent .duckdb file. None for in-memory (tests).
        """
        self.conn = duckdb.connect(db_path or ":memory:")

    def ingest_file(self, table_name: str, file_path: str, file_type: str) -> int:
        """Ingest a file using DuckDB's native readers. Returns row count."""
        readers = {"csv": "read_csv_auto", "json": "read_json_auto"}
        reader = readers.get(file_type)
        if not reader:
            raise ValueError(f"Unsupported file type for native ingestion: {file_type}")

        safe_path = file_path.replace("'", "''")
        self.conn.execute(
            f"CREATE OR REPLACE TABLE \"{table_name}\" AS "
            f"SELECT * FROM {reader}('{safe_path}')"
        )
        count = self.conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        return count

    def ingest_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        """Ingest a pandas DataFrame into a persistent DuckDB table."""
        self.conn.register("_temp_ingest", df)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _temp_ingest'
        )
        self.conn.unregister("_temp_ingest")
        return len(df)

    def list_tables(self) -> list[str]:
        """Return names of all tables in the database."""
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        return [r[0] for r in rows]

    def get_table_info(self, table_name: str) -> list[dict]:
        """Get column names and types for a table."""
        try:
            result = self.conn.execute(f'DESCRIBE "{table_name}"')
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception:
            return []

    def get_schema_text(self, table_name: str) -> str:
        """Generate human-readable schema text for one table."""
        row_count = self.conn.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]

        lines = [
            f'Table: {table_name} ({row_count:,} rows)',
            "  Columns:",
        ]

        col_info = self.get_table_info(table_name)
        for col in col_info:
            col_name = col.get("column_name", col.get("name", ""))
            col_type = col.get("column_type", col.get("type", ""))
            lines.append(f"    - {col_name} ({col_type})")

        # Sample distinct values for string/varchar columns
        sample_lines = []
        for col in col_info:
            col_name = col.get("column_name", col.get("name", ""))
            col_type = col.get("column_type", col.get("type", "")).upper()
            if "VARCHAR" in col_type or "TEXT" in col_type:
                try:
                    samples = self.conn.execute(
                        f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL LIMIT 3'
                    ).fetchall()
                    vals = [str(r[0]) for r in samples if r[0] is not None]
                    if vals:
                        sample_lines.append(f"    - {col_name}: {vals}")
                except Exception:
                    pass

        if sample_lines:
            lines.append("  Sample values:")
            lines.extend(sample_lines)

        lines.append("")
        return "\n".join(lines)

    def get_compact_description(self, table_name: str) -> str:
        """Generate a compact table description for ChromaDB indexing.

        Returns a short string with table name, row count, and column names.
        Optimized for embedding search, not for LLM prompts.
        """
        row_count = self.conn.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]

        col_info = self.get_table_info(table_name)
        col_names = [col.get("column_name", col.get("name", "")) for col in col_info]

        return (
            f"Table: {table_name} ({row_count:,} rows, {len(col_names)} columns). "
            f"Columns: {', '.join(col_names)}"
        )

    def get_table_directory(self) -> str:
        """Generate a compact table directory — one line per table.

        Used as a lightweight header so the LLM always knows what tables exist,
        even when full schemas are only shown for selected tables.
        """
        tables = self.list_tables()
        if not tables:
            return "No tables available."

        lines = ["TABLE DIRECTORY (all available tables):"]
        for table in tables:
            row_count = self.conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            col_count = len(self.get_table_info(table))
            lines.append(f"  - {table}: {row_count:,} rows, {col_count} columns")
        return "\n".join(lines)

    def get_all_schemas_text(self) -> str:
        """Generate schema text for ALL tables — injected into agent prompts."""
        tables = self.list_tables()
        if not tables:
            return "No tables available. Upload data files to get started."
        return "\n".join(self.get_schema_text(t) for t in tables)

    def execute(self, sql: str) -> list[dict]:
        """Execute SQL and return results as list of dicts."""
        try:
            result = self.conn.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            raise RuntimeError(f"SQL execution error: {e}") from e

    def explain(self, sql: str) -> str:
        """Validate SQL by running EXPLAIN. Returns error string or empty on success."""
        try:
            self.conn.execute(f"EXPLAIN {sql}")
            return ""
        except Exception as e:
            return str(e)

    def drop_table(self, table_name: str) -> None:
        """Drop a table if it exists."""
        self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    def close(self) -> None:
        self.conn.close()
