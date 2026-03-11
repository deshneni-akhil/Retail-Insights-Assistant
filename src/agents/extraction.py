# Data extraction agent — executes SQL queries against DuckDB.
from src.data.db import Database


def execute_query(db: Database, sql: str) -> dict:
    """Execute SQL and return results with metadata.

    Returns:
        dict with keys: success, results, error, row_count
    """
    # Safety check: only allow SELECT and WITH (CTE) statements
    sql_upper = sql.strip().upper()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        return {
            "success": False,
            "results": [],
            "error": "Only SELECT queries are allowed.",
            "row_count": 0,
        }

    # Block dangerous keywords
    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "CREATE TABLE"]
    for keyword in dangerous:
        if keyword in sql_upper:
            return {
                "success": False,
                "results": [],
                "error": f"Blocked: query contains forbidden keyword '{keyword}'.",
                "row_count": 0,
            }

    try:
        results = db.execute(sql)
        return {
            "success": True,
            "results": results,
            "error": "",
            "row_count": len(results),
        }
    except RuntimeError as e:
        return {
            "success": False,
            "results": [],
            "error": str(e),
            "row_count": 0,
        }
