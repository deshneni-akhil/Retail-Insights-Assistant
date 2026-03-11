# Summary agent — generates executive summaries via LLM-driven dynamic aggregation.
import json
from src.config import LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, get_llm_client
from src.data.db import Database
from src.data.vectorstore import SessionVectorStore


def load_prompt() -> str:
    return (PROMPTS_DIR / "summary.txt").read_text()


def _plan_aggregations(schema_text: str) -> list[dict]:
    """Ask the LLM to plan aggregation queries based on the available schema.

    Sends a compact schema (table + column names only, no sample values) to save tokens.
    """
    # Strip sample values block from each table — only table names + column names needed for planning
    compact_lines = []
    for line in schema_text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith("  Sample values:"):
            continue
        # Skip indent-4 sample value lines (they start with 4 spaces followed by "- colname: [")
        if stripped.startswith("    - ") and ": [" in stripped:
            continue
        compact_lines.append(stripped)
    compact_schema = "\n".join(compact_lines)

    plan_prompt = (
        "Given the following database schema, generate 3-5 DuckDB SQL queries that would "
        "provide useful aggregated data for an executive business summary.\n\n"
        f"SCHEMA:\n{compact_schema}\n\n"
        "Rules:\n"
        "- Only SELECT queries\n"
        "- Focus on: totals, group-by distributions, top-N rankings, time trends (if date columns exist)\n"
        "- Use ROUND() for numeric results, LIMIT 10 for group-by queries\n"
        "- Each query should target the most important business metrics\n\n"
        'Respond with ONLY a JSON array of objects like: [{"name": "descriptive_name", "sql": "SELECT ..."}]\n'
        "No explanations, no markdown, just the JSON array."
    )

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=LLM_TEMPERATURE,
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def _fallback_aggregations(db: Database) -> dict:
    """Deterministic fallback: basic stats for each table."""
    data = {}
    for table in db.list_tables():
        try:
            count = db.execute(f'SELECT COUNT(*) as row_count FROM "{table}"')
            data[f"{table}_count"] = count
        except Exception:
            pass

        # Get numeric column stats
        col_info = db.get_table_info(table)
        for col in col_info:
            col_name = col.get("column_name", col.get("name", ""))
            col_type = col.get("column_type", col.get("type", "")).upper()
            if any(t in col_type for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT"]):
                try:
                    stats = db.execute(
                        f'SELECT ROUND(MIN("{col_name}"), 2) as min_val, '
                        f'ROUND(MAX("{col_name}"), 2) as max_val, '
                        f'ROUND(AVG("{col_name}"), 2) as avg_val, '
                        f'ROUND(SUM("{col_name}"), 2) as sum_val '
                        f'FROM "{table}" WHERE "{col_name}" IS NOT NULL'
                    )
                    data[f"{table}_{col_name}_stats"] = stats
                except Exception:
                    pass
    return data


def generate_summary(
    db: Database,
    dynamic_schema: str,
    vectorstore: SessionVectorStore | None = None,
    past_context: str = "",
) -> str:
    """Generate an executive summary using LLM-planned aggregation queries."""
    # Phase 1: Plan aggregation queries
    planned = _plan_aggregations(dynamic_schema)

    # Phase 2: Execute planned queries (or fallback)
    aggregated_data = {}
    if planned:
        for item in planned:
            name = item.get("name", "unknown")
            sql = item.get("sql", "")
            if not sql:
                continue
            try:
                # Validate before executing
                error = db.explain(sql)
                if error:
                    continue
                results = db.execute(sql)
                aggregated_data[name] = results
            except Exception:
                pass

    # If LLM planning produced nothing usable, fall back to deterministic stats
    if not aggregated_data:
        aggregated_data = _fallback_aggregations(db)

    # Phase 3: Summarize
    data_str = json.dumps(aggregated_data, separators=(",", ":"), default=str)
    prompt = load_prompt().format(schema=dynamic_schema, data=data_str)

    if past_context:
        prompt += (
            "\n\nFor reference, here are previous summaries. "
            "Briefly note any changes or trends compared to prior summaries:\n"
            f"{past_context}"
        )

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    summary_text = response.choices[0].message.content.strip()

    # Store in vector DB for future retrieval
    if vectorstore:
        try:
            tables_used = ",".join(db.list_tables())
            vectorstore.add_summary(summary_text, metadata={"tables_used": tables_used})
        except Exception:
            pass

    return summary_text
