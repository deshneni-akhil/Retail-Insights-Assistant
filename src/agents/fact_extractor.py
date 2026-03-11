# Fact extractor — pulls verifiable facts from query results for anti-hallucination grounding.

_MAX_FACT_ROWS = 20


def _format_value(value) -> str | None:
    """Format a single value for the facts string. Returns None for skippable values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{value}'"


def extract_facts(query_result: dict) -> str:
    """Extract verifiable facts from query results.

    Pure Python — no LLM call. Returns a formatted string of facts
    that constrains the response LLM to only cite real numbers.

    Returns empty string if results are empty or query failed.
    """
    if not query_result.get("success") or not query_result.get("results"):
        return ""

    results = query_result["results"]
    total_rows = len(results)
    display_rows = results[:_MAX_FACT_ROWS]

    lines = []

    if total_rows == 1:
        # Single-row aggregation — compact list
        lines.append("VERIFIED FACTS FROM DATA:")
        row = display_rows[0]
        for col, val in row.items():
            formatted = _format_value(val)
            if formatted is not None:
                lines.append(f"  - {col} = {formatted}")
    else:
        # Multi-row results
        truncated = total_rows > _MAX_FACT_ROWS
        header = f"VERIFIED FACTS FROM DATA ({total_rows} rows"
        if truncated:
            header += f", showing first {_MAX_FACT_ROWS}"
        header += "):"
        lines.append(header)

        for i, row in enumerate(display_rows, 1):
            parts = []
            for col, val in row.items():
                formatted = _format_value(val)
                if formatted is not None:
                    parts.append(f"{col} = {formatted}")
            if parts:
                lines.append(f"  Row {i}: {', '.join(parts)}")

        # Numeric column aggregates
        numeric_cols = {}
        for row in results:
            for col, val in row.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if col not in numeric_cols:
                        numeric_cols[col] = []
                    numeric_cols[col].append(val)

        if numeric_cols:
            lines.append("  Numeric summaries:")
            for col, values in numeric_cols.items():
                col_min = min(values)
                col_max = max(values)
                col_total = sum(values)
                lines.append(
                    f"    {col}: min = {col_min}, max = {col_max}, total = {col_total}"
                )

    return "\n".join(lines)
