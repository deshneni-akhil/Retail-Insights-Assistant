# Validation agent — statistical pre-checks + LLM semantic validation.
import json
import math
from src.config import LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, get_llm_client


def load_prompt() -> str:
    return (PROMPTS_DIR / "validation.txt").read_text()


def compute_data_quality(results: list[dict]) -> dict:
    """Run statistical checks on query results. No LLM call needed.

    Returns a dict with:
        - row_count: int
        - null_ratios: {col: float} — fraction of nulls per column
        - negative_ratios: {col: float} — fraction of negative values per numeric column
        - duplicate_ratio: float — fraction of duplicate rows
        - outlier_columns: [col] — columns with outliers detected via IQR
        - issues: [str] — human-readable issue descriptions
    """
    metrics = {
        "row_count": len(results),
        "null_ratios": {},
        "negative_ratios": {},
        "duplicate_ratio": 0.0,
        "outlier_columns": [],
        "issues": [],
    }

    if not results:
        metrics["issues"].append("Query returned 0 rows.")
        return metrics

    columns = list(results[0].keys())
    n = len(results)

    # --- Null analysis ---
    for col in columns:
        null_count = sum(1 for row in results if row.get(col) is None)
        ratio = null_count / n
        metrics["null_ratios"][col] = round(ratio, 3)
        if ratio == 1.0:
            metrics["issues"].append(f"Column '{col}' is entirely NULL.")
        elif ratio > 0.5:
            metrics["issues"].append(
                f"Column '{col}' has {ratio:.0%} NULL values."
            )

    # --- Numeric column analysis (negatives + outliers) ---
    numeric_cols = {}
    for col in columns:
        values = []
        for row in results:
            v = row.get(col)
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv) and not math.isinf(fv):
                        values.append(fv)
                except (ValueError, TypeError):
                    break
        else:
            if values:
                numeric_cols[col] = values

    for col, values in numeric_cols.items():
        # Negative value check
        neg_count = sum(1 for v in values if v < 0)
        if neg_count > 0:
            ratio = neg_count / len(values)
            metrics["negative_ratios"][col] = round(ratio, 3)
            if ratio > 0.5:
                metrics["issues"].append(
                    f"Column '{col}' has {ratio:.0%} negative values."
                )

        # Outlier detection via IQR (only meaningful with 4+ values)
        if len(values) >= 4:
            sorted_vals = sorted(values)
            q1_idx = len(sorted_vals) // 4
            q3_idx = 3 * len(sorted_vals) // 4
            q1 = sorted_vals[q1_idx]
            q3 = sorted_vals[q3_idx]
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = sum(1 for v in values if v < lower or v > upper)
                if outlier_count > 0:
                    metrics["outlier_columns"].append(col)

    # --- Duplicate detection ---
    seen = set()
    dup_count = 0
    for row in results:
        key = tuple(sorted(row.items()))
        if key in seen:
            dup_count += 1
        seen.add(key)
    if n > 0:
        metrics["duplicate_ratio"] = round(dup_count / n, 3)
        if dup_count > 0 and dup_count / n > 0.5:
            metrics["issues"].append(
                f"{dup_count}/{n} rows are duplicates ({dup_count/n:.0%})."
            )

    return metrics


def format_quality_report(metrics: dict) -> str:
    """Format data quality metrics into a concise text report for the LLM."""
    lines = [f"Rows returned: {metrics['row_count']}"]

    # Null summary — only mention columns with nulls
    null_cols = {c: r for c, r in metrics["null_ratios"].items() if r > 0}
    if null_cols:
        parts = [f"{c}: {r:.0%}" for c, r in null_cols.items()]
        lines.append(f"NULL ratios: {', '.join(parts)}")

    # Negative summary
    if metrics["negative_ratios"]:
        parts = [f"{c}: {r:.0%}" for c, r in metrics["negative_ratios"].items()]
        lines.append(f"Negative value ratios: {', '.join(parts)}")

    # Outliers
    if metrics["outlier_columns"]:
        lines.append(f"Outliers detected (IQR) in: {', '.join(metrics['outlier_columns'])}")

    # Duplicates
    if metrics["duplicate_ratio"] > 0:
        lines.append(f"Duplicate row ratio: {metrics['duplicate_ratio']:.0%}")

    # Issues
    if metrics["issues"]:
        lines.append("Issues: " + "; ".join(metrics["issues"]))

    return "\n".join(lines)


def validate_result(question: str, sql: str, result: dict) -> dict:
    """Two-phase validation: statistical pre-checks then LLM semantic check.

    Returns:
        dict with keys: valid (bool), confidence (float), feedback (str),
                        data_quality (dict)
    """
    if not result["success"]:
        return {
            "valid": False,
            "confidence": 0.0,
            "feedback": f"SQL execution failed: {result['error']}",
            "data_quality": {},
        }

    # Phase 1: Statistical pre-checks (run on up to 20 rows for accuracy)
    sample_for_stats = result["results"][:20]
    metrics = compute_data_quality(sample_for_stats)
    quality_report = format_quality_report(metrics)

    # Phase 2: LLM semantic validation — only send 5 rows (enough to judge correctness)
    display_results = result["results"][:5]
    prompt_template = load_prompt()
    prompt = prompt_template.format(
        question=question,
        sql=sql,
        result=json.dumps(display_results, separators=(",", ":"), default=str),
        quality_report=quality_report,
    )

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )

    try:
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        return {
            "valid": parsed.get("valid", False),
            "confidence": float(parsed.get("confidence", 0.0)),
            "feedback": parsed.get("feedback", ""),
            "data_quality": metrics,
        }
    except (json.JSONDecodeError, AttributeError, ValueError, IndexError):
        return {
            "valid": True,
            "confidence": 0.5,
            "feedback": "Validation agent response could not be parsed. Passing through.",
            "data_quality": metrics,
        }
