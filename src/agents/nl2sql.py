# NL-to-SQL agent — converts natural language questions to DuckDB SQL.
from src.config import LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, SQL_ROW_LIMIT, get_llm_client


def load_prompt() -> str:
    return (PROMPTS_DIR / "nl2sql.txt").read_text()


def generate_sql(
    question: str,
    schema_text: str,
    conversation_context: str = "",
    validation_feedback: str = "",
) -> str:
    """Generate a DuckDB SQL query from a natural language question.

    Args:
        question: The user's natural language question.
        schema_text: Dynamic schema text (auto-generated from DuckDB DESCRIBE).
        conversation_context: Recent conversation history.
        validation_feedback: Feedback from a failed validation attempt.
    """
    prompt_template = load_prompt()

    context_block = ""
    if conversation_context:
        context_block += f"\nConversation context:\n{conversation_context}\n"
    if validation_feedback:
        context_block += (
            f"\nPrevious attempt failed validation: {validation_feedback}\n"
            "Please fix the SQL query based on this feedback.\n"
        )

    prompt = prompt_template.format(
        schema=schema_text,
        row_limit=SQL_ROW_LIMIT,
        question=question,
        conversation_context=context_block,
    )

    system_prompt = (PROMPTS_DIR / "system.txt").read_text()

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=LLM_TEMPERATURE,
    )

    sql = response.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return sql
