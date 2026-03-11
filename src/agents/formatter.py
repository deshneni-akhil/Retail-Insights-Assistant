# Formatter agent — converts raw LLM responses into structured markdown for Streamlit display.
from src.config import LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, get_llm_client


def load_prompt() -> str:
    return (PROMPTS_DIR / "formatter.txt").read_text()


def format_response(
    question: str,
    raw_response: str,
    intent: str = "question",
) -> str:
    """Reformat a raw agent response into clean structured markdown.

    Args:
        question: The original user question.
        raw_response: The raw text response from response_node or summary_node.
        intent: "question" or "summary" — controls formatting style.

    Returns:
        Structured markdown string ready for st.markdown().
    """
    if not raw_response or not raw_response.strip():
        return raw_response

    # If the response is already well-structured markdown (has headers + tables),
    # skip the extra LLM call to save cost and latency.
    lines = raw_response.strip().splitlines()
    has_headers = any(line.startswith("#") for line in lines)
    has_table = any(line.strip().startswith("|") for line in lines)
    has_bullets = sum(1 for line in lines if line.strip().startswith(("-", "*", "•"))) >= 3

    if has_headers and (has_table or has_bullets):
        return raw_response

    prompt_template = load_prompt()
    prompt = prompt_template.format(
        question=question,
        intent=intent,
        raw_response=raw_response,
    )

    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )

    formatted = response.choices[0].message.content.strip()

    # Strip accidental code fences if LLM wraps the output
    if formatted.startswith("```markdown"):
        formatted = formatted.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    elif formatted.startswith("```"):
        formatted = formatted.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return formatted if formatted else raw_response
