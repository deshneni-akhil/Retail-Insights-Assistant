# Router agent — classifies user intent as summary, question, or clarification.
import json
from src.config import LLM_MODEL, LLM_TEMPERATURE, PROMPTS_DIR, get_llm_client


def load_prompt() -> str:
    return (PROMPTS_DIR / "router.txt").read_text()


def classify_intent(query: str) -> str:
    """Classify user query intent. Returns 'summary', 'question', or 'clarification'."""
    prompt = load_prompt().format(query=query)
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
        intent = parsed.get("intent", "clarification")
        if intent not in ("summary", "question", "clarification"):
            return "clarification"
        return intent
    except (json.JSONDecodeError, AttributeError, IndexError):
        return "clarification"
