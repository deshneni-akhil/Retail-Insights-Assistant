# Application configuration — loads env vars and defines constants.
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Azure AI Foundry / OpenAI-compatible endpoint
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = Path(__file__).parent / "prompts"
SESSIONS_DIR = DATA_DIR / "sessions"

# Auto-create data directories on fresh installs
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 5
SQL_ROW_LIMIT = 100
RAG_SIMILARITY_THRESHOLD = 0.3  # Minimum relevance score to include RAG results
CACHE_HIT_THRESHOLD = 0.15  # Max ChromaDB distance for Q&A cache hit (tighter than RAG)

# Schema selection thresholds
SCHEMA_TABLE_THRESHOLD = 5  # Trigger table selection when > this many tables
SCHEMA_TOP_K_TABLES = 3  # Number of tables to select via semantic search


def get_llm_client():
    """Get an OpenAI-compatible client pointed at Azure AI Foundry.

    Automatically traced by LangSmith when LANGCHAIN_TRACING_V2=true.
    """
    from openai import OpenAI
    from langsmith.wrappers import wrap_openai
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        max_retries=MAX_RETRIES,
    )
    return wrap_openai(client)
