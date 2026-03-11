# LangGraph state graph — orchestrates multi-agent flow with RAG retrieval.
import json
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from src.config import LLM_MODEL, MAX_RETRIES, CACHE_HIT_THRESHOLD, PROMPTS_DIR, get_llm_client 
from src.data.db import Database
from src.data.vectorstore import SessionVectorStore
from src.agents.router import classify_intent
from src.agents.nl2sql import generate_sql
from src.agents.extraction import execute_query
from src.agents.validation import validate_result
from src.agents.summary import generate_summary
from src.agents.retrieval import retrieve_context, retrieve_similar_qa
from src.agents.formatter import format_response
from src.agents.fact_extractor import extract_facts
from src.agents.preprocess import sanitize_input, check_cache


class AgentState(TypedDict):
    user_query: str
    intent: str
    dynamic_schema: str
    generated_sql: str
    query_result: dict
    validation: dict
    retry_count: int
    final_response: str
    conversation_history: list[dict]
    retrieved_context: str
    verified_facts: str
    cache_hit: bool
    error: str


def build_graph(db: Database, vectorstore: SessionVectorStore) -> StateGraph:
    """Build the LangGraph state graph wired to the given database and vectorstore."""

    def preprocess_node(state: AgentState) -> dict:
        error = sanitize_input(state["user_query"])
        if error:
            return {"final_response": error, "cache_hit": True}
        cached = check_cache(vectorstore, state["user_query"], CACHE_HIT_THRESHOLD)
        if cached:
            return {
                "final_response": cached["answer"],
                "generated_sql": cached["sql"],
                "cache_hit": True,
                "intent": "question",
            }
        return {"cache_hit": False}

    def router_node(state: AgentState) -> dict:
        intent = classify_intent(state["user_query"])
        return {"intent": intent}

    def summary_node(state: AgentState) -> dict:
        past_context = retrieve_context(vectorstore, state["user_query"], top_k=2)
        response = generate_summary(
            db,
            dynamic_schema=state["dynamic_schema"],
            vectorstore=vectorstore,
            past_context=past_context,
        )
        return {"final_response": response, "retrieved_context": past_context, "intent": "summary"}

    def clarification_node(state: AgentState) -> dict:
        tables = db.list_tables()
        if tables:
            table_list = ", ".join(tables)
            msg = (
                "I'm a Retail Insights Assistant. I can help you with:\n\n"
                "- **Summarize** data performance (try: \"Give me an executive summary\")\n"
                "- **Answer questions** about your data (try: \"Which category had the highest revenue?\")\n\n"
                f"**Available tables:** {table_list}\n\n"
                "You can also upload more files (CSV, Excel, JSON, PDF, text) to expand the analysis.\n\n"
                "What would you like to know?"
            )
        else:
            msg = (
                "I'm a Retail Insights Assistant. Upload data files (CSV, Excel, JSON) "
                "to get started, then ask me questions about your data or request a summary."
            )
        return {"final_response": msg}

    def nl2sql_node(state: AgentState) -> dict:
        context = ""
        if state.get("conversation_history"):
            recent = state["conversation_history"][-2:]  # last turn only
            context = "\n".join(
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
                for h in recent
            )

        feedback = ""
        if state.get("validation") and not state["validation"].get("valid", True):
            validation_feedback = state["validation"].get("feedback", "")
            failed_sql = state.get("generated_sql", "")
            if failed_sql:
                feedback = (
                    f"Previous SQL that failed:\n{failed_sql}\n\n"
                    f"Error: {validation_feedback}\n"
                    "Fix the SQL based on this error. Do not repeat the same mistake."
                )
            else:
                feedback = validation_feedback

        sql = generate_sql(
            question=state["user_query"],
            schema_text=state["dynamic_schema"],
            conversation_context=context,
            validation_feedback=feedback,
        )
        return {"generated_sql": sql}

    def extraction_node(state: AgentState) -> dict:
        result = execute_query(db, state["generated_sql"])
        return {"query_result": result}

    def validation_node(state: AgentState) -> dict:
        validation = validate_result(
            question=state["user_query"],
            sql=state["generated_sql"],
            result=state["query_result"],
        )
        return {"validation": validation}

    def fact_extractor_node(state: AgentState) -> dict:
        facts = extract_facts(state["query_result"])
        return {"verified_facts": facts}

    def retrieval_node(state: AgentState) -> dict:
        context = retrieve_context(vectorstore, state["user_query"], top_k=3)
        return {"retrieved_context": context}

    def response_node(state: AgentState) -> dict:
        rag_context = state.get("retrieved_context", "")
        if rag_context:
            prompt_template = (PROMPTS_DIR / "rag_response.txt").read_text()
        else:
            prompt_template = (PROMPTS_DIR / "response.txt").read_text()

        conversation_context = ""
        if state.get("conversation_history"):
            recent = state["conversation_history"][-1:]  # last turn only
            conversation_context = "Previous turn:\n" + "\n".join(
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
                for h in recent
            )

        display_results = state["query_result"]["results"][:10]

        format_kwargs = {
            "question": state["user_query"],
            "result": json.dumps(display_results, separators=(",", ":"), default=str),
            "verified_facts": state.get("verified_facts", ""),
            "conversation_context": conversation_context,
        }
        if rag_context:
            format_kwargs["retrieved_context"] = rag_context

        prompt = prompt_template.format(**format_kwargs)

        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()

        # Store successful Q&A pair for future fallback retrieval
        try:
            vectorstore.add_qa_pair(
                question=state["user_query"],
                sql=state["generated_sql"],
                answer=answer,
            )
        except Exception:
            pass

        return {"final_response": answer, "intent": "question"}

    def fallback_node(state: AgentState) -> dict:
        similar_qa = retrieve_similar_qa(vectorstore, state["user_query"], top_k=2)

        error_msg = state.get("error", "")
        validation_msg = ""
        if state.get("validation"):
            validation_msg = state["validation"].get("feedback", "")

        if similar_qa:
            return {
                "final_response": (
                    "I wasn't able to answer that question directly, but here are similar questions "
                    "I've answered before that might help:\n\n"
                    f"{similar_qa}\n\n"
                    "Try rephrasing your question or ask something more specific."
                ),
                "retrieved_context": similar_qa,
            }

        return {
            "final_response": (
                "I wasn't able to confidently answer that question after multiple attempts. "
                f"Here's what went wrong: {validation_msg or error_msg}\n\n"
                "Try rephrasing your question or ask something more specific about your data."
            )
        }

    def formatter_node(state: AgentState) -> dict:
        formatted = format_response(
            question=state["user_query"],
            raw_response=state["final_response"],
            intent=state.get("intent", "question"),
        )
        return {"final_response": formatted}

    # Routing functions
    def route_intent(state: AgentState) -> Literal["summary", "question", "clarification"]:
        return state["intent"]

    def route_validation(state: AgentState) -> Literal["pass", "retry", "fallback"]:
        validation = state.get("validation", {})
        if validation.get("valid", False):
            return "pass"
        retry_count = state.get("retry_count", 0)
        if retry_count < MAX_RETRIES:
            return "retry"
        return "fallback"

    def increment_retry(state: AgentState) -> dict:
        return {"retry_count": state.get("retry_count", 0) + 1}

    def route_preprocess(state: AgentState) -> Literal["cache_hit", "cache_miss"]:
        if state.get("cache_hit", False):
            return "cache_hit"
        return "cache_miss"

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("preprocess", preprocess_node)
    graph.add_node("router", router_node)
    graph.add_node("summary", summary_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("validation", validation_node)
    graph.add_node("fact_extractor", fact_extractor_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("response", response_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("formatter", formatter_node)

    graph.set_entry_point("preprocess")

    graph.add_conditional_edges(
        "preprocess",
        route_preprocess,
        {
            "cache_hit": "formatter",
            "cache_miss": "router",
        },
    )

    graph.add_conditional_edges(
        "router",
        route_intent,
        {
            "summary": "summary",
            "question": "nl2sql",
            "clarification": "clarification",
        },
    )

    graph.add_edge("summary", "formatter")
    graph.add_edge("clarification", END)

    graph.add_edge("nl2sql", "extraction")
    graph.add_edge("extraction", "validation")

    graph.add_conditional_edges(
        "validation",
        route_validation,
        {
            "pass": "fact_extractor",
            "retry": "increment_retry",
            "fallback": "fallback",
        },
    )

    graph.add_edge("fact_extractor", "retrieval")
    graph.add_edge("retrieval", "response")
    graph.add_edge("increment_retry", "nl2sql")

    graph.add_edge("response", "formatter")
    graph.add_edge("formatter", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def run_query(
    compiled_graph,
    user_query: str,
    conversation_history: list[dict] | None = None,
    dynamic_schema: str = "",
) -> AgentState:
    """Run a user query through the agent graph."""
    initial_state: AgentState = {
        "user_query": user_query,
        "intent": "",
        "dynamic_schema": dynamic_schema,
        "generated_sql": "",
        "query_result": {},
        "validation": {},
        "retry_count": 0,
        "final_response": "",
        "conversation_history": conversation_history or [],
        "retrieved_context": "",
        "verified_facts": "",
        "cache_hit": False,
        "error": "",
    }

    result = compiled_graph.invoke(initial_state)
    return result
