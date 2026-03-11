# Streamlit UI for the Retail Insights Assistant — session-based with dynamic file uploads.
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

from src.config import SESSIONS_DIR
from src.session import SessionManager
from src.upload import parse_and_register, extract_text, is_tabular
from src.agents.graph import build_graph, run_query


def _get_session_mgr() -> SessionManager:
    """Get or create the SessionManager singleton in session state."""
    if "session_mgr" not in st.session_state:
        st.session_state.session_mgr = SessionManager(SESSIONS_DIR)
    return st.session_state.session_mgr


def _rebuild_graph(sm: SessionManager) -> None:
    """Rebuild the agent graph with the current session's DB and vectorstore."""
    if sm.db and sm.vectorstore:
        st.session_state.graph = build_graph(sm.db, sm.vectorstore)
        st.session_state.dynamic_schema = sm.get_dynamic_schema()


def main():
    st.set_page_config(
        page_title="Retail Insights Assistant",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Retail Insights Assistant")
    st.caption("Upload data files, then ask questions or request summaries")

    sm = _get_session_mgr()

    # --- Sidebar ---
    with st.sidebar:
        # Session management
        st.header("Sessions")

        sessions = sm.list_sessions()
        session_names = [s["name"] for s in sessions]

        # Create new session
        with st.expander("Create New Session", expanded=not sessions):
            new_name = st.text_input("Session name", key="new_session_name")
            if st.button("Create", key="create_session") and new_name.strip():
                try:
                    sm.create_session(new_name.strip())
                    _rebuild_graph(sm)
                    st.session_state.messages = []
                    st.session_state.conversation_history = []
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        # Select existing session
        if session_names:
            current_idx = 0
            if sm.active_session and sm.active_session in session_names:
                current_idx = session_names.index(sm.active_session)

            selected = st.selectbox(
                "Active session",
                session_names,
                index=current_idx,
                key="session_select",
            )

            if selected != sm.active_session:
                sm.load_session(selected)
                _rebuild_graph(sm)
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

            # Delete session
            if st.button("Delete Session", key="delete_session"):
                sm.delete_session(selected)
                st.session_state.pop("graph", None)
                st.session_state.pop("dynamic_schema", None)
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

        st.divider()

        # File upload (only when session is active)
        if sm.active_session:
            st.header("Upload Files")
            st.caption("CSV, Excel, JSON → queryable tables | PDF, TXT → knowledge base")

            uploaded_files = st.file_uploader(
                "Drop files here",
                type=["csv", "xlsx", "xls", "json", "pdf", "txt", "md"],
                accept_multiple_files=True,
                key="file_upload",
            )

            if uploaded_files:
                for f in uploaded_files:
                    file_key = f"uploaded_{sm.active_session}_{f.name}"
                    if file_key in st.session_state:
                        continue  # Already processed this file

                    file_bytes = f.read()

                    if is_tabular(f.name):
                        try:
                            result = parse_and_register(sm.db, file_bytes, f.name)
                            sm.update_meta(
                                result["table_name"], f.name, result["row_count"]
                            )
                            # Index in ChromaDB for semantic schema selection
                            if sm.vectorstore:
                                description = sm.db.get_compact_description(result["table_name"])
                                sm.vectorstore.index_table_schema(result["table_name"], description)
                            _rebuild_graph(sm)
                            st.session_state[file_key] = True
                            st.success(
                                f"✓ **{result['table_name']}** — "
                                f"{result['row_count']:,} rows, {result['col_count']} columns"
                            )
                        except Exception as e:
                            st.error(f"Failed to load {f.name}: {e}")
                    else:
                        try:
                            text = extract_text(file_bytes, f.name)
                            if text.strip():
                                chunks = sm.vectorstore.add_document(
                                    text, source=f.name
                                )
                                st.session_state[file_key] = True
                                st.success(
                                    f"✓ **{f.name}** — {chunks} chunks stored in knowledge base"
                                )
                        except Exception as e:
                            st.error(f"Failed to process {f.name}: {e}")

            # Show uploaded tables
            tables = sm.db.list_tables() if sm.db else []
            if tables:
                st.divider()
                st.subheader("Tables")
                for t in tables:
                    try:
                        count = sm.db.execute(f'SELECT COUNT(*) as n FROM "{t}"')[0]["n"]
                        st.caption(f"**{t}** — {count:,} rows")
                    except Exception:
                        st.caption(f"**{t}**")

            # Knowledge base stats
            if sm.vectorstore:
                counts = sm.vectorstore.get_collection_counts()
                total = sum(counts.values())
                if total > 0:
                    st.divider()
                    st.caption(
                        f"Knowledge base: {counts.get('summaries', 0)} summaries, "
                        f"{counts.get('documents', 0)} doc chunks, "
                        f"{counts.get('qa_pairs', 0)} Q&A pairs"
                    )

            st.divider()
            st.header("Example Questions")
            examples = [
                "Give me an executive summary",
                "How many rows are in each table?",
                "What are the top 5 categories by revenue?",
                "Show monthly trends",
                "What is the average order value?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.pending_query = ex

            st.divider()
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()

    # --- Main area ---

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # No active session
    if not sm.active_session:
        st.info(
            "**Get started:** Create a new session in the sidebar, "
            "then upload your data files (CSV, Excel, JSON, PDF, or text)."
        )
        return

    # Session active but no tables
    tables = sm.db.list_tables() if sm.db else []
    if not tables:
        st.info(
            f"**Session '{sm.active_session}' is active.** "
            "Upload data files (CSV, Excel, JSON) in the sidebar to start querying."
        )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("🔍 Generated SQL"):
                    st.code(msg["sql"], language="sql")
            if msg.get("results"):
                with st.expander(f"📋 Raw Results ({msg.get('row_count', 0)} rows)"):
                    st.dataframe(pd.DataFrame(msg["results"]))
            if msg.get("validation"):
                with st.expander("✅ Validation"):
                    st.json(msg["validation"])
            if msg.get("retrieved_context"):
                with st.expander("📚 Retrieved Context (RAG)"):
                    st.markdown(msg["retrieved_context"])

    # Handle input
    user_input = st.chat_input("Ask about your data...")

    if "pending_query" in st.session_state:
        user_input = st.session_state.pending_query
        del st.session_state.pending_query

    if user_input and "graph" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                dynamic_schema = sm.get_dynamic_schema(user_input)
                result = run_query(
                    st.session_state.graph,
                    user_input,
                    st.session_state.conversation_history,
                    dynamic_schema=dynamic_schema,
                )

            st.markdown(result["final_response"])

            assistant_msg = {
                "role": "assistant",
                "content": result["final_response"],
            }

            if result.get("generated_sql"):
                with st.expander("🔍 Generated SQL"):
                    st.code(result["generated_sql"], language="sql")
                assistant_msg["sql"] = result["generated_sql"]

            if result.get("query_result") and result["query_result"].get("results"):
                results = result["query_result"]["results"]
                row_count = result["query_result"].get("row_count", 0)
                with st.expander(f"📋 Raw Results ({row_count} rows)"):
                    st.dataframe(pd.DataFrame(results))
                assistant_msg["results"] = results[:50]
                assistant_msg["row_count"] = row_count

            if result.get("validation"):
                with st.expander("✅ Validation"):
                    st.json(result["validation"])
                assistant_msg["validation"] = result["validation"]

            if result.get("retrieved_context"):
                with st.expander("📚 Retrieved Context (RAG)"):
                    st.markdown(result["retrieved_context"])
                assistant_msg["retrieved_context"] = result["retrieved_context"]

            st.session_state.messages.append(assistant_msg)

            st.session_state.conversation_history.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": result["final_response"]}
            )

            if len(st.session_state.conversation_history) > 10:
                st.session_state.conversation_history = (
                    st.session_state.conversation_history[-10:]
                )

    elif user_input and "graph" not in st.session_state:
        st.warning("Upload data files first to start querying.")


if __name__ == "__main__":
    main()
