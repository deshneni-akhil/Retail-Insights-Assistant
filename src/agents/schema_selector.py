# Schema selector — intelligent table selection for large multi-table sessions.
from src.config import SCHEMA_TABLE_THRESHOLD, SCHEMA_TOP_K_TABLES
from src.data.db import Database
from src.data.vectorstore import SessionVectorStore


def select_schema(
    db: Database,
    vectorstore: SessionVectorStore,
    user_query: str,
) -> str:
    """Select relevant table schemas for a query.

    For small sessions (<=SCHEMA_TABLE_THRESHOLD tables), returns all schemas unchanged.
    For large sessions, uses ChromaDB semantic search to select top-k relevant tables
    and prepends a compact table directory so the LLM knows what else is available.

    Returns:
        Schema text ready for injection into the NL2SQL prompt.
    """
    tables = db.list_tables()

    if not tables:
        return "No tables available. Upload data files to get started."

    # Fast path: small schema, send everything
    if len(tables) <= SCHEMA_TABLE_THRESHOLD:
        return db.get_all_schemas_text()

    # Large schema: semantic table selection
    relevant_tables = vectorstore.search_tables(user_query, top_k=SCHEMA_TOP_K_TABLES)

    # If search returned nothing (empty collection), fall back to all tables
    if not relevant_tables:
        return db.get_all_schemas_text()

    # Filter to tables that actually exist (safety check)
    relevant_tables = [t for t in relevant_tables if t in tables]
    if not relevant_tables:
        return db.get_all_schemas_text()

    # Assemble: table directory + detailed schemas for selected tables
    directory = db.get_table_directory()
    selected_count = len(relevant_tables)
    total_count = len(tables)

    parts = [
        directory,
        "",
        f"DETAILED SCHEMA ({selected_count} of {total_count} tables, selected by relevance):",
        "",
    ]

    for table in relevant_tables:
        parts.append(db.get_schema_text(table))

    return "\n".join(parts)
