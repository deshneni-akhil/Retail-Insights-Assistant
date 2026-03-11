# Session manager — handles lifecycle of analysis sessions with isolated DuckDB + ChromaDB.
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from src.data.db import Database
from src.data.vectorstore import SessionVectorStore
from src.agents.schema_selector import select_schema


class SessionManager:
    """Manages analysis sessions, each with its own DuckDB and ChromaDB storage."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.db: Database | None = None
        self.vectorstore: SessionVectorStore | None = None
        self.active_session: str | None = None

    def create_session(self, name: str) -> None:
        """Create a new session with a fresh DuckDB and ChromaDB."""
        name = self._sanitize_name(name)
        session_dir = self.sessions_dir / name
        if session_dir.exists():
            raise ValueError(f"Session '{name}' already exists")

        session_dir.mkdir(parents=True)
        (session_dir / "chromadb").mkdir()

        meta = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "tables": [],
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self._activate(name)

    def load_session(self, name: str) -> None:
        """Open an existing session."""
        session_dir = self.sessions_dir / name
        if not session_dir.exists():
            raise ValueError(f"Session '{name}' not found")

        self._activate(name)

    def list_sessions(self) -> list[dict]:
        """Return metadata for all sessions."""
        sessions = []
        if not self.sessions_dir.exists():
            return sessions

        for entry in sorted(self.sessions_dir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    sessions.append(meta)
                except (json.JSONDecodeError, OSError):
                    sessions.append({"name": entry.name, "created_at": "", "tables": []})
        return sessions

    def delete_session(self, name: str) -> None:
        """Delete a session and all its data."""
        session_dir = self.sessions_dir / name
        if not session_dir.exists():
            return

        # Close if it's the active session
        if self.active_session == name:
            self.close()

        shutil.rmtree(session_dir)

    def update_meta(self, table_name: str, filename: str, row_count: int) -> None:
        """Record a newly uploaded table in the session metadata."""
        if not self.active_session:
            return

        meta_path = self.sessions_dir / self.active_session / "meta.json"
        meta = json.loads(meta_path.read_text())

        # Replace if same table name exists (re-upload)
        meta["tables"] = [t for t in meta["tables"] if t["name"] != table_name]
        meta["tables"].append({
            "name": table_name,
            "filename": filename,
            "row_count": row_count,
            "uploaded_at": datetime.now().isoformat(),
        })

        meta_path.write_text(json.dumps(meta, indent=2))

    def get_dynamic_schema(self, user_query: str | None = None) -> str:
        """Get schema text for the active session.

        Args:
            user_query: If provided and the session has many tables, uses semantic
                        search to select only the most relevant tables.
                        If None, returns all table schemas (used for caching).
        """
        if not self.db:
            return "No active session."
        if user_query and self.vectorstore:
            return select_schema(self.db, self.vectorstore, user_query)
        return self.db.get_all_schemas_text()

    def close(self) -> None:
        """Close the active session's connections."""
        if self.db:
            self.db.close()
            self.db = None
        self.vectorstore = None
        self.active_session = None

    def _activate(self, name: str) -> None:
        """Close any current session and activate the given one."""
        self.close()

        session_dir = self.sessions_dir / name
        db_path = str(session_dir / "data.duckdb")
        chromadb_path = str(session_dir / "chromadb")

        self.db = Database(db_path)
        self.vectorstore = SessionVectorStore(chromadb_path)
        self.active_session = name

        # Index existing tables in ChromaDB for semantic table selection
        self._index_existing_tables()

    def _index_existing_tables(self) -> None:
        """Index all existing tables in ChromaDB for schema selection.

        Called on session activation. Idempotent — uses upsert.
        """
        if not self.db or not self.vectorstore:
            return
        for table_name in self.db.list_tables():
            description = self.db.get_compact_description(table_name)
            self.vectorstore.index_table_schema(table_name, description)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize session name for filesystem safety."""
        name = name.strip().lower()
        name = re.sub(r"[^a-z0-9_\-]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return name[:60]
