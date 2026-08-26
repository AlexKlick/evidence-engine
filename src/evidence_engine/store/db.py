"""Engine/session factory + schema init (SQLite default, Postgres optional)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from evidence_engine.store.models import Base

SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_engine(db_url: str) -> Engine:
    if db_url.startswith("sqlite:///"):
        path = Path(db_url.removeprefix("sqlite:///"))
        if path != Path(":memory:") and str(path) != "":
            path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url, future=True)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version INTEGER NOT NULL, applied_at TEXT NOT NULL)"
            )
        )
        row = conn.execute(text("SELECT version FROM schema_version")).first()
        if row is None:
            conn.execute(
                text("INSERT INTO schema_version (version, applied_at) VALUES (:v, :ts)"),
                {"v": SCHEMA_VERSION, "ts": _utcnow_iso()},
            )
        # v2 (2026-08-26): decision_reason lands on existing DBs. SQLite's
        # ADD COLUMN does not support IF NOT EXISTS on the bundled version
        # shipped with some Python builds, so we guard against the duplicate
        # column error instead. PRAGMA table_info is portable across
        # SQLite and Postgres. No Alembic yet.
        existing_cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(experiment)"))
        }
        if "decision_reason" not in existing_cols:
            conn.execute(
                text(
                    "ALTER TABLE experiment ADD COLUMN decision_reason VARCHAR(500)"
                )
            )
