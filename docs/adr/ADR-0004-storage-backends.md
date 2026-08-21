# ADR-0004: SQLite default, Postgres+pgvector path

Date: 2026-08-21
Status: accepted

## Context

The founding doc prescribes PostgreSQL + pgvector for the evidence store. The
workstation has a Postgres 16 host cluster on `127.0.0.1:5432` with pgvector
0.6 installed system-wide, plus containerized PG instances owned by other
projects. The scaffold must also run its full test suite hermetically and work
for a single-operator personal pipeline with zero setup friction.

## Decision

1. SQLAlchemy 2.0 ORM for the evidence graph; engine selected by config:
   empty `database.dsn` → SQLite at `data/evidence_engine.db` (default),
   non-empty → that DSN (e.g. the host PG cluster).
2. In the scaffold, embeddings are stored as JSON columns and similarity is
   computed in-process (cosine). Vector counts at personal scale are small;
   this keeps both backends identical.
3. `init_db()` uses metadata create-all + a `schema_version` stamp.
   Alembic migrations are deliberately deferred until the schema stabilizes.
4. Upgrade path (documented, not built): a pgvector-native `vector(1024)`
   column + ANN index behind the same repository interface. Do not hijack
   other projects' PG containers; create a dedicated database on the host
   cluster when promoting.

## Consequences

- Tests run on in-memory SQLite with no external dependencies.
- SQLite lacks concurrent writers — acceptable for a single-operator CLI
  pipeline; moving to Postgres is a config change.
- ANN-grade search is not needed until evidence volume grows by orders of
  magnitude; the repository interface isolates that change.
