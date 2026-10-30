"""
src/database.py — SQLite persistence layer.

Design principles:
- All tables are created automatically on first use (CREATE TABLE IF NOT EXISTS).
- Deduplication key: canonical_domain (UNIQUE).  On conflict, the existing
  company_id and created_at are preserved; all other columns are updated.
- Every SQL statement uses parameterized queries — no string interpolation.
- Database operations are independent of Streamlit or any UI framework.
- Repeated pipeline runs are idempotent: re-processing the same domain
  updates the record rather than creating a duplicate.
- An in-memory database (path=":memory:") is supported for tests.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.schemas import CompanyRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid4())


def _json_dumps(value: object) -> str:
    """Serialise lists/dicts to JSON for storage in TEXT columns."""
    return json.dumps(value, default=str)


def _json_loads(value: Optional[str]) -> list:
    if not value:
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL_COMPANIES = """
CREATE TABLE IF NOT EXISTS companies (
    company_id              TEXT PRIMARY KEY,
    company_name            TEXT NOT NULL,
    canonical_domain        TEXT NOT NULL UNIQUE,
    location                TEXT,
    services                TEXT DEFAULT '[]',
    relevant_facts          TEXT DEFAULT '[]',
    source_urls             TEXT DEFAULT '[]',
    retrieved_at            TEXT,
    qualification_status    TEXT NOT NULL CHECK(
                                qualification_status IN (
                                    'qualified','not_qualified',
                                    'needs_review','error'
                                )
                            ),
    qualification_reasons   TEXT DEFAULT '[]',
    supporting_evidence     TEXT DEFAULT '[]',
    missing_information     TEXT DEFAULT '[]',
    used_llm_judgment       INTEGER NOT NULL DEFAULT 0,
    review_status           TEXT NOT NULL DEFAULT 'pending' CHECK(
                                review_status IN ('pending','approved','rejected')
                            ),
    processing_error        TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);
"""

_DDL_SOURCE_EVIDENCE = """
CREATE TABLE IF NOT EXISTS source_evidence (
    evidence_id     TEXT PRIMARY KEY,
    company_id      TEXT NOT NULL REFERENCES companies(company_id),
    url             TEXT NOT NULL,
    title           TEXT,
    snippet         TEXT,
    retrieved_at    TEXT
);
"""

_DDL_REVIEW_LOG = """
CREATE TABLE IF NOT EXISTS review_log (
    log_id          TEXT PRIMARY KEY,
    company_id      TEXT NOT NULL REFERENCES companies(company_id),
    action          TEXT NOT NULL CHECK(action IN ('approved','rejected')),
    reviewer_note   TEXT,
    acted_at        TEXT NOT NULL
);
"""

_DDL_EVALUATION_RUNS = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id                  TEXT PRIMARY KEY,
    run_at                  TEXT NOT NULL,
    total_samples           INTEGER NOT NULL,
    classified              INTEGER NOT NULL,
    unclassified            INTEGER NOT NULL,
    precision               REAL,
    recall                  REAL,
    f1_score                REAL,
    confusion_matrix_json   TEXT,
    estimated_cost_usd      REAL,
    notes                   TEXT
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """
    Thin wrapper around a SQLite connection.

    Usage:
        db = Database("data/prospects.db")   # or Database(":memory:")
        db.upsert_company(record)
        companies = db.list_companies()
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            self._path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.executescript(
                _DDL_COMPANIES
                + _DDL_SOURCE_EVIDENCE
                + _DDL_REVIEW_LOG
                + _DDL_EVALUATION_RUNS
            )

    # ------------------------------------------------------------------
    # Companies — upsert (insert or update on duplicate domain)
    # ------------------------------------------------------------------

    def upsert_company(self, record: CompanyRecord) -> str:
        """
        Insert the company record.  If a row with the same canonical_domain
        already exists, update all mutable columns and return the existing
        company_id (preserving identity across runs).

        Returns the company_id that was used (existing or new).
        """
        now = _utcnow_iso()

        # Check for an existing record with this domain
        existing = self._conn.execute(
            "SELECT company_id, created_at FROM companies WHERE canonical_domain = ?",
            (record.canonical_domain,),
        ).fetchone()

        if existing:
            company_id = existing["company_id"]
            created_at = existing["created_at"]
        else:
            company_id = record.company_id
            created_at = record.created_at.isoformat() if record.created_at else now

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO companies (
                    company_id, company_name, canonical_domain, location,
                    services, relevant_facts, source_urls, retrieved_at,
                    qualification_status, qualification_reasons,
                    supporting_evidence, missing_information,
                    used_llm_judgment, review_status, processing_error,
                    created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                ON CONFLICT(canonical_domain) DO UPDATE SET
                    company_name            = excluded.company_name,
                    location                = excluded.location,
                    services                = excluded.services,
                    relevant_facts          = excluded.relevant_facts,
                    source_urls             = excluded.source_urls,
                    retrieved_at            = excluded.retrieved_at,
                    qualification_status    = excluded.qualification_status,
                    qualification_reasons   = excluded.qualification_reasons,
                    supporting_evidence     = excluded.supporting_evidence,
                    missing_information     = excluded.missing_information,
                    used_llm_judgment       = excluded.used_llm_judgment,
                    processing_error        = excluded.processing_error,
                    updated_at              = excluded.updated_at
                """,
                (
                    company_id,
                    record.company_name,
                    record.canonical_domain,
                    record.location,
                    _json_dumps([str(u) for u in record.services]),
                    _json_dumps(record.relevant_facts),
                    _json_dumps([str(u) for u in record.source_urls]),
                    record.retrieved_at.isoformat() if record.retrieved_at else now,
                    record.qualification_status,
                    _json_dumps(record.qualification_reasons),
                    _json_dumps(record.supporting_evidence),
                    _json_dumps(record.missing_information),
                    int(record.used_llm_judgment),
                    record.review_status,
                    record.processing_error,
                    created_at,
                    now,
                ),
            )

        return company_id

    # ------------------------------------------------------------------
    # Source evidence
    # ------------------------------------------------------------------

    def add_source_evidence(
        self,
        company_id: str,
        url: str,
        title: Optional[str] = None,
        snippet: Optional[str] = None,
        retrieved_at: Optional[str] = None,
    ) -> str:
        """Insert a source evidence row.  Returns the evidence_id."""
        evidence_id = _new_id()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO source_evidence
                    (evidence_id, company_id, url, title, snippet, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    company_id,
                    url,
                    title,
                    snippet,
                    retrieved_at or _utcnow_iso(),
                ),
            )
        return evidence_id

    # ------------------------------------------------------------------
    # Human review
    # ------------------------------------------------------------------

    def record_review(
        self,
        company_id: str,
        action: str,          # "approved" | "rejected"
        reviewer_note: Optional[str] = None,
    ) -> None:
        """
        Update the company's review_status and append a row to review_log.
        Raises ValueError for unknown actions.
        """
        if action not in {"approved", "rejected"}:
            raise ValueError(f"Unknown review action: {action!r}")

        now = _utcnow_iso()
        log_id = _new_id()

        with self._conn:
            self._conn.execute(
                "UPDATE companies SET review_status = ?, updated_at = ? WHERE company_id = ?",
                (action, now, company_id),
            )
            self._conn.execute(
                """
                INSERT INTO review_log (log_id, company_id, action, reviewer_note, acted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (log_id, company_id, action, reviewer_note, now),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_company_by_domain(self, canonical_domain: str) -> Optional[dict]:
        """Return the company row as a dict, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM companies WHERE canonical_domain = ?",
            (canonical_domain,),
        ).fetchone()
        return dict(row) if row else None

    def get_company_by_id(self, company_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM companies WHERE company_id = ?",
            (company_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_companies(
        self,
        qualification_status: Optional[str] = None,
        review_status: Optional[str] = None,
    ) -> list[dict]:
        """Return all companies, optionally filtered by status."""
        query = "SELECT * FROM companies WHERE 1=1"
        params: list = []

        if qualification_status:
            query += " AND qualification_status = ?"
            params.append(qualification_status)
        if review_status:
            query += " AND review_status = ?"
            params.append(review_status)

        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_source_evidence(self, company_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM source_evidence WHERE company_id = ? ORDER BY retrieved_at",
            (company_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_review_log(self, company_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM review_log WHERE company_id = ? ORDER BY acted_at",
            (company_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_company_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

    # ------------------------------------------------------------------
    # Evaluation runs
    # ------------------------------------------------------------------

    def save_evaluation_run(
        self,
        total_samples: int,
        classified: int,
        unclassified: int,
        precision: Optional[float],
        recall: Optional[float],
        f1_score: Optional[float],
        confusion_matrix: dict,
        estimated_cost_usd: Optional[float],
        notes: str,
    ) -> str:
        run_id = _new_id()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, run_at, total_samples, classified, unclassified,
                    precision, recall, f1_score,
                    confusion_matrix_json, estimated_cost_usd, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _utcnow_iso(),
                    total_samples,
                    classified,
                    unclassified,
                    precision,
                    recall,
                    f1_score,
                    _json_dumps(confusion_matrix),
                    estimated_cost_usd,
                    notes,
                ),
            )
        return run_id

    def latest_evaluation_run(self) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM evaluation_runs ORDER BY run_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_) -> None:
        self.close()
