"""Append-only SQLite audit log for Case records.

One row per case, never UPDATEd or DELETEd — this is the audit trail the
dashboard reads from and the adjudicator arbitrates over.
"""

import json
import sqlite3
from pathlib import Path

from spine.schema import Case, Evidence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "cases.db"

_DDL = """
CREATE TABLE IF NOT EXISTS cases (
    case_id        TEXT PRIMARY KEY,
    source_agent   TEXT NOT NULL,
    entity_id      TEXT,
    entity_type    TEXT,
    evidence       TEXT NOT NULL,    -- JSON array of {signal, value, weight}
    confidence     REAL,
    cost_estimate  REAL,
    decision       TEXT,
    reasoning_text TEXT,
    timestamp      TEXT
);
"""


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_DDL)
    return conn


def insert_cases(conn: sqlite3.Connection, cases: list[Case]) -> int:
    """Append cases to the audit log. Returns the number of rows written."""
    rows = [
        (
            c.case_id,
            c.source_agent,
            c.entity_id,
            c.entity_type,
            json.dumps([e.__dict__ for e in c.evidence]),
            c.confidence,
            c.cost_estimate,
            c.decision,
            c.reasoning_text,
            c.timestamp.isoformat(),
        )
        for c in cases
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    return len(rows)


def fetch_cases(
    conn: sqlite3.Connection,
    source_agent: str | None = None,
    limit: int = 500,
) -> list[Case]:
    """Read cases back, newest first, optionally filtered by agent."""
    sql = "SELECT * FROM cases"
    params: tuple = ()
    if source_agent:
        sql += " WHERE source_agent = ?"
        params = (source_agent,)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    rows = conn.execute(sql + "", (*params, limit)).fetchall()

    cases = []
    for r in rows:
        cases.append(
            Case(
                case_id=r[0],
                source_agent=r[1],
                entity_id=r[2],
                entity_type=r[3],
                evidence=[Evidence(**e) for e in json.loads(r[4])],
                confidence=r[5],
                cost_estimate=r[6],
                decision=r[7],
                reasoning_text=r[8],
                timestamp=r[9],
            )
        )
    return cases


def count_cases(conn: sqlite3.Connection, source_agent: str | None = None) -> int:
    sql = "SELECT COUNT(*) FROM cases"
    params: tuple = ()
    if source_agent:
        sql += " WHERE source_agent = ?"
        params = (source_agent,)
    return conn.execute(sql, params).fetchone()[0]
