from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent / "rfp_prototype.db"

conn = sqlite3.connect(DB_PATH)
conn.execute("""
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    weight REAL NOT NULL,
    max_score REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
)
""")
conn.execute("""
CREATE TABLE IF NOT EXISTS supplier_results (
    rfp_run_id TEXT NOT NULL,
    supplier_name TEXT NOT NULL,
    submission_date TEXT NOT NULL,
    experience_rating REAL NOT NULL,
    absolute_score REAL NOT NULL,
    ppi REAL NOT NULL,
    final_rank INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (rfp_run_id, supplier_name)
)
""")

if conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0] == 0:
    conn.executemany(
        """INSERT INTO evaluation_criteria
        (criterion_id, name, description, weight, max_score, is_active)
        VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (1, "Technical Capability", "Architecture, integrations, scalability, technical fit", 30.0, 10.0, 1),
            (2, "Implementation Plan", "Timeline, milestones, staffing, risk plan", 20.0, 10.0, 1),
            (3, "Commercial Value", "Pricing clarity, total cost, assumptions", 20.0, 10.0, 1),
            (4, "Security & Compliance", "Controls, certifications, privacy, auditability", 20.0, 10.0, 1),
            (5, "Support & Experience", "Support model, similar projects, references", 10.0, 10.0, 1),
        ],
    )
conn.commit()
conn.close()
print(f"Initialized {DB_PATH}")
