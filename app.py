import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import fitz
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, ValidationError

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "rfp_prototype.db"


# -----------------------------
# SQLite
# -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    conn.commit()
    return conn


def seed_criteria(conn):
    if conn.execute("SELECT COUNT(*) FROM evaluation_criteria").fetchone()[0] == 0:
        rows = [
            (1, "Technical Capability", "Architecture, integrations, scalability, technical fit", 30.0, 10.0, 1),
            (2, "Implementation Plan", "Timeline, milestones, staffing, risk plan", 20.0, 10.0, 1),
            (3, "Commercial Value", "Pricing clarity, total cost, assumptions", 20.0, 10.0, 1),
            (4, "Security & Compliance", "Controls, certifications, privacy, auditability", 20.0, 10.0, 1),
            (5, "Support & Experience", "Support model, similar projects, references", 10.0, 10.0, 1),
        ]
        conn.executemany(
            """INSERT INTO evaluation_criteria
               (criterion_id, name, description, weight, max_score, is_active)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()


def load_active_criteria(conn):
    rows = conn.execute("""
        SELECT criterion_id, name, description, weight, max_score, is_active
        FROM evaluation_criteria
        WHERE is_active = 1
        ORDER BY criterion_id
    """).fetchall()

    criteria = [
        {
            "criterion_id": r[0],
            "name": r[1],
            "description": r[2],
            "weight": r[3],
            "max_score": r[4],
            "is_active": bool(r[5]),
        }
        for r in rows
    ]

    if not criteria:
        raise ValueError("No active evaluation criteria.")

    total = sum(c["weight"] for c in criteria)
    if not math.isclose(total, 100.0):
        raise ValueError(
            f"Active criterion weights must total 100%. Current total: {total:.2f}%"
        )

    return criteria


# -----------------------------
# Document Tool
# -----------------------------
def extract_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text("text") for page in doc).strip()


# -----------------------------
# Evaluation Agent
# -----------------------------
class CriterionEvaluation(BaseModel):
    criterion_id: int
    score: float
    max_score: float
    justification: str
    evidence: str


class EvaluationResult(BaseModel):
    supplier_name: str
    criteria: list[CriterionEvaluation]
    risks: list[str] = []
    overall_summary: str = ""


def build_evaluation_prompt(supplier_name, proposal_text, criteria):
    criteria_text = "\n".join(
        f"- ID {c['criterion_id']}: {c['name']} | Weight {c['weight']}% | "
        f"Max {c['max_score']} | Inspect: {c['description']}"
        for c in criteria
    )
    return (
        "You are an RFP proposal evaluation agent.\n\n"
        f"Supplier: {supplier_name}\n\n"
        f"ACTIVE CRITERIA:\n{criteria_text}\n\n"
        "RULES:\n"
        "1. Use ONLY evidence present in the proposal.\n"
        "2. Do not invent certifications, prices, references or capabilities.\n"
        "3. Return exactly one result for every active criterion.\n"
        "4. Score from 0 through the stated maximum.\n"
        "5. If information is missing, state that evidence is missing.\n"
        "6. Provide concise evidence and justification.\n"
        "7. Do NOT calculate weighted score, PPI, benchmark or final rank. Python will do that.\n"
        "8. Return JSON matching the requested schema.\n\n"
        f"PROPOSAL:\n--- BEGIN ---\n{proposal_text}\n--- END ---"
    )


def evaluation_json_schema():
    return {
        "type": "object",
        "properties": {
            "supplier_name": {"type": "string"},
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "integer"},
                        "score": {"type": "number"},
                        "max_score": {"type": "number"},
                        "justification": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["criterion_id", "score", "max_score", "justification", "evidence"],
                    "additionalProperties": False,
                },
            },
            "risks": {"type": "array", "items": {"type": "string"}},
            "overall_summary": {"type": "string"},
        },
        "required": ["supplier_name", "criteria", "risks", "overall_summary"],
        "additionalProperties": False,
    }


def call_llm(supplier_name, proposal_text, criteria):
    api_key = st.session_state.get("OPENROUTER_API_KEY")
    model_name = st.session_state.get("MODEL_NAME")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY has not been entered.")

    if not model_name:
        raise ValueError("MODEL_NAME has not been entered.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": build_evaluation_prompt(supplier_name, proposal_text, criteria),
        }],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rfp_evaluation",
                "strict": True,
                "schema": evaluation_json_schema(),
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError(
            f"LLM returned empty content. Finish reason: {response.choices[0].finish_reason}"
        )

    return json.loads(content)


# -----------------------------
# Validation Tool
# -----------------------------
def validate_and_normalize(raw_result, criteria):
    warnings = []
    try:
        parsed = EvaluationResult.model_validate(raw_result)
    except ValidationError as e:
        return None, [f"Schema validation failed: {e}"]

    criterion_map = {c["criterion_id"]: c for c in criteria}
    returned = {x.criterion_id: x for x in parsed.criteria}
    normalized = []

    for cid, c in criterion_map.items():
        if cid not in returned:
            warnings.append(f"Criterion {cid} ({c['name']}) missing; defaulted to 0.")
            normalized.append({
                "criterion_id": cid,
                "score": 0.0,
                "max_score": float(c["max_score"]),
                "justification": "No valid result returned.",
                "evidence": "No evidence available.",
            })
            continue

        item = returned[cid]
        score = float(item.score)
        max_score = float(c["max_score"])

        if score < 0:
            warnings.append(f"Criterion {cid} score {score} was below 0; clipped to 0.")
            score = 0.0
        if score > max_score:
            warnings.append(f"Criterion {cid} score {score} exceeded {max_score}; clipped.")
            score = max_score

        normalized.append({
            "criterion_id": cid,
            "score": score,
            "max_score": max_score,
            "justification": item.justification,
            "evidence": item.evidence,
        })

    return {
        "supplier_name": parsed.supplier_name,
        "criteria": normalized,
        "risks": parsed.risks,
        "overall_summary": parsed.overall_summary,
    }, warnings


# -----------------------------
# Ranking Tool
# -----------------------------
def calculate_absolute_score(evaluation, criteria):
    by_id = {c["criterion_id"]: c for c in criteria}
    total = 0.0
    details = []

    for item in evaluation["criteria"]:
        c = by_id[item["criterion_id"]]
        contribution = (item["score"] / item["max_score"]) * c["weight"]
        total += contribution
        details.append({
            "criterion_id": item["criterion_id"],
            "name": c["name"],
            "score": item["score"],
            "max_score": item["max_score"],
            "weight": c["weight"],
            "weighted_contribution": contribution,
            "evidence": item["evidence"],
            "justification": item["justification"],
        })

    return total, details


def add_peer_metrics(supplier_scores, criteria):
    benchmarks = {}

    for c in criteria:
        cid = c["criterion_id"]
        benchmarks[cid] = max(
            item["score"]
            for supplier in supplier_scores
            for item in supplier["criteria"]
            if item["criterion_id"] == cid
        )

    for supplier in supplier_scores:
        ppi = 0.0
        for item in supplier["criteria"]:
            benchmark = benchmarks[item["criterion_id"]]
            item["benchmark"] = benchmark
            item["gap"] = item["score"] - benchmark

            if benchmark == 0:
                relative = 100.0 if item["score"] == 0 else 0.0
            else:
                relative = (item["score"] / benchmark) * 100.0

            item["relative_performance_pct"] = relative
            weight = next(
                c["weight"] for c in criteria
                if c["criterion_id"] == item["criterion_id"]
            )
            ppi += relative * weight / 100.0

        supplier["ppi"] = ppi

    return supplier_scores, benchmarks


def rank_suppliers(supplier_scores):
    ranked = sorted(
        supplier_scores,
        key=lambda x: (
            -x["ppi"],
            x["submission_date"],
            -x["experience_rating"],
            x["supplier_name"].lower(),
        ),
    )

    for rank, supplier in enumerate(ranked, 1):
        supplier["final_rank"] = rank

    return ranked


# -----------------------------
# Persistence
# -----------------------------
def persist_run(conn, run_id, created_at, criteria, ranked_suppliers):
    conn.execute(
        "INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
        (run_id, created_at, "completed"),
    )

    for supplier in ranked_suppliers:
        result_json = json.dumps(supplier, ensure_ascii=False)
        conn.execute(
            """INSERT INTO supplier_results
               (rfp_run_id, supplier_name, submission_date, experience_rating,
                absolute_score, ppi, final_rank, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                supplier["supplier_name"],
                supplier["submission_date"],
                supplier["experience_rating"],
                supplier["absolute_score"],
                supplier["ppi"],
                supplier["final_rank"],
                result_json,
            ),
        )

    conn.commit()


def build_final_result(run_id, created_at, criteria, benchmarks, ranked_suppliers):
    return {
        "rfp_run_id": run_id,
        "created_at": created_at,
        "criteria": criteria,
        "criterion_benchmarks": benchmarks,
        "suppliers": ranked_suppliers,
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Agentic RFP Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("Agentic RFP Evaluation & Supplier Ranking")
st.caption(
    "LLM-assisted proposal evaluation with deterministic scoring, "
    "peer benchmarking, PPI, ranking, and SQLite persistence."
)

# -------------------------------------------------
# Runtime LLM Configuration
# -------------------------------------------------
st.sidebar.header("LLM Configuration")

openrouter_api_key = st.sidebar.text_input(
    "OPENROUTER_API_KEY",
    type="password",
    help="Enter your OpenRouter API key. It is used only for the current session.",
)

model_name = st.sidebar.text_input(
    "MODEL_NAME",
    value="openai/gpt-oss-120b",
    help="OpenRouter model identifier.",
)

st.session_state["OPENROUTER_API_KEY"] = openrouter_api_key
st.session_state["MODEL_NAME"] = model_name

if openrouter_api_key and model_name:
    st.sidebar.success("LLM configuration ready.")
else:
    st.sidebar.warning("Enter both API key and model name before evaluation.")

conn = get_conn()
seed_criteria(conn)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Criteria", "Supplier Input", "Results", "Run Details"]
)

with tab1:
    st.subheader("Active evaluation criteria")
    criteria = load_active_criteria(conn)
    criteria_df = pd.DataFrame(criteria)

    st.dataframe(
        criteria_df[["criterion_id", "name", "description", "weight", "max_score"]],
        use_container_width=True,
        hide_index=True,
    )

    st.info("Criteria and weights are loaded from SQLite. Active weights must total 100%.")

with tab2:
    st.subheader("Upload supplier proposals")
    st.write("Upload multiple supplier PDFs and provide metadata for each supplier.")

    uploads = st.file_uploader(
        "Supplier proposal PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    metadata = {}

    if uploads:
        st.caption(
            "Supplier name is taken from the PDF filename by default; you can edit it below."
        )

        for idx, uploaded in enumerate(uploads):
            default_name = (
                Path(uploaded.name).stem
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )

            c1, c2, c3 = st.columns([2, 1, 1])

            with c1:
                name = st.text_input(
                    "Supplier name",
                    value=default_name,
                    key=f"name_{idx}",
                )

            with c2:
                date = st.date_input(
                    "Submission date",
                    key=f"date_{idx}",
                )

            with c3:
                exp = st.number_input(
                    "Experience rating",
                    min_value=0.0,
                    max_value=10.0,
                    value=5.0,
                    step=1.0,
                    key=f"exp_{idx}",
                )

            metadata[uploaded.name] = {
                "supplier_name": name,
                "submission_date": date.isoformat(),
                "experience_rating": exp,
            }

    if st.button("Evaluate suppliers", type="primary", disabled=not uploads):
        if not openrouter_api_key or not model_name:
            st.error(
                "Enter OPENROUTER_API_KEY and MODEL_NAME in the LLM Configuration panel before evaluating suppliers."
            )
            st.stop()

        criteria = load_active_criteria(conn)
        supplier_scores = []
        all_warnings = {}
        errors = []

        progress = st.progress(0.0)

        for idx, uploaded in enumerate(uploads):
            meta = metadata[uploaded.name]

            try:
                proposal_text = extract_pdf_text(uploaded.getvalue())
                raw = call_llm(meta["supplier_name"], proposal_text, criteria)
                validated, warnings = validate_and_normalize(raw, criteria)

                if validated is None:
                    raise ValueError("; ".join(warnings))

                score, details = calculate_absolute_score(validated, criteria)

                supplier_scores.append({
                    "supplier_name": meta["supplier_name"],
                    "submission_date": meta["submission_date"],
                    "experience_rating": meta["experience_rating"],
                    "absolute_score": score,
                    "criteria": details,
                    "warnings": warnings,
                    "risks": validated["risks"],
                    "overall_summary": validated["overall_summary"],
                })

                all_warnings[meta["supplier_name"]] = warnings

            except Exception as e:
                errors.append(f"{meta['supplier_name']}: {e}")

            progress.progress((idx + 1) / len(uploads))

        if errors:
            for error in errors:
                st.error(error)

        elif not supplier_scores:
            st.error("No supplier evaluations were completed.")

        else:
            supplier_scores, benchmarks = add_peer_metrics(
                supplier_scores,
                criteria,
            )

            ranked = rank_suppliers(supplier_scores)

            run_id = "RFP-PROTOTYPE-" + datetime.now().strftime("%Y%m%d-%H%M%S")
            created_at = datetime.now().isoformat()

            final_result = build_final_result(
                run_id,
                created_at,
                criteria,
                benchmarks,
                ranked,
            )

            try:
                persist_run(
                    conn,
                    run_id,
                    created_at,
                    criteria,
                    ranked,
                )
            except sqlite3.IntegrityError:
                run_id = (
                    "RFP-PROTOTYPE-"
                    + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                )
                created_at = datetime.now().isoformat()

                final_result = build_final_result(
                    run_id,
                    created_at,
                    criteria,
                    benchmarks,
                    ranked,
                )

                persist_run(
                    conn,
                    run_id,
                    created_at,
                    criteria,
                    ranked,
                )

            st.session_state["latest_result"] = final_result
            st.session_state["latest_run_id"] = run_id

            st.success(f"Evaluation complete. RFP_RUN_ID: {run_id}")

            if any(all_warnings.values()):
                st.warning("One or more suppliers produced validation warnings.")
            else:
                st.success("All suppliers passed validation with zero warnings.")

with tab3:
    result = st.session_state.get("latest_result")

    if not result:
        st.info("Run an evaluation from Supplier Input to see results.")

    else:
        ranked = result["suppliers"]

        leaderboard = pd.DataFrame([
            {
                "Rank": s["final_rank"],
                "Supplier": s["supplier_name"],
                "Absolute Score": round(s["absolute_score"], 2),
                "PPI %": round(s["ppi"], 2),
                "Submission Date": s["submission_date"],
                "Experience": s["experience_rating"],
            }
            for s in ranked
        ])

        st.subheader("Leaderboard")
        st.dataframe(
            leaderboard,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Detailed scorecards")

        selected = st.selectbox(
            "Select supplier",
            [s["supplier_name"] for s in ranked],
        )

        supplier = next(
            s for s in ranked if s["supplier_name"] == selected
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Rank", supplier["final_rank"])
        c2.metric("Absolute Score", f"{supplier['absolute_score']:.2f}")
        c3.metric("PPI", f"{supplier['ppi']:.2f}%")

        score_rows = []

        for item in supplier["criteria"]:
            score_rows.append({
                "Criterion": item["name"],
                "Score": f"{item['score']:.1f}/{item['max_score']:.1f}",
                "Weight": item["weight"],
                "Benchmark": item["benchmark"],
                "Gap": item["gap"],
                "Relative %": round(item["relative_performance_pct"], 2),
                "Weighted Contribution": round(item["weighted_contribution"], 2),
            })

        st.dataframe(
            pd.DataFrame(score_rows),
            use_container_width=True,
            hide_index=True,
        )

        for item in supplier["criteria"]:
            with st.expander(item["name"]):
                st.markdown("**Evidence**")
                st.write(item["evidence"])

                st.markdown("**Justification**")
                st.write(item["justification"])

        if supplier.get("risks"):
            st.markdown("**Risks**")
            for risk in supplier["risks"]:
                st.write(f"- {risk}")

        st.markdown("**Overall summary**")
        st.write(supplier.get("overall_summary", ""))

with tab4:
    result = st.session_state.get("latest_result")

    if not result:
        st.info("Run an evaluation to see run details.")

    else:
        st.write("**RFP_RUN_ID:**", result["rfp_run_id"])
        st.write("**Created at:**", result["created_at"])
        st.write(
            "**Tie-break order:** Higher PPI → earlier submission date → "
            "higher historical experience rating → supplier name ascending."
        )

        json_bytes = json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")

        st.download_button(
            "Download complete result as JSON",
            data=json_bytes,
            file_name=f"{result['rfp_run_id']}.json",
            mime="application/json",
        )
