# Agentic RFP Evaluation & Supplier Ranking

Streamlit implementation of the classroom mini-project.

## Architecture

PDF upload → PDF extraction → LLM structured evaluation → validation/normalization → deterministic weighted scoring → peer benchmark/PPI → deterministic ranking → SQLite persistence → leaderboard/scorecards/JSON.

The LLM evaluates proposal content. Python performs arithmetic, benchmarking, PPI, tie-breaks, and ranking.

## Required files

- `app.py` — Streamlit application
- `rfp_prototype.db` — SQLite database (created automatically on first run)
- `requirements.txt` — Python dependencies
- `sample_pdfs/` — four synthetic supplier proposals

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create `.streamlit/secrets.toml` locally:

```toml
OPENROUTER_API_KEY = "your-key-here"
```

Never commit `secrets.toml`.

## Deploy to Streamlit Community Cloud

Push this repository to GitHub, connect GitHub to Streamlit Community Cloud, select the repository/branch and `app.py`, and add `OPENROUTER_API_KEY` in the deployment Secrets settings.

## Scoring

Absolute score:
`sum((criterion score / max score) * criterion weight)`

Benchmark:
highest valid score for each criterion across suppliers.

Relative performance:
`(supplier score / benchmark) * 100`

PPI:
weighted average of criterion relative-performance percentages.

Tie-break:
1. Higher PPI
2. Earlier submission date
3. Higher historical experience rating
4. Supplier name ascending

## SQLite

The app loads configurable active criteria from `evaluation_criteria` and persists each completed run in `rfp_runs` and `supplier_results`.

## Synthetic suppliers

Apex Systems, BrightPath Tech, NexaWorks, and Orbit Digital are fictional proposals created for this project.
