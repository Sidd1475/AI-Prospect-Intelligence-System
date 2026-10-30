"""
app.py — Streamlit UI entry point for the AI Prospect Intelligence System.

UI sections:
  1. Sidebar — configuration display (no secrets)
  2. Search — query input + result limit
  3. Research + Qualification — pipeline trigger
  4. Prospects table — with status badges
  5. Prospect detail — evidence, reasons, human review
  6. Evaluation — metrics against labeled dataset
  7. CSV export

The UI imports from config and src modules only.
No API keys are displayed anywhere in the UI.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configure logging before any other imports that might log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Prospect Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Lazy imports — load config after page_config to show clear startup errors
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_config():
    try:
        from config import settings
        return settings, None
    except SystemExit as e:
        return None, str(e)


@st.cache_resource(show_spinner=False)
def _get_database():
    from config import settings
    from src.database import Database
    db = Database(settings.DATABASE_PATH)
    return db


@st.cache_resource(show_spinner=False)
def _get_llm():
    from config import settings
    from src.researcher import build_llm_client
    api_key = (
        settings.GOOGLE_API_KEY
        if settings.LLM_PROVIDER == "gemini"
        else settings.ANTHROPIC_API_KEY
        if settings.LLM_PROVIDER == "anthropic"
        else settings.OPENAI_API_KEY
    )
    return build_llm_client(
        provider=settings.LLM_PROVIDER,
        model=settings.LLM_MODEL,
        api_key=api_key,
        timeout=settings.REQUEST_TIMEOUT_SECONDS,
        max_retries=settings.MAX_RETRIES,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _badge(status: str) -> str:
    colors = {
        "qualified": "🟢",
        "not_qualified": "🔴",
        "needs_review": "🟡",
        "error": "⚫",
    }
    review_colors = {
        "pending": "⏳",
        "approved": "✅",
        "rejected": "❌",
    }
    return colors.get(status, "⚪") + " " + status.replace("_", " ").title()


def _review_badge(status: str) -> str:
    icons = {"pending": "⏳", "approved": "✅", "rejected": "❌"}
    return icons.get(status, "⏳") + " " + status.title()


def _safe_json_loads(value, default=None):
    if default is None:
        default = []
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert DB rows to a display-ready DataFrame."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Parse JSON columns
    for col in ["services", "qualification_reasons", "supporting_evidence",
                "missing_information", "source_urls", "relevant_facts"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _safe_json_loads(v, []))
    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(query: str, max_results: int) -> tuple[int, int, int, list[str]]:
    """
    Execute the full research + qualification + persist pipeline.
    Returns (total, qualified, needs_review, error_messages).
    """
    from config import settings
    from src.search import run_search
    from src.researcher import research_companies
    from src.qualification import qualify_company
    from src.schemas import CompanyRecord
    from src.database import Database

    db = _get_database()
    llm = _get_llm()
    errors: list[str] = []

    # Step 1 — Search
    progress = st.progress(0, text="🔍 Searching with Tavily…")
    search_results = run_search(
        query=query,
        max_results=max_results,
        api_key=settings.TAVILY_API_KEY,
        timeout=settings.REQUEST_TIMEOUT_SECONDS,
        max_retries=settings.MAX_RETRIES,
    )
    progress.progress(25, text=f"🔍 Found {len(search_results)} candidate pages")

    if not search_results:
        progress.empty()
        return 0, 0, 0, ["Tavily returned no results for this query."]

    # Step 2 — Research
    progress.progress(35, text="🧠 Extracting company information with LLM…")
    extraction_results = research_companies(search_results=search_results, llm=llm)
    progress.progress(65, text=f"🧠 Extracted {len(extraction_results)} companies")

    # Step 3 — Qualify + Persist
    qualified_count = 0
    needs_review_count = 0
    total = 0

    for i, (extracted, extract_error) in enumerate(extraction_results):
        pct = 65 + int((i / len(extraction_results)) * 30)
        progress.progress(pct, text=f"⚖️ Qualifying {i+1}/{len(extraction_results)}…")

        if extract_error or extracted is None:
            # Store error record
            domain = (
                search_results[i].source_domain if i < len(search_results) else f"unknown-{i}"
            )
            name = domain
            err_record = CompanyRecord.error_record(
                company_name=name,
                canonical_domain=domain,
                error_message=extract_error or "Unknown extraction error",
            )
            try:
                db.upsert_company(err_record)
            except Exception as e:
                errors.append(f"DB error for {domain}: {e}")
            errors.append(f"Extraction failed for {domain}: {extract_error}")
            total += 1
            continue

        # Qualify
        try:
            qualification = qualify_company(extracted, llm=llm)
        except Exception as e:
            errors.append(f"Qualification failed for {extracted.canonical_domain}: {e}")
            err_record = CompanyRecord.error_record(
                company_name=extracted.company_name,
                canonical_domain=extracted.canonical_domain,
                error_message=str(e),
                source_urls=extracted.source_urls,
            )
            try:
                db.upsert_company(err_record)
            except Exception as dbe:
                errors.append(f"DB error: {dbe}")
            total += 1
            continue

        # Build full record
        record = CompanyRecord.from_extraction_and_qualification(extracted, qualification)

        # Persist
        try:
            company_id = db.upsert_company(record)
            # Store source evidence
            for sr in search_results:
                if sr.source_domain == extracted.canonical_domain:
                    db.add_source_evidence(
                        company_id=company_id,
                        url=str(sr.url),
                        title=sr.title,
                        snippet=sr.snippet,
                        retrieved_at=sr.retrieved_at.isoformat(),
                    )
        except Exception as e:
            errors.append(f"DB persist failed for {extracted.canonical_domain}: {e}")

        # Count
        if qualification.qualification_status == "qualified":
            qualified_count += 1
        elif qualification.qualification_status == "needs_review":
            needs_review_count += 1
        total += 1

    progress.progress(100, text="✅ Pipeline complete!")
    time.sleep(0.5)
    progress.empty()

    return total, qualified_count, needs_review_count, errors


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("🔍 Prospect Intelligence")
        st.caption("AI-powered imaging center discovery")
        st.divider()

        settings, cfg_error = _load_config()
        if cfg_error:
            st.error(f"Configuration error:\n\n{cfg_error}")
            st.stop()

        st.subheader("Configuration")
        st.code(repr(settings))

        st.divider()
        st.caption("**Data source:** Tavily web search")
        st.caption(f"**LLM:** {settings.LLM_PROVIDER} / {settings.LLM_MODEL}")
        st.caption("**Storage:** SQLite")
        st.caption("ℹ️ API keys are never displayed here.")


# ---------------------------------------------------------------------------
# UI — Main tabs
# ---------------------------------------------------------------------------

def render_search_tab():
    st.header("🔎 Search & Research")

    settings, _ = _load_config()

    with st.form("search_form"):
        query = st.text_input(
            "Search query",
            value="outpatient imaging centers in Texas USA",
            help="Natural language query for Tavily web search.",
        )
        max_results = st.slider(
            "Max candidate results",
            min_value=1,
            max_value=settings.MAX_SEARCH_RESULTS,
            value=min(5, settings.MAX_SEARCH_RESULTS),
            help="Each result = one LLM extraction call.",
        )
        submitted = st.form_submit_button("🚀 Run Research", type="primary")

    if submitted:
        if not query.strip():
            st.error("Please enter a search query.")
            return

        st.info(f"Running pipeline: **{query!r}** — up to {max_results} candidates")

        with st.spinner(""):
            total, qualified, needs_review, errors = run_pipeline(query, max_results)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total processed", total)
        col2.metric("🟢 Qualified", qualified)
        col3.metric("🟡 Needs review", needs_review)
        col4.metric("🔴 Errors", len(errors))

        if errors:
            with st.expander(f"⚠️ {len(errors)} error(s) during pipeline"):
                for e in errors:
                    st.warning(e)

        st.success("Pipeline complete. Switch to the **Prospects** tab to review results.")


def render_prospects_tab():
    st.header("📋 Prospects")

    db = _get_database()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Filter by qualification status",
            ["All", "qualified", "not_qualified", "needs_review", "error"],
        )
    with col2:
        review_filter = st.selectbox(
            "Filter by review status",
            ["All", "pending", "approved", "rejected"],
        )

    rows = db.list_companies(
        qualification_status=None if status_filter == "All" else status_filter,
        review_status=None if review_filter == "All" else review_filter,
    )

    if not rows:
        st.info("No prospects found. Run a search in the **Search** tab first.")
        return

    df = _rows_to_df(rows)

    # Display table
    display_df = pd.DataFrame({
        "Company": df["company_name"],
        "Domain": df["canonical_domain"],
        "Location": df.get("location", pd.Series([""] * len(df))),
        "Qualification": df["qualification_status"].apply(_badge),
        "Review": df["review_status"].apply(_review_badge),
        "Services": df["services"].apply(lambda s: ", ".join(s[:3]) if s else ""),
        "Updated": pd.to_datetime(df["updated_at"]).dt.strftime("%Y-%m-%d %H:%M"),
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # CSV Export
    st.divider()
    csv_buf = StringIO()
    export_df = df.copy()
    for col in ["services", "qualification_reasons", "supporting_evidence",
                "missing_information", "source_urls", "relevant_facts"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda v: "; ".join(v) if isinstance(v, list) else v
            )
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Export to CSV",
        data=csv_buf.getvalue(),
        file_name=f"prospects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # Detail view
    st.divider()
    st.subheader("🔍 Prospect Detail")
    company_names = [r["company_name"] for r in rows]
    selected = st.selectbox("Select a company to inspect", company_names)

    if selected:
        row = next((r for r in rows if r["company_name"] == selected), None)
        if row:
            render_company_detail(row, db)


def render_company_detail(row: dict, db):
    cid = row["company_id"]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {row['company_name']}")
        st.markdown(f"**Domain:** [{row['canonical_domain']}](https://{row['canonical_domain']})")
        st.markdown(f"**Location:** {row.get('location') or '—'}")
        services = _safe_json_loads(row.get("services"), [])
        st.markdown(f"**Services:** {', '.join(services) if services else '—'}")

    with col2:
        st.markdown(f"**Qualification:** {_badge(row['qualification_status'])}")
        st.markdown(f"**Review status:** {_review_badge(row['review_status'])}")
        if row.get("processing_error"):
            st.error(f"Processing error: {row['processing_error']}")

    # Qualification reasons
    reasons = _safe_json_loads(row.get("qualification_reasons"), [])
    if reasons:
        with st.expander("📋 Qualification reasons", expanded=True):
            for r in reasons:
                st.markdown(f"- {r}")

    # Supporting evidence
    evidence = _safe_json_loads(row.get("supporting_evidence"), [])
    if evidence:
        with st.expander("🔗 Supporting evidence"):
            for e in evidence:
                if e.startswith("http"):
                    st.markdown(f"- [{e}]({e})")
                else:
                    st.markdown(f"- {e}")

    # Missing information
    missing = _safe_json_loads(row.get("missing_information"), [])
    if missing:
        with st.expander("❓ Missing information"):
            for m in missing:
                st.markdown(f"- {m}")

    # Relevant facts
    facts = _safe_json_loads(row.get("relevant_facts"), [])
    if facts:
        with st.expander("📄 Relevant facts"):
            for f in facts:
                st.markdown(f"- {f}")

    # Source evidence from DB
    ev_rows = db.list_source_evidence(cid)
    if ev_rows:
        with st.expander(f"🌐 Source pages ({len(ev_rows)})"):
            for ev in ev_rows:
                st.markdown(f"**[{ev['title'] or ev['url']}]({ev['url']})**")
                if ev.get("snippet"):
                    st.caption(ev["snippet"][:300])
                st.divider()

    # Human review
    st.divider()
    st.subheader("👤 Human Review")

    review_log = db.list_review_log(cid)
    if review_log:
        st.caption(f"Review history: {len(review_log)} action(s)")
        for log in review_log[-3:]:
            st.caption(
                f"  {log['action'].upper()} at {log['acted_at']}"
                + (f": {log['reviewer_note']}" if log.get("reviewer_note") else "")
            )

    col_a, col_r = st.columns(2)
    with col_a:
        note_a = st.text_input("Note (optional)", key=f"note_a_{cid}")
        if st.button("✅ Approve", key=f"approve_{cid}"):
            db.record_review(cid, "approved", note_a or None)
            st.success("Approved.")
            st.rerun()

    with col_r:
        note_r = st.text_input("Note (optional)", key=f"note_r_{cid}")
        if st.button("❌ Reject", key=f"reject_{cid}"):
            db.record_review(cid, "rejected", note_r or None)
            st.warning("Rejected.")
            st.rerun()


def render_evaluation_tab():
    st.header("📊 Evaluation")

    from config import settings
    from src.evaluation import load_eval_dataset, run_evaluation

    db = _get_database()
    samples = load_eval_dataset(settings.EVAL_DATASET_PATH)

    if not samples:
        st.warning(f"No evaluation dataset found at `{settings.EVAL_DATASET_PATH}`.")
        st.info("Create or copy a JSONL file with EvaluationSample records there.")
        return

    col1, col2 = st.columns(2)
    with col1:
        use_holdout = st.checkbox(
            "Use holdout subset only",
            value=False,
            help="Evaluate only samples marked is_holdout=True.",
        )
    with col2:
        if st.button("▶️ Run Evaluation"):
            with st.spinner("Computing metrics…"):
                report = run_evaluation(db=db, samples=samples, use_holdout_only=use_holdout)
            st.session_state["eval_report"] = report

    # Display
    report = st.session_state.get("eval_report")
    if report:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total samples", report.total_samples)
        col2.metric("Classified", report.classified)
        col3.metric("Unclassified", report.unclassified)
        col4.metric(
            "F1 (qualified)",
            f"{report.f1_score:.3f}" if report.f1_score is not None else "N/A",
        )

        col5, col6 = st.columns(2)
        col5.metric(
            "Precision (qualified)",
            f"{report.precision:.3f}" if report.precision is not None else "N/A",
        )
        col6.metric(
            "Recall (qualified)",
            f"{report.recall:.3f}" if report.recall is not None else "N/A",
        )

        if report.confusion_matrix:
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(report.confusion_matrix).T
            cm_df.index.name = "Actual \\ Predicted"
            st.dataframe(cm_df, use_container_width=False)

        if report.notes:
            st.info(report.notes)

    # Dataset preview
    with st.expander("📝 Evaluation dataset preview"):
        sample_df = pd.DataFrame([s.model_dump() for s in samples])
        st.dataframe(sample_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    settings, cfg_error = _load_config()
    if cfg_error:
        st.error(f"**Configuration error:**\n\n{cfg_error}")
        st.info("Copy `.env.example` to `.env` and fill in your API keys.")
        st.stop()

    tab_search, tab_prospects, tab_eval = st.tabs([
        "🔍 Search",
        "📋 Prospects",
        "📊 Evaluation",
    ])

    with tab_search:
        render_search_tab()

    with tab_prospects:
        render_prospects_tab()

    with tab_eval:
        render_evaluation_tab()


if __name__ == "__main__" or True:
    main()
