"""
tests/test_database.py — SQLite persistence layer tests.

All tests use an in-memory database (:memory:) so no files are created.
No API keys or network access required.
"""

from __future__ import annotations

import pytest

from src.database import Database
from src.schemas import CompanyRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    """Fresh in-memory database for each test."""
    return Database(":memory:")


def _make_record(
    name: str = "Acme Imaging",
    domain: str = "acmeimaging.com",
    status: str = "qualified",
) -> CompanyRecord:
    return CompanyRecord(
        company_name=name,
        canonical_domain=domain,
        source_urls=["https://" + domain],
        qualification_status=status,
        qualification_reasons=["MRI services present", "US location confirmed"],
        location="Dallas, TX",
        services=["MRI", "CT Scan"],
        relevant_facts=["Outpatient center"],
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------

class TestBasicCRUD:
    def test_insert_and_retrieve_by_domain(self, db: Database):
        record = _make_record()
        db.upsert_company(record)

        row = db.get_company_by_domain("acmeimaging.com")
        assert row is not None
        assert row["company_name"] == "Acme Imaging"

    def test_insert_and_retrieve_by_id(self, db: Database):
        record = _make_record()
        company_id = db.upsert_company(record)

        row = db.get_company_by_id(company_id)
        assert row is not None
        assert row["canonical_domain"] == "acmeimaging.com"

    def test_list_companies_empty(self, db: Database):
        assert db.list_companies() == []

    def test_list_companies_returns_inserted(self, db: Database):
        db.upsert_company(_make_record("Acme", "acme.com"))
        db.upsert_company(_make_record("Beta", "beta.com"))
        companies = db.list_companies()
        assert len(companies) == 2

    def test_get_company_count(self, db: Database):
        assert db.get_company_count() == 0
        db.upsert_company(_make_record())
        assert db.get_company_count() == 1

    def test_get_nonexistent_domain_returns_none(self, db: Database):
        assert db.get_company_by_domain("nope.com") is None

    def test_get_nonexistent_id_returns_none(self, db: Database):
        assert db.get_company_by_id("00000000-0000-0000-0000-000000000000") is None


# ---------------------------------------------------------------------------
# Deduplication (upsert semantics)
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_domain_does_not_create_second_row(self, db: Database):
        db.upsert_company(_make_record("Acme v1", "acme.com"))
        db.upsert_company(_make_record("Acme v2", "acme.com"))
        assert db.get_company_count() == 1

    def test_upsert_updates_mutable_fields(self, db: Database):
        db.upsert_company(_make_record("Old Name", "acme.com", "qualified"))
        db.upsert_company(_make_record("New Name", "acme.com", "not_qualified"))

        row = db.get_company_by_domain("acme.com")
        assert row["company_name"] == "New Name"
        assert row["qualification_status"] == "not_qualified"

    def test_upsert_preserves_company_id(self, db: Database):
        record_v1 = _make_record("Acme", "acme.com")
        original_id = db.upsert_company(record_v1)

        record_v2 = _make_record("Acme Updated", "acme.com")
        returned_id = db.upsert_company(record_v2)

        assert original_id == returned_id

    def test_upsert_preserves_created_at(self, db: Database):
        db.upsert_company(_make_record("Acme", "acme.com"))
        row_before = db.get_company_by_domain("acme.com")
        created_before = row_before["created_at"]

        db.upsert_company(_make_record("Acme Updated", "acme.com"))
        row_after = db.get_company_by_domain("acme.com")

        assert row_after["created_at"] == created_before

    def test_two_different_domains_both_stored(self, db: Database):
        db.upsert_company(_make_record("Acme", "acme.com"))
        db.upsert_company(_make_record("Beta", "beta.com"))
        assert db.get_company_count() == 2

    def test_idempotent_same_record(self, db: Database):
        record = _make_record()
        db.upsert_company(record)
        db.upsert_company(record)
        assert db.get_company_count() == 1


# ---------------------------------------------------------------------------
# Source evidence
# ---------------------------------------------------------------------------

class TestSourceEvidence:
    def test_add_evidence_and_retrieve(self, db: Database):
        record = _make_record()
        company_id = db.upsert_company(record)

        db.add_source_evidence(
            company_id=company_id,
            url="https://acmeimaging.com/about",
            title="About Us",
            snippet="Outpatient MRI center in Dallas.",
        )

        evidence = db.list_source_evidence(company_id)
        assert len(evidence) == 1
        assert evidence[0]["url"] == "https://acmeimaging.com/about"

    def test_no_evidence_returns_empty_list(self, db: Database):
        record = _make_record()
        company_id = db.upsert_company(record)
        assert db.list_source_evidence(company_id) == []

    def test_multiple_evidence_rows(self, db: Database):
        record = _make_record()
        cid = db.upsert_company(record)
        db.add_source_evidence(cid, "https://acme.com/page1", "P1", "Snippet 1")
        db.add_source_evidence(cid, "https://acme.com/page2", "P2", "Snippet 2")
        assert len(db.list_source_evidence(cid)) == 2


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------

class TestHumanReview:
    def test_approve_updates_review_status(self, db: Database):
        cid = db.upsert_company(_make_record())
        db.record_review(cid, "approved")

        row = db.get_company_by_id(cid)
        assert row["review_status"] == "approved"

    def test_reject_updates_review_status(self, db: Database):
        cid = db.upsert_company(_make_record())
        db.record_review(cid, "rejected")

        row = db.get_company_by_id(cid)
        assert row["review_status"] == "rejected"

    def test_review_log_entry_created(self, db: Database):
        cid = db.upsert_company(_make_record())
        db.record_review(cid, "approved", reviewer_note="Verified manually.")

        log = db.list_review_log(cid)
        assert len(log) == 1
        assert log[0]["action"] == "approved"
        assert log[0]["reviewer_note"] == "Verified manually."

    def test_multiple_review_actions_logged(self, db: Database):
        cid = db.upsert_company(_make_record())
        db.record_review(cid, "rejected")
        db.record_review(cid, "approved")

        log = db.list_review_log(cid)
        assert len(log) == 2

    def test_unknown_action_raises(self, db: Database):
        cid = db.upsert_company(_make_record())
        with pytest.raises(ValueError, match="Unknown review action"):
            db.record_review(cid, "maybe")

    def test_default_review_status_is_pending(self, db: Database):
        cid = db.upsert_company(_make_record())
        row = db.get_company_by_id(cid)
        assert row["review_status"] == "pending"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestListFiltering:
    def test_filter_by_qualification_status(self, db: Database):
        db.upsert_company(_make_record("A", "a.com", "qualified"))
        db.upsert_company(_make_record("B", "b.com", "not_qualified"))
        db.upsert_company(_make_record("C", "c.com", "needs_review"))

        qualified = db.list_companies(qualification_status="qualified")
        assert len(qualified) == 1
        assert qualified[0]["company_name"] == "A"

    def test_filter_by_review_status(self, db: Database):
        cid1 = db.upsert_company(_make_record("A", "a.com"))
        cid2 = db.upsert_company(_make_record("B", "b.com"))
        db.record_review(cid1, "approved")

        approved = db.list_companies(review_status="approved")
        assert len(approved) == 1

        pending = db.list_companies(review_status="pending")
        assert len(pending) == 1


# ---------------------------------------------------------------------------
# Evaluation runs
# ---------------------------------------------------------------------------

class TestEvaluationRuns:
    def test_save_and_retrieve_latest(self, db: Database):
        db.save_evaluation_run(
            total_samples=10,
            classified=8,
            unclassified=2,
            precision=0.75,
            recall=0.60,
            f1_score=0.67,
            confusion_matrix={"qualified": {"qualified": 5, "not_qualified": 1}},
            estimated_cost_usd=0.12,
            notes="Initial run.",
        )

        run = db.latest_evaluation_run()
        assert run is not None
        assert run["total_samples"] == 10
        assert run["precision"] == 0.75

    def test_no_evaluation_run_returns_none(self, db: Database):
        assert db.latest_evaluation_run() is None


# ---------------------------------------------------------------------------
# Error records
# ---------------------------------------------------------------------------

class TestErrorRecords:
    def test_error_record_stored(self, db: Database):
        error_rec = CompanyRecord.error_record(
            company_name="Broken Co",
            canonical_domain="broken.com",
            error_message="Timeout during LLM call.",
        )
        cid = db.upsert_company(error_rec)
        row = db.get_company_by_id(cid)
        assert row["qualification_status"] == "error"
        assert row["processing_error"] == "Timeout during LLM call."
