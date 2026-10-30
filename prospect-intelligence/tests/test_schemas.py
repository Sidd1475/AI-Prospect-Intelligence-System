"""
tests/test_schemas.py — Pydantic schema validation tests.

These tests require no API keys or network access.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    CompanyRecord,
    EvaluationReport,
    EvaluationSample,
    ExtractedCompany,
    QualificationResult,
    SearchResult,
)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_valid(self):
        sr = SearchResult(
            title="Acme Imaging Center",
            url="https://acmeimaging.com",
            snippet="Leading MRI and CT provider in Texas.",
            source_domain="acmeimaging.com",
        )
        assert sr.source_domain == "acmeimaging.com"

    def test_domain_normalised(self):
        sr = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="A snippet.",
            source_domain="WWW.Example.COM/",
        )
        assert sr.source_domain == "example.com"

    def test_invalid_url_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(
                title="Test",
                url="not-a-url",
                snippet="snippet",
                source_domain="example.com",
            )

    def test_blank_title_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(
                title="   ",
                url="https://example.com",
                snippet="snippet",
                source_domain="example.com",
            )

    def test_blank_snippet_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(
                title="Title",
                url="https://example.com",
                snippet="",
                source_domain="example.com",
            )


# ---------------------------------------------------------------------------
# ExtractedCompany
# ---------------------------------------------------------------------------

class TestExtractedCompany:
    def _valid_kwargs(self) -> dict:
        return dict(
            company_name="Acme Imaging",
            canonical_domain="acmeimaging.com",
            location="Dallas, TX",
            services=["MRI", "CT Scan"],
            relevant_facts=["Outpatient center", "Founded 2005"],
            source_urls=["https://acmeimaging.com/about"],
            extraction_confidence="high",
        )

    def test_valid(self):
        ec = ExtractedCompany(**self._valid_kwargs())
        assert ec.company_name == "Acme Imaging"
        assert ec.canonical_domain == "acmeimaging.com"

    def test_domain_normalised(self):
        kw = self._valid_kwargs()
        kw["canonical_domain"] = "https://www.AcmeImaging.com/"
        ec = ExtractedCompany(**kw)
        assert ec.canonical_domain == "acmeimaging.com"

    def test_missing_source_urls_raises(self):
        kw = self._valid_kwargs()
        kw["source_urls"] = []
        with pytest.raises(ValidationError, match="At least one source URL"):
            ExtractedCompany(**kw)

    def test_invalid_url_in_source_urls_raises(self):
        kw = self._valid_kwargs()
        kw["source_urls"] = ["not-a-url"]
        with pytest.raises(ValidationError):
            ExtractedCompany(**kw)

    def test_blank_company_name_raises(self):
        kw = self._valid_kwargs()
        kw["company_name"] = "   "
        with pytest.raises(ValidationError, match="must not be blank"):
            ExtractedCompany(**kw)

    def test_invalid_confidence_raises(self):
        kw = self._valid_kwargs()
        kw["extraction_confidence"] = "very_high"
        with pytest.raises(ValidationError):
            ExtractedCompany(**kw)

    def test_empty_services_allowed(self):
        kw = self._valid_kwargs()
        kw["services"] = []
        ec = ExtractedCompany(**kw)
        assert ec.services == []

    def test_optional_location(self):
        kw = self._valid_kwargs()
        del kw["location"]
        ec = ExtractedCompany(**kw)
        assert ec.location is None


# ---------------------------------------------------------------------------
# QualificationResult
# ---------------------------------------------------------------------------

class TestQualificationResult:
    def test_valid_qualified(self):
        qr = QualificationResult(
            qualification_status="qualified",
            reasons=["US location confirmed", "MRI services present"],
            supporting_evidence=["https://acmeimaging.com/services"],
        )
        assert qr.qualification_status == "qualified"

    def test_valid_not_qualified(self):
        qr = QualificationResult(
            qualification_status="not_qualified",
            reasons=["No imaging keywords found"],
        )
        assert qr.qualification_status == "not_qualified"

    def test_valid_needs_review(self):
        qr = QualificationResult(
            qualification_status="needs_review",
            reasons=["Hospital affiliation flag"],
        )
        assert qr.used_llm_judgment is False

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            QualificationResult(
                qualification_status="maybe",
                reasons=["some reason"],
            )

    def test_empty_reasons_raises(self):
        with pytest.raises(ValidationError, match="At least one reason"):
            QualificationResult(
                qualification_status="qualified",
                reasons=[],
            )

    def test_llm_judgment_fields(self):
        qr = QualificationResult(
            qualification_status="qualified",
            reasons=["LLM confirmed imaging services"],
            used_llm_judgment=True,
            llm_judgment_rationale="Company name includes 'radiology'.",
        )
        assert qr.used_llm_judgment is True
        assert qr.llm_judgment_rationale is not None


# ---------------------------------------------------------------------------
# CompanyRecord
# ---------------------------------------------------------------------------

class TestCompanyRecord:
    def _valid_kwargs(self) -> dict:
        return dict(
            company_name="Acme Imaging",
            canonical_domain="acmeimaging.com",
            source_urls=["https://acmeimaging.com"],
            qualification_status="qualified",
            qualification_reasons=["MRI services present", "US location confirmed"],
        )

    def test_valid(self):
        cr = CompanyRecord(**self._valid_kwargs())
        assert cr.review_status == "pending"
        assert cr.company_id  # auto-generated UUID

    def test_domain_normalised(self):
        kw = self._valid_kwargs()
        kw["canonical_domain"] = "www.AcmeImaging.com"
        cr = CompanyRecord(**kw)
        assert cr.canonical_domain == "acmeimaging.com"

    def test_invalid_review_status_raises(self):
        kw = self._valid_kwargs()
        kw["review_status"] = "maybe"
        with pytest.raises(ValidationError):
            CompanyRecord(**kw)

    def test_invalid_qualification_status_raises(self):
        kw = self._valid_kwargs()
        kw["qualification_status"] = "unknown"
        with pytest.raises(ValidationError):
            CompanyRecord(**kw)

    def test_missing_source_urls_non_error_raises(self):
        kw = self._valid_kwargs()
        kw["source_urls"] = []
        with pytest.raises(ValidationError):
            CompanyRecord(**kw)

    def test_error_record_allows_no_urls(self):
        cr = CompanyRecord.error_record(
            company_name="Unknown",
            canonical_domain="unknown.com",
            error_message="Timeout during extraction.",
        )
        assert cr.qualification_status == "error"
        assert cr.source_urls == []
        assert cr.processing_error == "Timeout during extraction."

    def test_from_extraction_and_qualification(self):
        from src.schemas import ExtractedCompany, QualificationResult
        ec = ExtractedCompany(
            company_name="Test Imaging",
            canonical_domain="testimaging.com",
            source_urls=["https://testimaging.com"],
            extraction_confidence="medium",
        )
        qr = QualificationResult(
            qualification_status="qualified",
            reasons=["US imaging center"],
        )
        cr = CompanyRecord.from_extraction_and_qualification(ec, qr)
        assert cr.company_name == "Test Imaging"
        assert cr.qualification_status == "qualified"


# ---------------------------------------------------------------------------
# EvaluationSample
# ---------------------------------------------------------------------------

class TestEvaluationSample:
    def test_valid(self):
        es = EvaluationSample(
            sample_id="eval-001",
            company_name="Acme Imaging",
            canonical_domain="acmeimaging.com",
            reference_label="qualified",
            label_rationale="Outpatient MRI center in Texas.",
        )
        assert es.reference_label == "qualified"
        assert es.is_holdout is False

    def test_invalid_label_raises(self):
        with pytest.raises(ValidationError):
            EvaluationSample(
                sample_id="eval-002",
                company_name="X",
                canonical_domain="x.com",
                reference_label="maybe",
                label_rationale="Some rationale.",
            )

    def test_blank_rationale_raises(self):
        with pytest.raises(ValidationError, match="must not be blank"):
            EvaluationSample(
                sample_id="eval-003",
                company_name="X",
                canonical_domain="x.com",
                reference_label="qualified",
                label_rationale="   ",
            )


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_valid_all_classified(self):
        er = EvaluationReport(
            total_samples=10,
            classified=10,
            unclassified=0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
        )
        assert er.f1_score == 1.0

    def test_counts_inconsistent_raises(self):
        with pytest.raises(ValidationError, match="must equal total_samples"):
            EvaluationReport(
                total_samples=10,
                classified=8,
                unclassified=1,  # 8+1 != 10
            )

    def test_optional_metrics_none(self):
        er = EvaluationReport(
            total_samples=5,
            classified=3,
            unclassified=2,
        )
        assert er.precision is None
        assert er.cost_per_accepted_prospect_usd is None
