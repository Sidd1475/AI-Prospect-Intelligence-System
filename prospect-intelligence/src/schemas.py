"""
src/schemas.py — Pydantic models for the prospect intelligence pipeline.

All external data (LLM output, search results, database rows) passes through
one of these models before being used anywhere else in the pipeline.

Design notes:
- Use Pydantic v2 model_validator / field_validator for cross-field checks.
- Literal types enforce enumerated values at the edge, not in business logic.
- HttpUrl fields validate URL structure; they do NOT make network requests.
- list[str] fields default to [] so that partial LLM output is tolerated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


# ---------------------------------------------------------------------------
# 1. Search layer
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """One result item returned by Tavily."""

    title: str
    url: HttpUrl
    snippet: str
    source_domain: str          # normalised: lowercase, no www., no protocol
    retrieved_at: datetime = Field(default_factory=_utcnow)

    @field_validator("source_domain")
    @classmethod
    def normalise_domain(cls, v: str) -> str:
        v = v.lower().strip()
        for prefix in ("https://", "http://", "www."):
            v = v.removeprefix(prefix)
        return v.rstrip("/")

    @field_validator("title", "snippet")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field must not be blank.")
        return v.strip()


# ---------------------------------------------------------------------------
# 2. Researcher / extraction layer
# ---------------------------------------------------------------------------

class ExtractedCompany(BaseModel):
    """
    Structured company information extracted by the LLM from search results.
    This is the *raw* extraction before qualification rules are applied.
    """

    company_name: str
    canonical_domain: str           # e.g. "acmeimaging.com" — no protocol
    location: Optional[str] = None  # city + state, or full address
    services: list[str] = Field(default_factory=list)
    relevant_facts: list[str] = Field(default_factory=list)
    source_urls: list[HttpUrl]      # at least one required — validated below
    extraction_confidence: Literal["high", "medium", "low"]
    extraction_notes: Optional[str] = None  # LLM caveats or reasoning

    @field_validator("canonical_domain")
    @classmethod
    def normalise_domain(cls, v: str) -> str:
        v = v.lower().strip()
        for prefix in ("https://", "http://", "www."):
            v = v.removeprefix(prefix)
        return v.rstrip("/")

    @field_validator("source_urls")
    @classmethod
    def at_least_one_url(cls, v: list[HttpUrl]) -> list[HttpUrl]:
        if not v:
            raise ValueError("At least one source URL is required.")
        return v

    @field_validator("company_name")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("company_name must not be blank.")
        return v.strip()


# ---------------------------------------------------------------------------
# 3. Qualification layer
# ---------------------------------------------------------------------------

class QualificationResult(BaseModel):
    """
    The output of the qualification module for a single company.
    Every decision is backed by explicit reasons and evidence.
    """

    qualification_status: Literal["qualified", "not_qualified", "needs_review"]
    reasons: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    used_llm_judgment: bool = False
    llm_judgment_rationale: Optional[str] = None

    @field_validator("reasons")
    @classmethod
    def at_least_one_reason(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "At least one reason is required for every qualification result."
            )
        return v


# ---------------------------------------------------------------------------
# 4. Full stored record (combines extraction + qualification)
# ---------------------------------------------------------------------------

class CompanyRecord(BaseModel):
    """
    The canonical record persisted to SQLite and displayed in the UI.
    Created by merging ExtractedCompany + QualificationResult.
    """

    company_id: str = Field(default_factory=_new_id)
    company_name: str
    canonical_domain: str
    location: Optional[str] = None
    services: list[str] = Field(default_factory=list)
    relevant_facts: list[str] = Field(default_factory=list)
    source_urls: list[HttpUrl]
    retrieved_at: datetime = Field(default_factory=_utcnow)

    # Qualification
    qualification_status: Literal[
        "qualified", "not_qualified", "needs_review", "error"
    ]
    qualification_reasons: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    used_llm_judgment: bool = False

    # Human review
    review_status: Literal["pending", "approved", "rejected"] = "pending"

    # Error handling
    processing_error: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("canonical_domain")
    @classmethod
    def normalise_domain(cls, v: str) -> str:
        v = v.lower().strip()
        for prefix in ("https://", "http://", "www."):
            v = v.removeprefix(prefix)
        return v.rstrip("/")

    @model_validator(mode="after")
    def error_status_allows_empty_fields(self) -> "CompanyRecord":
        """
        When status is 'error', we relax the source_urls requirement
        because the pipeline may have failed before extraction completed.
        For all other statuses at least one source URL is expected.
        """
        if self.qualification_status != "error" and not self.source_urls:
            raise ValueError(
                "source_urls must contain at least one URL for non-error records."
            )
        return self

    @classmethod
    def from_extraction_and_qualification(
        cls,
        extracted: ExtractedCompany,
        qualification: QualificationResult,
    ) -> "CompanyRecord":
        """Convenience constructor — merges the two pipeline outputs."""
        return cls(
            company_name=extracted.company_name,
            canonical_domain=extracted.canonical_domain,
            location=extracted.location,
            services=extracted.services,
            relevant_facts=extracted.relevant_facts,
            source_urls=extracted.source_urls,
            retrieved_at=extracted.source_urls and _utcnow() or _utcnow(),
            qualification_status=qualification.qualification_status,
            qualification_reasons=qualification.reasons,
            supporting_evidence=qualification.supporting_evidence,
            missing_information=qualification.missing_information,
            used_llm_judgment=qualification.used_llm_judgment,
        )

    @classmethod
    def error_record(
        cls,
        company_name: str,
        canonical_domain: str,
        error_message: str,
        source_urls: Optional[list[HttpUrl]] = None,
    ) -> "CompanyRecord":
        """Create a minimal error record when the pipeline fails."""
        return cls(
            company_name=company_name,
            canonical_domain=canonical_domain,
            source_urls=source_urls or [],
            qualification_status="error",
            qualification_reasons=["Pipeline processing failed."],
            processing_error=error_message,
        )


# ---------------------------------------------------------------------------
# 5. Evaluation layer
# ---------------------------------------------------------------------------

class EvaluationSample(BaseModel):
    """One manually labelled example in the evaluation dataset."""

    sample_id: str
    company_name: str
    canonical_domain: str
    reference_label: Literal["qualified", "not_qualified", "needs_review"]
    label_rationale: str    # human explanation for this label
    is_holdout: bool = False  # True = held out from development

    @field_validator("label_rationale")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("label_rationale must not be blank.")
        return v.strip()


class EvaluationReport(BaseModel):
    """
    Metrics computed by evaluation.py.
    Fields are Optional[float] because they cannot be calculated when
    there are no true positives, or when cost data is unavailable.
    Callers must never invent values for None fields.
    """

    total_samples: int
    classified: int
    unclassified: int           # records that errored or are missing from DB

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None

    # confusion_matrix[actual][predicted] = count
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)

    estimated_api_cost_usd: Optional[float] = None
    cost_per_accepted_prospect_usd: Optional[float] = None

    notes: str = ""

    @model_validator(mode="after")
    def counts_consistent(self) -> "EvaluationReport":
        if self.classified + self.unclassified != self.total_samples:
            raise ValueError(
                f"classified ({self.classified}) + unclassified "
                f"({self.unclassified}) must equal total_samples "
                f"({self.total_samples})."
            )
        return self
