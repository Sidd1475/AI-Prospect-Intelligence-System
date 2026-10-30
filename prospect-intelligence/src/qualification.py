"""
src/qualification.py — Deterministic + LLM-assisted qualification rules.

Qualification logic follows the architecture plan exactly:

Hard disqualifiers (Python only, no LLM):
  D1 — Not a US entity
  D2 — Not an imaging center
  D3 — No verifiable domain
  D4 — Pipeline error (handled at the caller level)

Soft qualifiers (Python only):
  S1 — Imaging keywords present
  S2 — US location confirmed
  S3 — Independent / outpatient center
  S4 — Hospital system flag → needs_review

LLM semantic rules (only when deterministic rules are inconclusive):
  L1 — Confirm imaging center type (ambiguous services)
  L2 — Confirm US location from context (location field missing)
  L3 — Assess independence from hospital system

Final verdict:
  - D1 or D2 or D3 → not_qualified
  - S1 and S2 and S3 and not S4 → qualified
  - S4 flag → needs_review
  - Ambiguous soft qualifiers → LLM → qualified or needs_review
  - Cannot resolve → needs_review

Design:
  - No unexplained numeric score.
  - LLM used only when deterministic rules are inconclusive.
  - Every verdict backed by at least one reason.
  - Hospital-affiliated centers default to needs_review (not not_qualified).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.schemas import ExtractedCompany, QualificationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — keyword lists
# ---------------------------------------------------------------------------

IMAGING_KEYWORDS = frozenset({
    "mri", "ct scan", "ct", "pet scan", "pet", "x-ray", "xray",
    "ultrasound", "mammography", "mammogram", "radiology", "radiologist",
    "imaging", "diagnostic imaging", "nuclear medicine", "fluoroscopy",
    "dexa", "bone density", "echocardiogram", "echo", "interventional radiology",
    "angiography", "biopsy imaging", "scan", "scanner",
})

OUTPATIENT_KEYWORDS = frozenset({
    "outpatient", "independent imaging", "freestanding", "imaging center",
    "diagnostic center", "imaging clinic", "imaging suite", "stand-alone",
    "standalone", "walk-in imaging",
})

HOSPITAL_KEYWORDS = frozenset({
    "hospital", "health system", "medical center", "university hospital",
    "university medical", "regional medical center", "memorial hospital",
    "general hospital", "academic medical", "teaching hospital",
    "integrated health", "health network",
})

# Full state names — unambiguous
_US_STATE_NAMES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
})

# Two-letter abbreviations — only checked in the location field (too short for safe corpus match)
# NOTE: "ca" is intentionally excluded (ambiguous with Canada) and "in" (ambiguous with preposition)
_US_STATE_ABBREVS = frozenset({
    "al", "ak", "az", "ar", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
})

# Explicit non-US terms that override a weak US match
_NON_US_TERMS = frozenset({
    "canada", "ontario", "british columbia", "alberta", "quebec",
    "uk", "united kingdom", "england", "australia", "india", "china",
    "germany", "france", "mexico", "brazil",
})

# Strong US identifiers safe to check in any corpus
_US_STRONG = frozenset({"usa", "u.s.", "u.s.a.", "united states"})
US_STATES = _US_STATE_NAMES | _US_STATE_ABBREVS | _US_STRONG


# ---------------------------------------------------------------------------
# Keyword matching helpers
# ---------------------------------------------------------------------------

def _text_corpus(company: ExtractedCompany) -> str:
    """Build a single lowercased string from all company text fields."""
    parts = [
        company.company_name or "",
        company.location or "",
        " ".join(company.services),
        " ".join(company.relevant_facts),
        company.extraction_notes or "",
    ]
    return " ".join(parts).lower()


def _has_imaging_keyword(corpus: str) -> bool:
    return any(re.search(r"\b" + re.escape(kw) + r"\b", corpus) for kw in IMAGING_KEYWORDS)


def _has_us_indicator(corpus: str, location: Optional[str]) -> bool:
    """Check location field and full corpus for US indicators.

    Strategy:
    1. If the text contains explicit non-US terms, return False immediately.
    2. Check location field for full state names and two-letter abbreviations.
    3. Check full corpus only for unambiguous full state names and strong markers.
       (Two-letter abbreviations are NOT checked in the corpus to avoid false
       positives like 'or', 'in', 'me', etc.)
    """
    loc = (location or "").lower()
    full_text = (corpus + " " + loc).lower()

    # Explicit non-US exclusion — if any non-US term appears, reject
    for term in _NON_US_TERMS:
        if re.search(r"\b" + re.escape(term) + r"\b", full_text):
            return False

    # Strong global indicators (safe anywhere)
    for term in _US_STRONG:
        if term in full_text:
            return True

    # Full state names — safe to check in location AND corpus
    for state in _US_STATE_NAMES:
        if re.search(r"\b" + re.escape(state) + r"\b", full_text):
            return True

    # Two-letter abbreviations — only check in the structured location field
    for abbrev in _US_STATE_ABBREVS:
        if re.search(r"\b" + re.escape(abbrev) + r"\b", loc):
            return True

    return False


def _has_outpatient_indicator(corpus: str) -> bool:
    return any(kw in corpus for kw in OUTPATIENT_KEYWORDS)


def _has_hospital_flag(corpus: str) -> bool:
    return any(kw in corpus for kw in HOSPITAL_KEYWORDS)


# ---------------------------------------------------------------------------
# LLM semantic judgment prompts
# ---------------------------------------------------------------------------

_SEMANTIC_SYSTEM = """\
You are a business analyst helping classify US outpatient medical imaging centers as sales prospects.

Answer only with a JSON object. Do not add explanations outside the JSON.
"""

def _build_semantic_prompt(company: ExtractedCompany, questions: list[str]) -> str:
    facts = "\n".join(f"- {f}" for f in company.relevant_facts) or "  (none provided)"
    services = ", ".join(company.services) or "(none listed)"
    return f"""\
Company name: {company.company_name}
Domain: {company.canonical_domain}
Location: {company.location or '(not specified)'}
Services listed: {services}
Relevant facts:
{facts}
Extraction notes: {company.extraction_notes or '(none)'}

Questions to answer:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(questions))}

Respond with a JSON object with one key per question number (as a string), each containing:
  - "answer": true | false | "unknown"
  - "rationale": one sentence explaining your answer based only on the facts above

Example:
{{
  "1": {{"answer": true, "rationale": "The snippets mention MRI and CT services."}},
  "2": {{"answer": "unknown", "rationale": "No US state or city was mentioned."}}
}}
"""


def _call_llm_semantic(
    company: ExtractedCompany,
    questions: list[str],
    llm,
) -> dict:
    """Call LLM for semantic judgment. Returns dict of question answers."""
    try:
        prompt = _build_semantic_prompt(company, questions)
        result = llm.complete_json(_SEMANTIC_SYSTEM, prompt)
        return result
    except Exception as exc:
        logger.warning("Semantic LLM call failed for %s: %s", company.canonical_domain, exc)
        return {}


def _llm_answer(llm_result: dict, question_num: int) -> tuple[Optional[bool], str]:
    """Extract answer and rationale from LLM result dict."""
    key = str(question_num)
    item = llm_result.get(key, {})
    answer = item.get("answer")
    rationale = item.get("rationale", "No rationale provided.")
    if answer is True or answer == "true":
        return True, rationale
    elif answer is False or answer == "false":
        return False, rationale
    else:
        return None, rationale  # "unknown"


# ---------------------------------------------------------------------------
# Main qualifier
# ---------------------------------------------------------------------------

class Qualifier:
    """
    Applies qualification rules to an ExtractedCompany and returns a
    QualificationResult.

    Args:
        llm: Optional LLMClient for semantic judgment. If None, semantic
             rules are skipped and ambiguous cases become needs_review.
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def qualify(self, company: ExtractedCompany) -> QualificationResult:
        """
        Run all qualification rules and return a QualificationResult.
        Rules are applied in order; the first hard disqualifier short-circuits.
        """
        reasons: list[str] = []
        evidence: list[str] = []
        missing: list[str] = []
        used_llm = False
        llm_rationale: Optional[str] = None

        corpus = _text_corpus(company)
        source_urls = [str(u) for u in company.source_urls]

        # ------------------------------------------------------------------
        # D3 — No verifiable domain
        # ------------------------------------------------------------------
        if not company.canonical_domain or company.extraction_confidence == "low" and not source_urls:
            return QualificationResult(
                qualification_status="not_qualified",
                reasons=["D3: No verifiable domain — extraction confidence is low with no source URLs."],
                supporting_evidence=[],
                missing_information=["Verifiable company website or domain"],
                used_llm_judgment=False,
            )

        # ------------------------------------------------------------------
        # D1 — US location check
        # ------------------------------------------------------------------
        has_us = _has_us_indicator(corpus, company.location)

        if not has_us:
            # Try LLM to confirm US from context  (L2)
            if self._llm and company.location is None:
                logger.info("L2: Asking LLM to confirm US location for %s", company.canonical_domain)
                llm_result = _call_llm_semantic(
                    company,
                    ["Is this company located in the United States? Base your answer only on the facts above."],
                    self._llm,
                )
                used_llm = True
                llm_ans, llm_rat = _llm_answer(llm_result, 1)
                llm_rationale = llm_rat
                if llm_ans is True:
                    has_us = True
                    reasons.append("L2: LLM confirmed US location from context.")
                    evidence.append(f"LLM rationale: {llm_rat}")
                elif llm_ans is False:
                    return QualificationResult(
                        qualification_status="not_qualified",
                        reasons=["D1: Not a US entity — LLM confirmed company is not US-based."],
                        supporting_evidence=source_urls,
                        missing_information=[],
                        used_llm_judgment=True,
                        llm_judgment_rationale=llm_rat,
                    )
                # llm_ans is None (unknown) → fall through to missing_information

            if not has_us:
                if company.location is None:
                    missing.append("US location (not found in snippets)")
                else:
                    return QualificationResult(
                        qualification_status="not_qualified",
                        reasons=[f"D1: Not a US entity — location '{company.location}' contains no US state indicator."],
                        supporting_evidence=source_urls,
                        missing_information=[],
                        used_llm_judgment=used_llm,
                        llm_judgment_rationale=llm_rationale,
                    )

        # ------------------------------------------------------------------
        # D2 — Imaging center check
        # ------------------------------------------------------------------
        has_imaging = _has_imaging_keyword(corpus)

        if not has_imaging:
            # Try LLM to confirm imaging services (L1)
            if self._llm:
                logger.info("L1: Asking LLM to confirm imaging services for %s", company.canonical_domain)
                llm_result = _call_llm_semantic(
                    company,
                    ["Does this company provide medical imaging services (MRI, CT, X-ray, ultrasound, etc.)?"],
                    self._llm,
                )
                used_llm = True
                llm_ans, llm_rat = _llm_answer(llm_result, 1)
                llm_rationale = (llm_rationale or "") + " | " + llm_rat
                if llm_ans is True:
                    has_imaging = True
                    reasons.append("L1: LLM confirmed imaging services from context.")
                    evidence.append(f"LLM rationale: {llm_rat}")
                elif llm_ans is False:
                    return QualificationResult(
                        qualification_status="not_qualified",
                        reasons=["D2: Not an imaging center — LLM confirmed no imaging services found."],
                        supporting_evidence=source_urls,
                        missing_information=[],
                        used_llm_judgment=True,
                        llm_judgment_rationale=llm_rat,
                    )
                else:
                    missing.append("Confirmation of imaging services")
            else:
                missing.append("Imaging services (no imaging keywords found in snippets)")

        if not has_imaging:
            return QualificationResult(
                qualification_status="not_qualified",
                reasons=["D2: Not an imaging center — no imaging keywords found in extracted data."],
                supporting_evidence=source_urls,
                missing_information=missing,
                used_llm_judgment=used_llm,
                llm_judgment_rationale=llm_rationale,
            )

        # ------------------------------------------------------------------
        # Soft qualifiers
        # ------------------------------------------------------------------
        has_outpatient = _has_outpatient_indicator(corpus)
        has_hospital_flag = _has_hospital_flag(corpus)

        # Build positive evidence
        if has_us:
            reasons.append(f"S2: US location confirmed — '{company.location or 'US indicator found in text'}'.")
            evidence.extend(source_urls[:2])

        if has_imaging:
            imaging_found = [kw for kw in IMAGING_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", corpus)]
            reasons.append(f"S1: Imaging services detected — {', '.join(imaging_found[:5])}.")

        if has_outpatient:
            outpatient_found = [kw for kw in OUTPATIENT_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", corpus)]
            reasons.append(f"S3: Outpatient/independent indicator found — {', '.join(outpatient_found[:3])}.")

        # ------------------------------------------------------------------
        # S4 — Hospital flag → needs_review
        # ------------------------------------------------------------------
        if has_hospital_flag:
            hospital_found = [kw for kw in HOSPITAL_KEYWORDS if re.search(r"\b" + re.escape(kw) + r"\b", corpus)]

            # Try LLM to assess independence (L3)
            if self._llm:
                logger.info("L3: Asking LLM to assess hospital independence for %s", company.canonical_domain)
                llm_result = _call_llm_semantic(
                    company,
                    ["Is this an independent/freestanding imaging center, or is it part of a hospital or large health system?"],
                    self._llm,
                )
                used_llm = True
                llm_ans, llm_rat = _llm_answer(llm_result, 1)
                llm_rationale = (llm_rationale or "") + " | " + llm_rat
                evidence.append(f"LLM rationale: {llm_rat}")

                if llm_ans is True:
                    # LLM says "independent" — clear the hospital flag and qualify
                    reasons.append("L3: LLM assessed company as independent/freestanding despite keyword match.")
                    has_hospital_flag = False
                else:
                    reasons.append(
                        f"S4: Hospital system flag — keywords found: {', '.join(hospital_found[:3])}. "
                        "Routed to needs_review."
                    )
                    return QualificationResult(
                        qualification_status="needs_review",
                        reasons=reasons,
                        supporting_evidence=evidence or source_urls,
                        missing_information=missing or ["Confirmation of independent vs. hospital-affiliated status"],
                        used_llm_judgment=used_llm,
                        llm_judgment_rationale=llm_rationale,
                    )
            else:
                reasons.append(
                    f"S4: Hospital system flag — keywords found: {', '.join(hospital_found[:3])}. "
                    "Routed to needs_review."
                )
                return QualificationResult(
                    qualification_status="needs_review",
                    reasons=reasons,
                    supporting_evidence=evidence or source_urls,
                    missing_information=missing or ["Confirmation of independent vs. hospital-affiliated status"],
                    used_llm_judgment=used_llm,
                    llm_judgment_rationale=llm_rationale,
                )

        # ------------------------------------------------------------------
        # Final verdict — all hard disqualifiers passed
        # ------------------------------------------------------------------
        if has_imaging and has_us:
            if has_outpatient:
                status = "qualified"
                reasons.append("All key qualification signals present: imaging services, US location, outpatient indicator.")
            elif missing:
                # Some info was missing but not definitively disqualifying
                status = "needs_review"
                reasons.append("Qualifies on imaging + US signals but missing information requires review.")
            else:
                # Imaging + US confirmed, but no explicit outpatient signal
                status = "needs_review"
                reasons.append(
                    "Imaging services and US location confirmed, but outpatient/independent status "
                    "could not be confirmed from available snippets."
                )
                missing.append("Confirmation of outpatient/independent status")
        elif has_imaging and not has_us:
            status = "needs_review"
            reasons.append("Imaging services confirmed but US location could not be verified.")
        else:
            # Should not normally reach here — covered by D2 above
            status = "not_qualified"
            reasons.append("Insufficient evidence to qualify as a US imaging center.")

        return QualificationResult(
            qualification_status=status,
            reasons=reasons,
            supporting_evidence=evidence or source_urls,
            missing_information=missing,
            used_llm_judgment=used_llm,
            llm_judgment_rationale=llm_rationale if used_llm else None,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def qualify_company(company: ExtractedCompany, llm=None) -> QualificationResult:
    """Module-level convenience wrapper."""
    return Qualifier(llm=llm).qualify(company)
