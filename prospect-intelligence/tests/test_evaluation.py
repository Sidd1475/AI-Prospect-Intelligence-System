"""
tests/test_evaluation.py — Evaluation metrics tests.

All tests use in-memory database and constructed data.
No API keys or network access required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.database import Database
from src.evaluation import (
    compute_metrics,
    load_eval_dataset,
    run_evaluation,
    POSITIVE_CLASS,
)
from src.schemas import CompanyRecord, EvaluationSample


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db() -> Database:
    return Database(":memory:")


def _insert_company(db: Database, domain: str, status: str) -> str:
    record = CompanyRecord(
        company_name=domain,
        canonical_domain=domain,
        source_urls=[f"https://{domain}"],
        qualification_status=status,
        qualification_reasons=["Test reason"],
    )
    return db.upsert_company(record)


def _make_sample(domain: str, label: str, holdout: bool = False) -> EvaluationSample:
    return EvaluationSample(
        sample_id=f"eval-{domain}",
        company_name=domain,
        canonical_domain=domain,
        reference_label=label,
        label_rationale="Test rationale.",
        is_holdout=holdout,
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_precision_recall_f1(self):
        y_true = ["qualified", "qualified", "not_qualified"]
        y_pred = ["qualified", "qualified", "not_qualified"]
        p, r, f1, _ = compute_metrics(y_true, y_pred)
        assert p == 1.0
        assert r == 1.0
        assert f1 == 1.0

    def test_all_wrong(self):
        y_true = ["qualified", "qualified"]
        y_pred = ["not_qualified", "not_qualified"]
        p, r, f1, _ = compute_metrics(y_true, y_pred)
        assert p is None    # no positive predictions → zero denominator
        assert r == 0.0
        assert f1 is None

    def test_no_true_positives(self):
        y_true = ["not_qualified", "not_qualified"]
        y_pred = ["qualified", "qualified"]
        p, r, f1, _ = compute_metrics(y_true, y_pred)
        assert p == 0.0     # tp/(tp+fp) = 0/2
        assert r is None    # no actual positives → zero denominator
        assert f1 is None

    def test_zero_division_safe(self):
        # Empty lists
        p, r, f1, cm = compute_metrics([], [])
        assert p is None
        assert r is None
        assert f1 is None

    def test_partial_correct(self):
        y_true = ["qualified", "qualified", "not_qualified", "not_qualified"]
        y_pred = ["qualified", "not_qualified", "qualified", "not_qualified"]
        p, r, f1, cm = compute_metrics(y_true, y_pred)
        # tp=1, fp=1, fn=1
        assert p == 0.5
        assert r == 0.5
        assert f1 == 0.5

    def test_confusion_matrix_shape(self):
        y_true = ["qualified", "not_qualified", "needs_review"]
        y_pred = ["qualified", "qualified", "needs_review"]
        _, _, _, cm = compute_metrics(y_true, y_pred)
        assert "qualified" in cm
        assert "not_qualified" in cm
        assert "needs_review" in cm
        assert cm["qualified"]["qualified"] == 1
        assert cm["not_qualified"]["qualified"] == 1  # misclassified

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_metrics(["qualified"], ["qualified", "not_qualified"])


# ---------------------------------------------------------------------------
# load_eval_dataset
# ---------------------------------------------------------------------------

class TestLoadEvalDataset:
    def test_loads_valid_jsonl(self, tmp_path: Path):
        f = tmp_path / "eval.jsonl"
        sample = {
            "sample_id": "eval-001",
            "company_name": "Test Imaging",
            "canonical_domain": "test.com",
            "reference_label": "qualified",
            "label_rationale": "Test imaging center.",
            "is_holdout": False,
        }
        f.write_text(json.dumps(sample) + "\n")
        samples = load_eval_dataset(f)
        assert len(samples) == 1
        assert samples[0].sample_id == "eval-001"

    def test_returns_empty_for_missing_file(self, tmp_path: Path):
        samples = load_eval_dataset(tmp_path / "nonexistent.jsonl")
        assert samples == []

    def test_skips_invalid_lines(self, tmp_path: Path):
        f = tmp_path / "eval.jsonl"
        valid = json.dumps({
            "sample_id": "eval-001",
            "company_name": "Test",
            "canonical_domain": "test.com",
            "reference_label": "qualified",
            "label_rationale": "Valid.",
        })
        f.write_text(valid + "\n" + "not valid json {\n")
        samples = load_eval_dataset(f)
        assert len(samples) == 1

    def test_skips_comment_lines(self, tmp_path: Path):
        f = tmp_path / "eval.jsonl"
        valid = json.dumps({
            "sample_id": "eval-001",
            "company_name": "Test",
            "canonical_domain": "test.com",
            "reference_label": "qualified",
            "label_rationale": "Valid.",
        })
        f.write_text("# This is a comment\n" + valid + "\n")
        samples = load_eval_dataset(f)
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_empty_samples_returns_zero_report(self, db: Database):
        report = run_evaluation(db=db, samples=[])
        assert report.total_samples == 0
        assert report.classified == 0
        assert report.unclassified == 0

    def test_all_classified_correctly(self, db: Database):
        _insert_company(db, "a.com", "qualified")
        _insert_company(db, "b.com", "not_qualified")
        samples = [
            _make_sample("a.com", "qualified"),
            _make_sample("b.com", "not_qualified"),
        ]
        report = run_evaluation(db=db, samples=samples)
        assert report.total_samples == 2
        assert report.classified == 2
        assert report.unclassified == 0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0

    def test_missing_from_db_counted_as_unclassified(self, db: Database):
        samples = [_make_sample("notindb.com", "qualified")]
        report = run_evaluation(db=db, samples=samples)
        assert report.unclassified == 1
        assert report.classified == 0

    def test_error_records_counted_as_unclassified(self, db: Database):
        err_record = CompanyRecord.error_record(
            company_name="Broken",
            canonical_domain="broken.com",
            error_message="Timeout",
        )
        db.upsert_company(err_record)
        samples = [_make_sample("broken.com", "qualified")]
        report = run_evaluation(db=db, samples=samples)
        assert report.unclassified == 1

    def test_holdout_filter(self, db: Database):
        _insert_company(db, "a.com", "qualified")
        _insert_company(db, "b.com", "not_qualified")
        samples = [
            _make_sample("a.com", "qualified", holdout=False),
            _make_sample("b.com", "not_qualified", holdout=True),
        ]
        report = run_evaluation(db=db, samples=samples, use_holdout_only=True)
        assert report.total_samples == 1
        assert report.classified == 1

    def test_no_cost_data_returns_none(self, db: Database):
        _insert_company(db, "a.com", "qualified")
        samples = [_make_sample("a.com", "qualified")]
        report = run_evaluation(db=db, samples=samples)
        assert report.estimated_api_cost_usd is None
        assert report.cost_per_accepted_prospect_usd is None

    def test_counts_consistent(self, db: Database):
        _insert_company(db, "a.com", "qualified")
        samples = [
            _make_sample("a.com", "qualified"),
            _make_sample("missing.com", "qualified"),
        ]
        report = run_evaluation(db=db, samples=samples)
        assert report.classified + report.unclassified == report.total_samples
