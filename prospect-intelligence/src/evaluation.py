"""
src/evaluation.py — Evaluation metrics against a manually labeled dataset.

Responsibilities:
- Load a JSONL labeled evaluation dataset.
- Compare system predictions (from SQLite) against reference labels.
- Compute precision, recall, F1 score and a full confusion matrix.
- Count unclassified (missing or errored) records.
- Estimate API cost when token usage data is available.
- Never invent results or cost figures — report None for unknown values.

Evaluation is always against the "qualified" class (one-vs-rest), which is
the primary business objective. The confusion matrix covers all three labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.database import Database
from src.schemas import EvaluationReport, EvaluationSample

logger = logging.getLogger(__name__)

# The positive class for precision/recall/F1
POSITIVE_CLASS = "qualified"

# All valid labels
ALL_LABELS = ["qualified", "not_qualified", "needs_review"]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eval_dataset(path: Path) -> list[EvaluationSample]:
    """
    Load evaluation samples from a JSON Lines file.

    Each line must be a valid JSON object matching EvaluationSample schema.
    Lines that fail validation are logged and skipped.
    Returns an empty list if the file does not exist.
    """
    if not path.exists():
        logger.warning("Evaluation dataset not found at: %s", path)
        return []

    samples: list[EvaluationSample] = []
    errors = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                samples.append(EvaluationSample(**data))
            except Exception as exc:
                logger.warning("Skipping invalid eval record at line %d: %s", lineno, exc)
                errors += 1

    logger.info(
        "Loaded %d eval samples from %s (%d line(s) skipped).",
        len(samples), path, errors,
    )
    return samples


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _build_empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {actual: {pred: 0 for pred in ALL_LABELS} for actual in ALL_LABELS}


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> tuple[Optional[float], Optional[float], Optional[float], dict]:
    """
    Compute precision, recall and F1 for the POSITIVE_CLASS label.
    Also builds a full confusion matrix for all labels.

    Returns (precision, recall, f1, confusion_matrix).
    Returns None for any metric that cannot be computed (zero denominator).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")

    confusion = _build_empty_confusion_matrix()

    tp = fp = fn = 0

    for actual, predicted in zip(y_true, y_pred):
        # Clamp unknown predicted labels
        if actual in ALL_LABELS and predicted in ALL_LABELS:
            confusion[actual][predicted] += 1

        if actual == POSITIVE_CLASS and predicted == POSITIVE_CLASS:
            tp += 1
        elif actual != POSITIVE_CLASS and predicted == POSITIVE_CLASS:
            fp += 1
        elif actual == POSITIVE_CLASS and predicted != POSITIVE_CLASS:
            fn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = None

    return precision, recall, f1, confusion


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    db: Database,
    samples: list[EvaluationSample],
    use_holdout_only: bool = False,
) -> EvaluationReport:
    """
    Compare system predictions (in SQLite) with reference labels.

    Args:
        db:               Database instance to query for predictions.
        samples:          Labeled evaluation samples.
        use_holdout_only: If True, evaluate only samples marked is_holdout=True.

    Returns:
        EvaluationReport with computed metrics.
        Metrics are None if no predictions could be compared.
    """
    if use_holdout_only:
        samples = [s for s in samples if s.is_holdout]
        if not samples:
            logger.warning("use_holdout_only=True but no samples have is_holdout=True.")

    if not samples:
        return EvaluationReport(
            total_samples=0,
            classified=0,
            unclassified=0,
            notes="No evaluation samples to compare.",
        )

    y_true: list[str] = []
    y_pred: list[str] = []
    unclassified_count = 0
    unclassified_reasons: list[str] = []

    for sample in samples:
        row = db.get_company_by_domain(sample.canonical_domain)

        if row is None:
            logger.debug("Eval sample not in DB: %s", sample.canonical_domain)
            unclassified_count += 1
            unclassified_reasons.append(
                f"{sample.canonical_domain}: not found in database"
            )
            continue

        pred_status = row.get("qualification_status", "")

        if pred_status == "error" or not pred_status:
            unclassified_count += 1
            unclassified_reasons.append(
                f"{sample.canonical_domain}: processing error in pipeline"
            )
            continue

        y_true.append(sample.reference_label)
        y_pred.append(pred_status)

    classified = len(y_true)

    if classified == 0:
        notes = (
            f"No classified records found. {unclassified_count} sample(s) were "
            "either missing from the database or had processing errors. "
            "Run the pipeline on the evaluation domains first."
        )
        return EvaluationReport(
            total_samples=len(samples),
            classified=0,
            unclassified=unclassified_count,
            notes=notes,
        )

    precision, recall, f1, confusion = compute_metrics(y_true, y_pred)

    notes_parts = [
        f"Evaluated {classified}/{len(samples)} samples (positive class: '{POSITIVE_CLASS}').",
    ]
    if unclassified_count:
        notes_parts.append(
            f"{unclassified_count} sample(s) unclassified: "
            + "; ".join(unclassified_reasons[:5])
        )
    if precision is None:
        notes_parts.append("Precision is None (no positive predictions were made).")
    if recall is None:
        notes_parts.append("Recall is None (no positive reference labels in evaluated set).")

    return EvaluationReport(
        total_samples=len(samples),
        classified=classified,
        unclassified=unclassified_count,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=confusion,
        estimated_api_cost_usd=None,      # Not tracked in MVP — no fabrication
        cost_per_accepted_prospect_usd=None,
        notes=" ".join(notes_parts),
    )
