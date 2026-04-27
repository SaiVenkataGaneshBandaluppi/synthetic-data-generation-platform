import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

FIDELITY_THRESHOLD = 70.0


def _score_numeric_column(orig: pd.Series, gen: pd.Series) -> float:
    orig_clean = orig.dropna().astype(float)
    gen_clean = gen.dropna().astype(float)
    if len(orig_clean) < 2 or len(gen_clean) < 2:
        return 80.0
    try:
        stat, _ = sp_stats.ks_2samp(orig_clean.values, gen_clean.values)
        return float(max(0.0, min(100.0, (1.0 - stat) * 100.0)))
    except Exception:
        return 80.0


def _score_categorical_column(orig: pd.Series, gen: pd.Series) -> float:
    orig_str = orig.dropna().astype(str)
    gen_str = gen.dropna().astype(str)
    if len(orig_str) == 0 or len(gen_str) == 0:
        return 80.0
    all_cats = sorted(set(orig_str.unique()) | set(gen_str.unique()))
    if len(all_cats) < 2:
        return 90.0
    try:
        orig_counts = orig_str.value_counts().reindex(all_cats, fill_value=0).values
        gen_counts = gen_str.value_counts().reindex(all_cats, fill_value=0).values
        orig_counts = orig_counts + 1
        gen_counts = gen_counts + 1
        orig_exp = orig_counts / orig_counts.sum() * gen_counts.sum()
        stat, _ = sp_stats.chisquare(gen_counts, f_exp=orig_exp)
        degrees = max(len(all_cats) - 1, 1)
        normalized = min(stat / (degrees * 10.0), 1.0)
        return float(max(0.0, min(100.0, (1.0 - normalized) * 100.0)))
    except Exception:
        return 80.0


def _score_boolean_column(orig: pd.Series, gen: pd.Series) -> float:
    orig_clean = orig.dropna().astype(bool)
    gen_clean = gen.dropna().astype(bool)
    if len(orig_clean) == 0 or len(gen_clean) == 0:
        return 80.0
    orig_p = float(orig_clean.mean())
    gen_p = float(gen_clean.mean())
    diff = abs(orig_p - gen_p)
    return float(max(0.0, (1.0 - diff * 2.0) * 100.0))


class QualityValidatorAgent:
    def run(
        self,
        sample_records: list[dict],
        generated_records: list[dict],
        schema_analysis: dict,
        threshold: float = FIDELITY_THRESHOLD,
    ) -> dict:
        if not sample_records or not generated_records:
            return {
                "column_scores": [],
                "overall_fidelity_score": 0.0,
                "threshold": threshold,
                "flagged_columns": [],
                "summary": "Insufficient data for validation",
            }

        orig_df = pd.DataFrame(sample_records)
        gen_df = pd.DataFrame(generated_records)
        columns_meta = {c["name"]: c for c in schema_analysis.get("columns", [])}
        column_scores = []

        for col in orig_df.columns:
            if col not in gen_df.columns:
                continue
            col_meta = columns_meta.get(col, {"type": "text"})
            col_type = col_meta.get("type", "text")
            orig_series = orig_df[col]
            gen_series = gen_df[col]

            if col_type == "numeric":
                score = _score_numeric_column(orig_series, gen_series)
            elif col_type == "categorical":
                score = _score_categorical_column(orig_series, gen_series)
            elif col_type == "boolean":
                score = _score_boolean_column(orig_series, gen_series)
            else:
                score = 80.0

            column_scores.append(
                {
                    "column": col,
                    "type": col_type,
                    "score": round(score, 2),
                    "below_threshold": score < threshold,
                }
            )

        if column_scores:
            overall = float(np.mean([c["score"] for c in column_scores]))
        else:
            overall = 0.0

        flagged = [c["column"] for c in column_scores if c["below_threshold"]]

        return {
            "column_scores": column_scores,
            "overall_fidelity_score": round(overall, 2),
            "threshold": threshold,
            "flagged_columns": flagged,
            "summary": (
                f"Validated {len(column_scores)} columns. "
                f"Overall fidelity: {overall:.1f}. "
                f"Flagged: {len(flagged)} column(s) below threshold."
            ),
        }
