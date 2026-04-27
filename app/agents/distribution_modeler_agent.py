import json
import logging

from app.services.groq_client import groq_complete

logger = logging.getLogger(__name__)


def _fit_numeric(col_stats: dict) -> dict:
    mean = col_stats.get("mean", 0.0)
    std = max(col_stats.get("std", 1.0), 1e-9)
    col_min = col_stats.get("min")
    col_max = col_stats.get("max")
    dist_type = "normal"
    if col_min is not None and col_max is not None and col_min >= 0:
        range_val = col_max - col_min
        if std > 0 and (range_val / std) > 4:
            dist_type = "normal"
        if mean > 0 and std > mean:
            dist_type = "exponential"
    params: dict = {"mean": float(mean), "std": float(std)}
    if dist_type == "exponential" and mean > 0:
        params["rate"] = 1.0 / float(mean)
    return {
        "distribution": dist_type,
        "params": params,
        "min": col_min,
        "max": col_max,
    }


def _fit_categorical(col_stats: dict) -> dict:
    freqs = col_stats.get("frequencies", {})
    if not freqs:
        freqs = {str(k): float(v) for k, v in col_stats.get("value_counts", {}).items()}
        total = max(sum(freqs.values()), 1)
        freqs = {k: v / total for k, v in freqs.items()}
    categories = list(freqs.keys())
    probabilities = [float(freqs[c]) for c in categories]
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    else:
        probabilities = [1.0 / len(categories)] * len(categories)
    return {
        "distribution": "categorical",
        "params": {"categories": categories, "probabilities": probabilities},
    }


def _fit_datetime(col_stats: dict) -> dict:
    return {
        "distribution": "datetime_uniform",
        "params": {
            "min": col_stats.get("min", "2020-01-01T00:00:00"),
            "max": col_stats.get("max", "2024-12-31T23:59:59"),
        },
    }


def _fit_boolean(col_stats: dict) -> dict:
    return {
        "distribution": "bernoulli",
        "params": {"p_true": float(col_stats.get("p_true", 0.5))},
    }


def _fit_text(col_stats: dict, col_name: str) -> dict:
    avg_len = int(col_stats.get("avg_length", 20))
    return {
        "distribution": "text_template",
        "params": {"template": f"{col_name}_{{i}}", "avg_length": avg_len},
    }


class DistributionModelerAgent:
    def run(self, schema_analysis: dict) -> dict:
        columns = schema_analysis.get("columns", [])
        column_models = []
        for col in columns:
            name = col["name"]
            col_type = col["type"]
            stats = col.get("statistics", {})
            nullable = col.get("nullable", False)
            null_rate = 0.0
            if nullable and stats.get("row_count", 0) > 0:
                null_rate = float(stats.get("null_count", 0)) / float(stats["row_count"])

            if col_type == "numeric":
                dist = _fit_numeric(stats)
            elif col_type == "categorical":
                dist = _fit_categorical(stats)
            elif col_type == "datetime":
                dist = _fit_datetime(stats)
            elif col_type == "boolean":
                dist = _fit_boolean(stats)
            else:
                dist = _fit_text(stats, name)

            column_models.append(
                {
                    "name": name,
                    "type": col_type,
                    "nullable": nullable,
                    "null_rate": null_rate,
                    **dist,
                }
            )

        groq_result = self._try_groq_enhancement(column_models)
        if groq_result and isinstance(groq_result.get("column_models"), list):
            enhanced = groq_result["column_models"]
            if len(enhanced) == len(column_models):
                column_models = enhanced

        return {
            "column_models": column_models,
            "relationships": schema_analysis.get("relationships", []),
        }

    def _try_groq_enhancement(self, column_models: list[dict]) -> dict | None:
        prompt = json.dumps(
            {"task": "validate_distributions", "column_models": column_models},
            default=str,
        )
        system = (
            "You are a statistical modelling expert. Review the fitted distributions for "
            "each column. Return JSON with key 'column_models' containing the same list "
            "with any corrections. Keep all original keys. Only modify distribution or "
            "params if clearly wrong."
        )
        return groq_complete(prompt, system, max_tokens=1024)
