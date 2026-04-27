import json
import logging

import numpy as np
import pandas as pd

from app.services.groq_client import groq_complete

logger = logging.getLogger(__name__)

_NUMERIC_DTYPES = (np.integer, np.floating)


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= 10 and series.nunique(dropna=True) > 0:
            if all(float(v).is_integer() for v in series.dropna().unique()):
                unique_ratio = series.nunique(dropna=True) / max(len(series.dropna()), 1)
                if unique_ratio < 0.05:
                    return "categorical"
        return "numeric"
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"
    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / max(len(non_null), 1)
    if unique_count <= 20 or (unique_count <= 50 and unique_ratio < 0.3):
        return "categorical"
    try:
        pd.to_datetime(non_null.iloc[:20])
        return "datetime"
    except Exception:
        pass
    return "text"


def _compute_statistics(series: pd.Series, col_type: str) -> dict:
    null_count = int(series.isnull().sum())
    base: dict = {"null_count": null_count, "row_count": len(series)}
    non_null = series.dropna()
    if col_type == "numeric":
        base.update(
            {
                "mean": float(non_null.mean()) if len(non_null) else 0.0,
                "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                "min": float(non_null.min()) if len(non_null) else 0.0,
                "max": float(non_null.max()) if len(non_null) else 0.0,
                "median": float(non_null.median()) if len(non_null) else 0.0,
                "q25": float(non_null.quantile(0.25)) if len(non_null) else 0.0,
                "q75": float(non_null.quantile(0.75)) if len(non_null) else 0.0,
            }
        )
    elif col_type == "categorical":
        counts = non_null.astype(str).value_counts()
        total = max(len(non_null), 1)
        base.update(
            {
                "unique_count": int(series.nunique(dropna=True)),
                "value_counts": {str(k): int(v) for k, v in counts.items()},
                "frequencies": {str(k): float(v) / total for k, v in counts.items()},
            }
        )
    elif col_type == "datetime":
        try:
            dt = pd.to_datetime(non_null)
            base.update(
                {
                    "min": str(dt.min().isoformat()) if len(dt) else "",
                    "max": str(dt.max().isoformat()) if len(dt) else "",
                }
            )
        except Exception:
            base.update({"min": "", "max": ""})
    elif col_type == "boolean":
        true_count = int(non_null.astype(bool).sum())
        total = max(len(non_null), 1)
        base.update({"p_true": float(true_count) / total})
    else:
        lengths = non_null.astype(str).str.len()
        base.update(
            {
                "avg_length": float(lengths.mean()) if len(lengths) else 0.0,
                "max_length": int(lengths.max()) if len(lengths) else 0,
            }
        )
    return base


def _detect_relationships(df: pd.DataFrame, columns: list[dict]) -> list[dict]:
    relationships = []
    col_names = [c["name"] for c in columns]
    id_cols = [n for n in col_names if n.lower().endswith("_id") or n.lower() == "id"]
    ref_cols = [
        n for n in col_names if not n.lower().endswith("_id") and n.lower() != "id"
    ]
    for id_col in id_cols:
        for ref_col in ref_cols:
            if id_col.lower().startswith(ref_col.lower()):
                relationships.append(
                    {"type": "foreign_key_hint", "from": id_col, "to": ref_col}
                )
    numeric_cols = [
        c["name"] for c in columns if c["type"] == "numeric"
    ]
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            if col_a in df.columns and col_b in df.columns:
                try:
                    corr = df[col_a].corr(df[col_b])
                    if not np.isnan(corr) and abs(corr) > 0.7:
                        relationships.append(
                            {
                                "type": "correlation",
                                "columns": [col_a, col_b],
                                "value": round(float(corr), 4),
                            }
                        )
                except Exception:
                    pass
    return relationships


class SchemaAnalyzerAgent:
    def run(self, raw_input: dict, sample_records: list[dict]) -> dict:
        if sample_records:
            df = pd.DataFrame(sample_records)
            return self._analyze_dataframe(df)
        if raw_input.get("columns"):
            return self._analyze_json_schema(raw_input)
        return {"columns": [], "row_count": 0, "relationships": []}

    def _analyze_dataframe(self, df: pd.DataFrame) -> dict:
        columns = []
        for col in df.columns:
            col_type = _infer_column_type(df[col])
            nullable = bool(df[col].isnull().any())
            stats = _compute_statistics(df[col], col_type)
            columns.append(
                {
                    "name": col,
                    "type": col_type,
                    "nullable": nullable,
                    "statistics": stats,
                }
            )
        relationships = _detect_relationships(df, columns)
        groq_hints = self._try_groq_enhancement(columns)
        if groq_hints:
            columns = groq_hints.get("columns", columns)
        return {
            "columns": columns,
            "row_count": len(df),
            "relationships": relationships,
        }

    def _analyze_json_schema(self, schema: dict) -> dict:
        raw_cols = schema.get("columns", [])
        columns = []
        for col in raw_cols:
            name = col.get("name", "unknown")
            raw_type = col.get("type", "text").lower()
            if raw_type in ("integer", "float", "number", "double"):
                col_type = "numeric"
            elif raw_type in ("bool", "boolean"):
                col_type = "boolean"
            elif raw_type in ("date", "datetime", "timestamp"):
                col_type = "datetime"
            elif raw_type in ("category", "categorical", "enum"):
                col_type = "categorical"
            else:
                col_type = "text"
            statistics: dict = {"null_count": 0, "row_count": 0}
            if col_type == "numeric":
                statistics.update(
                    {
                        "mean": float(col.get("mean", 0)),
                        "std": float(col.get("std", 1)),
                        "min": float(col.get("min", 0)),
                        "max": float(col.get("max", 100)),
                        "median": float(col.get("median", 50)),
                        "q25": float(col.get("q25", 25)),
                        "q75": float(col.get("q75", 75)),
                    }
                )
            elif col_type == "categorical":
                cats = col.get("categories", ["A", "B", "C"])
                n = len(cats)
                statistics.update(
                    {
                        "unique_count": n,
                        "value_counts": {c: 1 for c in cats},
                        "frequencies": {c: 1.0 / n for c in cats},
                    }
                )
            elif col_type == "datetime":
                statistics.update(
                    {
                        "min": col.get("min", "2020-01-01T00:00:00"),
                        "max": col.get("max", "2024-12-31T23:59:59"),
                    }
                )
            elif col_type == "boolean":
                statistics.update({"p_true": float(col.get("p_true", 0.5))})
            else:
                statistics.update({"avg_length": 20.0, "max_length": 100})
            columns.append(
                {
                    "name": name,
                    "type": col_type,
                    "nullable": bool(col.get("nullable", False)),
                    "statistics": statistics,
                }
            )
        return {"columns": columns, "row_count": 0, "relationships": []}

    def _try_groq_enhancement(self, columns: list[dict]) -> dict | None:
        prompt = json.dumps(
            {"task": "validate_column_types", "columns": columns}, default=str
        )
        system = (
            "You are a data schema expert. Given a list of columns with inferred types, "
            "validate or correct the types. Return JSON with key 'columns' containing the "
            "same list with corrected types if needed. Valid types: numeric, categorical, "
            "datetime, boolean, text. Keep all original fields."
        )
        return groq_complete(prompt, system, max_tokens=1024)
