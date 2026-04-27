import logging
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RNG = np.random.default_rng()


def _generate_column(model: dict, n: int) -> list:
    dist = model.get("distribution", "")
    params = model.get("params", {})
    nullable = model.get("nullable", False)
    null_rate = float(model.get("null_rate", 0.0))

    if dist == "normal":
        mean = float(params.get("mean", 0))
        std = max(float(params.get("std", 1)), 1e-9)
        values = _RNG.normal(mean, std, n).tolist()
        col_min = model.get("min")
        col_max = model.get("max")
        if col_min is not None and col_max is not None:
            values = [max(col_min, min(col_max, v)) for v in values]
    elif dist == "exponential":
        rate = float(params.get("rate", 1.0))
        scale = 1.0 / max(rate, 1e-9)
        values = _RNG.exponential(scale, n).tolist()
        col_min = model.get("min")
        col_max = model.get("max")
        if col_min is not None and col_max is not None:
            values = [max(col_min, min(col_max, v)) for v in values]
    elif dist == "categorical":
        categories = params.get("categories", ["A"])
        probs = params.get("probabilities", [1.0])
        total = sum(probs)
        probs = [p / total for p in probs]
        if len(probs) != len(categories):
            probs = [1.0 / len(categories)] * len(categories)
        indices = _RNG.choice(len(categories), size=n, p=probs)
        values = [str(categories[i]) for i in indices]
    elif dist == "bernoulli":
        p_true = float(params.get("p_true", 0.5))
        p_true = max(0.0, min(1.0, p_true))
        values = [bool(v) for v in _RNG.choice([True, False], size=n, p=[p_true, 1 - p_true])]
    elif dist == "datetime_uniform":
        try:
            dt_min = datetime.fromisoformat(str(params.get("min", "2020-01-01T00:00:00")))
            dt_max = datetime.fromisoformat(str(params.get("max", "2024-12-31T23:59:59")))
        except Exception:
            dt_min = datetime(2020, 1, 1)
            dt_max = datetime(2024, 12, 31)
        delta_seconds = max(int((dt_max - dt_min).total_seconds()), 1)
        offsets = _RNG.integers(0, delta_seconds, size=n)
        values = [(dt_min + timedelta(seconds=int(o))).isoformat() for o in offsets]
    elif dist == "text_template":
        template = params.get("template", "value_{i}")
        values = [template.format(i=i) for i in range(n)]
    else:
        values = [f"value_{i}" for i in range(n)]

    if nullable and null_rate > 0:
        null_mask = _RNG.random(n) < null_rate
        values = [None if m else v for v, m in zip(values, null_mask)]

    return values


def _ensure_id_uniqueness(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        c for c in df.columns if c.lower() == "id" or c.lower().endswith("_id")
    ]
    for col in id_cols:
        if col.lower() == "id":
            df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            existing = df[col].dropna()
            if existing.nunique() < len(existing) * 0.5:
                pass
    return df


class DataGeneratorAgent:
    def run(self, distribution_model: dict, row_count: int) -> list[dict]:
        column_models = distribution_model.get("column_models", [])
        if not column_models:
            return []

        row_count = max(1, min(row_count, 10_000))
        data: dict = {}
        for model in column_models:
            name = model["name"]
            try:
                data[name] = _generate_column(model, row_count)
            except Exception as err:
                logger.warning("Failed to generate column %s: %s", name, err)
                data[name] = [None] * row_count

        df = pd.DataFrame(data)
        df = _ensure_id_uniqueness(df)
        raw_records = df.to_dict(orient="records")
        for record in raw_records:
            for key, val in record.items():
                if val is not None and isinstance(val, float) and pd.isna(val):
                    record[key] = None
        return raw_records
