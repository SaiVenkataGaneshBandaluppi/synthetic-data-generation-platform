import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SENSITIVE_KEYWORDS = {
    "ssn", "social_security", "passport", "credit_card", "card_number",
    "bank_account", "routing_number", "tax_id", "driver_license",
}

_QUASI_IDENTIFIER_KEYWORDS = {
    "age", "zip", "zipcode", "postal", "gender", "sex", "race", "ethnicity",
    "dob", "birth", "nationality", "religion", "marital", "occupation",
}

_DIRECT_IDENTIFIER_KEYWORDS = {
    "name", "email", "phone", "address", "street", "city", "ssn",
    "passport", "license", "ip_address",
}


def _classify_columns(columns: list[str]) -> dict:
    sensitive = []
    quasi = []
    direct = []
    for col in columns:
        col_lower = col.lower()
        if any(k in col_lower for k in _SENSITIVE_KEYWORDS):
            sensitive.append(col)
        elif any(k in col_lower for k in _DIRECT_IDENTIFIER_KEYWORDS):
            direct.append(col)
        elif any(k in col_lower for k in _QUASI_IDENTIFIER_KEYWORDS):
            quasi.append(col)
    return {"sensitive": sensitive, "quasi": quasi, "direct": direct}


def _check_exact_matches(orig_df: pd.DataFrame, gen_df: pd.DataFrame) -> dict | None:
    shared_cols = [c for c in orig_df.columns if c in gen_df.columns]
    if not shared_cols:
        return None
    try:
        orig_tuples = set(
            map(tuple, orig_df[shared_cols].fillna("__NULL__").astype(str).values.tolist())
        )
        gen_tuples = gen_df[shared_cols].fillna("__NULL__").astype(str).values.tolist()
        match_count = sum(1 for row in gen_tuples if tuple(row) in orig_tuples)
        if match_count == 0:
            return None
        match_rate = match_count / max(len(gen_tuples), 1)
        return {
            "type": "exact_match",
            "severity": "high" if match_rate > 0.01 else "low",
            "count": match_count,
            "rate": round(match_rate, 4),
            "recommendation": (
                "Add more noise or increase row count to reduce exact matches with original data"
            ),
        }
    except Exception:
        return None


def _check_quasi_identifiers(gen_df: pd.DataFrame, col_classes: dict) -> dict | None:
    quasi_cols = [c for c in col_classes.get("quasi", []) if c in gen_df.columns]
    if len(quasi_cols) < 2:
        return None
    try:
        subset = gen_df[quasi_cols].astype(str)
        total = len(subset)
        if total == 0:
            return None
        unique_combos = subset.drop_duplicates().shape[0]
        uniqueness = unique_combos / total
        if uniqueness > 0.9:
            return {
                "type": "quasi_identifier_linkage",
                "severity": "medium",
                "fields": quasi_cols,
                "uniqueness_rate": round(uniqueness, 4),
                "recommendation": (
                    "Generalise or suppress quasi-identifier fields to prevent re-identification"
                ),
            }
    except Exception:
        pass
    return None


def _check_sensitive_exposure(gen_df: pd.DataFrame, col_classes: dict) -> dict | None:
    sensitive_cols = [c for c in col_classes.get("sensitive", []) if c in gen_df.columns]
    if not sensitive_cols:
        return None
    return {
        "type": "sensitive_field_exposure",
        "severity": "high",
        "fields": sensitive_cols,
        "recommendation": (
            "Consider removing or tokenising sensitive fields before sharing synthetic data"
        ),
    }


def _check_direct_identifiers(gen_df: pd.DataFrame, col_classes: dict) -> dict | None:
    direct_cols = [c for c in col_classes.get("direct", []) if c in gen_df.columns]
    if not direct_cols:
        return None
    return {
        "type": "direct_identifier_present",
        "severity": "medium",
        "fields": direct_cols,
        "recommendation": (
            "Direct identifiers detected. Verify generated values do not resemble real individuals"
        ),
    }


class PrivacyAuditorAgent:
    def run(
        self,
        sample_records: list[dict],
        generated_records: list[dict],
    ) -> dict:
        risks: list[dict] = []

        if not generated_records:
            return {
                "risks": [],
                "risk_level": "safe",
                "recommendations": [],
                "summary": "No generated data to audit",
            }

        gen_df = pd.DataFrame(generated_records)
        orig_df = pd.DataFrame(sample_records) if sample_records else pd.DataFrame()
        col_classes = _classify_columns(list(gen_df.columns))

        if not orig_df.empty:
            exact = _check_exact_matches(orig_df, gen_df)
            if exact:
                risks.append(exact)

        quasi = _check_quasi_identifiers(gen_df, col_classes)
        if quasi:
            risks.append(quasi)

        sensitive = _check_sensitive_exposure(gen_df, col_classes)
        if sensitive:
            risks.append(sensitive)

        direct = _check_direct_identifiers(gen_df, col_classes)
        if direct:
            risks.append(direct)

        severity_rank = {"high": 3, "medium": 2, "low": 1}
        if risks:
            max_sev = max(severity_rank.get(r["severity"], 0) for r in risks)
            risk_level = {3: "high", 2: "medium", 1: "low"}.get(max_sev, "low")
        else:
            risk_level = "safe"

        return {
            "risks": risks,
            "risk_level": risk_level,
            "recommendations": [r["recommendation"] for r in risks],
            "column_classification": col_classes,
            "summary": (
                f"Audited {len(gen_df.columns)} columns. "
                f"Found {len(risks)} risk(s). "
                f"Overall risk level: {risk_level}."
            ),
        }
