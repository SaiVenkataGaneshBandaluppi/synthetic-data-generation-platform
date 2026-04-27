"""
Generate five domain-specific schema definitions and sample datasets.
Outputs data/sample_schemas.json with schemas and sample rows for each domain.
"""

import json
import os
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_schemas.json")
SAMPLE_ROWS = 50


def _random_date(start_year: int = 2020, end_year: int = 2024) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return (start + timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def _random_float(low: float, high: float, decimals: int = 2) -> float:
    return round(random.uniform(low, high), decimals)


def _random_normal(mean: float, std: float, low: float, high: float) -> float:
    value = mean + random.gauss(0, std)
    return round(max(low, min(high, value)), 2)


def build_healthcare_schema() -> dict:
    schema = {
        "domain": "healthcare",
        "columns": [
            {"name": "patient_id", "type": "text"},
            {"name": "age", "type": "integer", "min": 0, "max": 110, "mean": 45, "std": 20},
            {"name": "gender", "type": "categorical", "categories": ["M", "F", "Other"]},
            {"name": "blood_pressure_systolic", "type": "float", "min": 80, "max": 200, "mean": 120, "std": 15},
            {"name": "blood_pressure_diastolic", "type": "float", "min": 50, "max": 130, "mean": 80, "std": 10},
            {"name": "cholesterol_mgdl", "type": "float", "min": 100, "max": 400, "mean": 200, "std": 40},
            {"name": "bmi", "type": "float", "min": 10, "max": 60, "mean": 27, "std": 5},
            {"name": "has_diabetes", "type": "boolean", "p_true": 0.12},
            {"name": "has_hypertension", "type": "boolean", "p_true": 0.30},
            {"name": "admission_date", "type": "datetime", "min": "2020-01-01", "max": "2024-12-31"},
            {"name": "ward", "type": "categorical", "categories": ["cardiology", "neurology", "oncology", "general", "icu"]},
            {"name": "insurance_type", "type": "categorical", "categories": ["private", "public", "self_pay", "none"]},
        ],
    }
    rows = []
    genders = ["M", "F", "Other"]
    wards = ["cardiology", "neurology", "oncology", "general", "icu"]
    insurance = ["private", "public", "self_pay", "none"]
    for _ in range(SAMPLE_ROWS):
        row = {
            "patient_id": str(uuid.uuid4())[:8].upper(),
            "age": random.randint(0, 110),
            "gender": random.choices(genders, weights=[0.49, 0.49, 0.02])[0],
            "blood_pressure_systolic": _random_normal(120, 15, 80, 200),
            "blood_pressure_diastolic": _random_normal(80, 10, 50, 130),
            "cholesterol_mgdl": _random_normal(200, 40, 100, 400),
            "bmi": _random_normal(27, 5, 10, 60),
            "has_diabetes": random.random() < 0.12,
            "has_hypertension": random.random() < 0.30,
            "admission_date": _random_date(2020, 2024),
            "ward": random.choice(wards),
            "insurance_type": random.choices(insurance, weights=[0.4, 0.35, 0.15, 0.1])[0],
        }
        rows.append(row)
    return {"schema": schema, "sample_rows": rows}


def build_finance_schema() -> dict:
    schema = {
        "domain": "finance",
        "columns": [
            {"name": "account_id", "type": "text"},
            {"name": "customer_age", "type": "integer", "min": 18, "max": 85, "mean": 42, "std": 15},
            {"name": "annual_income_usd", "type": "float", "min": 15000, "max": 500000, "mean": 72000, "std": 45000},
            {"name": "credit_score", "type": "integer", "min": 300, "max": 850, "mean": 680, "std": 80},
            {"name": "account_type", "type": "categorical", "categories": ["checking", "savings", "investment", "credit"]},
            {"name": "balance_usd", "type": "float", "min": -5000, "max": 250000, "mean": 18000, "std": 30000},
            {"name": "loan_amount_usd", "type": "float", "min": 0, "max": 500000, "mean": 45000, "std": 60000},
            {"name": "loan_default", "type": "boolean", "p_true": 0.05},
            {"name": "transaction_count_monthly", "type": "integer", "min": 0, "max": 500, "mean": 35, "std": 25},
            {"name": "account_opened_date", "type": "datetime", "min": "2010-01-01", "max": "2024-12-31"},
        ],
    }
    rows = []
    account_types = ["checking", "savings", "investment", "credit"]
    for _ in range(SAMPLE_ROWS):
        row = {
            "account_id": f"ACC{random.randint(100000, 999999)}",
            "customer_age": random.randint(18, 85),
            "annual_income_usd": _random_normal(72000, 45000, 15000, 500000),
            "credit_score": int(_random_normal(680, 80, 300, 850)),
            "account_type": random.choice(account_types),
            "balance_usd": _random_normal(18000, 30000, -5000, 250000),
            "loan_amount_usd": max(0, _random_normal(45000, 60000, 0, 500000)),
            "loan_default": random.random() < 0.05,
            "transaction_count_monthly": int(_random_normal(35, 25, 0, 500)),
            "account_opened_date": _random_date(2010, 2024),
        }
        rows.append(row)
    return {"schema": schema, "sample_rows": rows}


def build_retail_schema() -> dict:
    schema = {
        "domain": "retail",
        "columns": [
            {"name": "order_id", "type": "text"},
            {"name": "product_category", "type": "categorical", "categories": ["electronics", "clothing", "food", "books", "home", "sports"]},
            {"name": "product_price_usd", "type": "float", "min": 0.99, "max": 2000, "mean": 85, "std": 150},
            {"name": "quantity", "type": "integer", "min": 1, "max": 50, "mean": 3, "std": 5},
            {"name": "discount_pct", "type": "float", "min": 0, "max": 70, "mean": 10, "std": 15},
            {"name": "customer_region", "type": "categorical", "categories": ["north", "south", "east", "west", "central"]},
            {"name": "payment_method", "type": "categorical", "categories": ["credit_card", "debit_card", "paypal", "cash", "crypto"]},
            {"name": "order_date", "type": "datetime", "min": "2022-01-01", "max": "2024-12-31"},
            {"name": "returned", "type": "boolean", "p_true": 0.08},
            {"name": "customer_satisfaction", "type": "integer", "min": 1, "max": 5, "mean": 4, "std": 1},
            {"name": "delivery_days", "type": "integer", "min": 1, "max": 30, "mean": 5, "std": 4},
        ],
    }
    rows = []
    categories = ["electronics", "clothing", "food", "books", "home", "sports"]
    regions = ["north", "south", "east", "west", "central"]
    payments = ["credit_card", "debit_card", "paypal", "cash", "crypto"]
    for _ in range(SAMPLE_ROWS):
        row = {
            "order_id": f"ORD{random.randint(10000000, 99999999)}",
            "product_category": random.choice(categories),
            "product_price_usd": _random_normal(85, 150, 0.99, 2000),
            "quantity": max(1, int(_random_normal(3, 5, 1, 50))),
            "discount_pct": max(0, min(70, _random_normal(10, 15, 0, 70))),
            "customer_region": random.choice(regions),
            "payment_method": random.choices(payments, weights=[0.4, 0.3, 0.2, 0.07, 0.03])[0],
            "order_date": _random_date(2022, 2024),
            "returned": random.random() < 0.08,
            "customer_satisfaction": max(1, min(5, int(_random_normal(4, 1, 1, 5)))),
            "delivery_days": max(1, int(_random_normal(5, 4, 1, 30))),
        }
        rows.append(row)
    return {"schema": schema, "sample_rows": rows}


def build_hr_schema() -> dict:
    schema = {
        "domain": "hr",
        "columns": [
            {"name": "employee_id", "type": "text"},
            {"name": "age", "type": "integer", "min": 18, "max": 70, "mean": 38, "std": 10},
            {"name": "gender", "type": "categorical", "categories": ["M", "F", "Non-binary"]},
            {"name": "department", "type": "categorical", "categories": ["engineering", "sales", "hr", "finance", "marketing", "operations"]},
            {"name": "job_level", "type": "categorical", "categories": ["junior", "mid", "senior", "lead", "manager", "director"]},
            {"name": "years_at_company", "type": "float", "min": 0, "max": 40, "mean": 6, "std": 5},
            {"name": "salary_usd", "type": "float", "min": 30000, "max": 300000, "mean": 90000, "std": 45000},
            {"name": "performance_score", "type": "float", "min": 1, "max": 5, "mean": 3.5, "std": 0.8},
            {"name": "remote_work", "type": "boolean", "p_true": 0.45},
            {"name": "attrition", "type": "boolean", "p_true": 0.15},
            {"name": "hire_date", "type": "datetime", "min": "2000-01-01", "max": "2024-12-31"},
            {"name": "training_hours_annual", "type": "integer", "min": 0, "max": 200, "mean": 40, "std": 25},
        ],
    }
    rows = []
    genders = ["M", "F", "Non-binary"]
    departments = ["engineering", "sales", "hr", "finance", "marketing", "operations"]
    levels = ["junior", "mid", "senior", "lead", "manager", "director"]
    for _ in range(SAMPLE_ROWS):
        row = {
            "employee_id": f"EMP{random.randint(1000, 9999)}",
            "age": int(_random_normal(38, 10, 18, 70)),
            "gender": random.choices(genders, weights=[0.48, 0.48, 0.04])[0],
            "department": random.choice(departments),
            "job_level": random.choices(levels, weights=[0.2, 0.3, 0.25, 0.1, 0.1, 0.05])[0],
            "years_at_company": max(0, round(_random_normal(6, 5, 0, 40), 1)),
            "salary_usd": _random_normal(90000, 45000, 30000, 300000),
            "performance_score": round(_random_normal(3.5, 0.8, 1, 5), 1),
            "remote_work": random.random() < 0.45,
            "attrition": random.random() < 0.15,
            "hire_date": _random_date(2000, 2024),
            "training_hours_annual": max(0, int(_random_normal(40, 25, 0, 200))),
        }
        rows.append(row)
    return {"schema": schema, "sample_rows": rows}


def build_iot_schema() -> dict:
    schema = {
        "domain": "iot",
        "columns": [
            {"name": "device_id", "type": "text"},
            {"name": "sensor_type", "type": "categorical", "categories": ["temperature", "humidity", "pressure", "vibration", "light", "motion"]},
            {"name": "temperature_celsius", "type": "float", "min": -40, "max": 120, "mean": 22, "std": 15},
            {"name": "humidity_pct", "type": "float", "min": 0, "max": 100, "mean": 55, "std": 20},
            {"name": "pressure_hpa", "type": "float", "min": 900, "max": 1100, "mean": 1013, "std": 10},
            {"name": "battery_level_pct", "type": "float", "min": 0, "max": 100, "mean": 72, "std": 25},
            {"name": "signal_strength_dbm", "type": "float", "min": -120, "max": 0, "mean": -65, "std": 20},
            {"name": "is_active", "type": "boolean", "p_true": 0.92},
            {"name": "alert_triggered", "type": "boolean", "p_true": 0.03},
            {"name": "timestamp", "type": "datetime", "min": "2024-01-01", "max": "2024-12-31"},
            {"name": "location_zone", "type": "categorical", "categories": ["zone_a", "zone_b", "zone_c", "zone_d", "outdoor"]},
        ],
    }
    rows = []
    sensor_types = ["temperature", "humidity", "pressure", "vibration", "light", "motion"]
    zones = ["zone_a", "zone_b", "zone_c", "zone_d", "outdoor"]
    for _ in range(SAMPLE_ROWS):
        row = {
            "device_id": f"DEV{random.randint(100000, 999999)}",
            "sensor_type": random.choice(sensor_types),
            "temperature_celsius": _random_normal(22, 15, -40, 120),
            "humidity_pct": _random_normal(55, 20, 0, 100),
            "pressure_hpa": _random_normal(1013, 10, 900, 1100),
            "battery_level_pct": _random_normal(72, 25, 0, 100),
            "signal_strength_dbm": _random_normal(-65, 20, -120, 0),
            "is_active": random.random() < 0.92,
            "alert_triggered": random.random() < 0.03,
            "timestamp": _random_date(2024, 2024),
            "location_zone": random.choice(zones),
        }
        rows.append(row)
    return {"schema": schema, "sample_rows": rows}


def main():
    domains = {
        "healthcare": build_healthcare_schema(),
        "finance": build_finance_schema(),
        "retail": build_retail_schema(),
        "hr": build_hr_schema(),
        "iot": build_iot_schema(),
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(domains, f, indent=2, default=str)
    print(f"Sample schemas written to {OUTPUT_PATH}")
    for domain, data in domains.items():
        print(f"  {domain}: {len(data['schema']['columns'])} columns, {len(data['sample_rows'])} rows")


if __name__ == "__main__":
    main()
