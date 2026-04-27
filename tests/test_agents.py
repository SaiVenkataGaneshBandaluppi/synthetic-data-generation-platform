import pytest

from app.agents.data_generator_agent import DataGeneratorAgent
from app.agents.distribution_modeler_agent import DistributionModelerAgent
from app.agents.privacy_auditor_agent import PrivacyAuditorAgent
from app.agents.quality_validator_agent import QualityValidatorAgent
from app.agents.schema_analyzer_agent import SchemaAnalyzerAgent

_NUMERIC_RECORDS = [
    {"age": 25, "salary": 55000.0, "score": 88.5},
    {"age": 30, "salary": 62000.0, "score": 91.0},
    {"age": 45, "salary": 95000.0, "score": 78.3},
    {"age": 22, "salary": 48000.0, "score": 95.1},
    {"age": 38, "salary": 71000.0, "score": 82.7},
]

_CATEGORICAL_RECORDS = [
    {"gender": "M", "status": "active", "region": "north"},
    {"gender": "F", "status": "inactive", "region": "south"},
    {"gender": "M", "status": "active", "region": "east"},
    {"gender": "F", "status": "active", "region": "west"},
    {"gender": "M", "status": "inactive", "region": "north"},
]

_MIXED_RECORDS = [
    {"id": "1", "age": 28, "gender": "M", "active": True, "created_at": "2023-01-15"},
    {"id": "2", "age": 35, "gender": "F", "active": False, "created_at": "2023-03-22"},
    {"id": "3", "age": 42, "gender": "M", "active": True, "created_at": "2023-06-10"},
    {"id": "4", "age": 29, "gender": "F", "active": True, "created_at": "2023-09-05"},
    {"id": "5", "age": 55, "gender": "M", "active": False, "created_at": "2023-11-30"},
]

_NULLABLE_RECORDS = [
    {"name": "Alice", "email": None, "score": 80.0},
    {"name": "Bob", "email": "bob@test.com", "score": None},
    {"name": "Carol", "email": None, "score": 92.0},
    {"name": "Dave", "email": "dave@test.com", "score": 75.0},
    {"name": None, "email": "eve@test.com", "score": 88.0},
]


class TestSchemaAnalyzerAgent:
    @pytest.mark.asyncio
    async def test_identifies_numeric_columns(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _NUMERIC_RECORDS)
        col_types = {c["name"]: c["type"] for c in result["columns"]}
        assert col_types["age"] == "numeric"
        assert col_types["salary"] == "numeric"

    @pytest.mark.asyncio
    async def test_identifies_categorical_columns(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _CATEGORICAL_RECORDS)
        col_types = {c["name"]: c["type"] for c in result["columns"]}
        assert col_types["gender"] == "categorical"
        assert col_types["status"] == "categorical"

    @pytest.mark.asyncio
    async def test_identifies_datetime_columns(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _MIXED_RECORDS)
        col_types = {c["name"]: c["type"] for c in result["columns"]}
        assert col_types["created_at"] in ("datetime", "text", "categorical")

    @pytest.mark.asyncio
    async def test_identifies_boolean_columns(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _MIXED_RECORDS)
        col_types = {c["name"]: c["type"] for c in result["columns"]}
        assert col_types["active"] in ("boolean", "categorical")

    @pytest.mark.asyncio
    async def test_handles_nullable_fields(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _NULLABLE_RECORDS)
        cols = {c["name"]: c for c in result["columns"]}
        assert cols["email"]["nullable"] is True
        assert cols["score"]["nullable"] is True

    @pytest.mark.asyncio
    async def test_computes_numeric_statistics(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _NUMERIC_RECORDS)
        age_col = next(c for c in result["columns"] if c["name"] == "age")
        stats = age_col["statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] <= stats["mean"] <= stats["max"]

    @pytest.mark.asyncio
    async def test_computes_categorical_statistics(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _CATEGORICAL_RECORDS)
        gender_col = next(c for c in result["columns"] if c["name"] == "gender")
        stats = gender_col["statistics"]
        assert "value_counts" in stats
        assert "M" in stats["value_counts"] or "F" in stats["value_counts"]

    @pytest.mark.asyncio
    async def test_returns_row_count(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, _NUMERIC_RECORDS)
        assert result["row_count"] == len(_NUMERIC_RECORDS)

    @pytest.mark.asyncio
    async def test_analyzes_json_schema(self):
        agent = SchemaAnalyzerAgent()
        schema = {
            "columns": [
                {"name": "age", "type": "integer", "min": 18, "max": 90},
                {"name": "category", "type": "categorical", "categories": ["A", "B", "C"]},
            ]
        }
        result = agent.run(schema, [])
        col_types = {c["name"]: c["type"] for c in result["columns"]}
        assert col_types["age"] == "numeric"
        assert col_types["category"] == "categorical"

    @pytest.mark.asyncio
    async def test_handles_empty_input(self):
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, [])
        assert result["columns"] == []
        assert result["row_count"] == 0

    @pytest.mark.asyncio
    async def test_relationships_detected_for_correlated_numerics(self):
        records = [{"x": i, "y": i * 2 + 1} for i in range(20)]
        agent = SchemaAnalyzerAgent()
        result = agent.run({}, records)
        assert "relationships" in result

    @pytest.mark.asyncio
    async def test_json_schema_datetime_type(self):
        agent = SchemaAnalyzerAgent()
        schema = {
            "columns": [
                {"name": "created_at", "type": "datetime", "min": "2020-01-01", "max": "2024-12-31"}
            ]
        }
        result = agent.run(schema, [])
        col = result["columns"][0]
        assert col["type"] == "datetime"


class TestDistributionModelerAgent:
    def _get_numeric_schema(self) -> dict:
        return {
            "columns": [
                {
                    "name": "age",
                    "type": "numeric",
                    "nullable": False,
                    "statistics": {
                        "mean": 35.0,
                        "std": 10.0,
                        "min": 18.0,
                        "max": 70.0,
                        "null_count": 0,
                        "row_count": 100,
                    },
                }
            ],
            "relationships": [],
        }

    def _get_categorical_schema(self) -> dict:
        return {
            "columns": [
                {
                    "name": "status",
                    "type": "categorical",
                    "nullable": False,
                    "statistics": {
                        "unique_count": 3,
                        "value_counts": {"active": 60, "inactive": 30, "pending": 10},
                        "frequencies": {"active": 0.6, "inactive": 0.3, "pending": 0.1},
                        "null_count": 0,
                        "row_count": 100,
                    },
                }
            ],
            "relationships": [],
        }

    @pytest.mark.asyncio
    async def test_fits_normal_distribution_for_numeric(self):
        agent = DistributionModelerAgent()
        schema = self._get_numeric_schema()
        result = agent.run(schema)
        models = result["column_models"]
        age_model = next(m for m in models if m["name"] == "age")
        assert age_model["distribution"] in ("normal", "exponential")
        assert "params" in age_model
        assert "mean" in age_model["params"] or "rate" in age_model["params"]

    @pytest.mark.asyncio
    async def test_fits_categorical_frequency(self):
        agent = DistributionModelerAgent()
        schema = self._get_categorical_schema()
        result = agent.run(schema)
        status_model = next(m for m in result["column_models"] if m["name"] == "status")
        assert status_model["distribution"] == "categorical"
        assert "categories" in status_model["params"]
        assert "probabilities" in status_model["params"]
        assert len(status_model["params"]["categories"]) == 3

    @pytest.mark.asyncio
    async def test_probabilities_sum_to_one(self):
        agent = DistributionModelerAgent()
        schema = self._get_categorical_schema()
        result = agent.run(schema)
        status_model = next(m for m in result["column_models"] if m["name"] == "status")
        total = sum(status_model["params"]["probabilities"])
        assert abs(total - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_fits_datetime_distribution(self):
        schema = {
            "columns": [
                {
                    "name": "created_at",
                    "type": "datetime",
                    "nullable": False,
                    "statistics": {
                        "min": "2020-01-01T00:00:00",
                        "max": "2024-12-31T23:59:59",
                        "null_count": 0,
                        "row_count": 100,
                    },
                }
            ],
            "relationships": [],
        }
        agent = DistributionModelerAgent()
        result = agent.run(schema)
        dt_model = result["column_models"][0]
        assert dt_model["distribution"] == "datetime_uniform"
        assert "min" in dt_model["params"]
        assert "max" in dt_model["params"]

    @pytest.mark.asyncio
    async def test_fits_boolean_distribution(self):
        schema = {
            "columns": [
                {
                    "name": "active",
                    "type": "boolean",
                    "nullable": False,
                    "statistics": {"p_true": 0.7, "null_count": 0, "row_count": 100},
                }
            ],
            "relationships": [],
        }
        agent = DistributionModelerAgent()
        result = agent.run(schema)
        bool_model = result["column_models"][0]
        assert bool_model["distribution"] == "bernoulli"
        assert abs(bool_model["params"]["p_true"] - 0.7) < 1e-6

    @pytest.mark.asyncio
    async def test_handles_nullable_column(self):
        schema = {
            "columns": [
                {
                    "name": "score",
                    "type": "numeric",
                    "nullable": True,
                    "statistics": {
                        "mean": 75.0,
                        "std": 15.0,
                        "min": 0.0,
                        "max": 100.0,
                        "null_count": 20,
                        "row_count": 100,
                    },
                }
            ],
            "relationships": [],
        }
        agent = DistributionModelerAgent()
        result = agent.run(schema)
        model = result["column_models"][0]
        assert model["nullable"] is True
        assert model["null_rate"] > 0

    @pytest.mark.asyncio
    async def test_handles_empty_schema(self):
        agent = DistributionModelerAgent()
        result = agent.run({"columns": [], "relationships": []})
        assert result["column_models"] == []

    @pytest.mark.asyncio
    async def test_text_distribution_template(self):
        schema = {
            "columns": [
                {
                    "name": "notes",
                    "type": "text",
                    "nullable": False,
                    "statistics": {"avg_length": 50.0, "max_length": 200, "null_count": 0, "row_count": 100},
                }
            ],
            "relationships": [],
        }
        agent = DistributionModelerAgent()
        result = agent.run(schema)
        text_model = result["column_models"][0]
        assert text_model["distribution"] == "text_template"


class TestDataGeneratorAgent:
    def _build_simple_model(self) -> dict:
        return {
            "column_models": [
                {
                    "name": "age",
                    "type": "numeric",
                    "nullable": False,
                    "null_rate": 0.0,
                    "distribution": "normal",
                    "params": {"mean": 35.0, "std": 10.0},
                    "min": 18.0,
                    "max": 80.0,
                },
                {
                    "name": "status",
                    "type": "categorical",
                    "nullable": False,
                    "null_rate": 0.0,
                    "distribution": "categorical",
                    "params": {
                        "categories": ["active", "inactive"],
                        "probabilities": [0.7, 0.3],
                    },
                },
                {
                    "name": "active",
                    "type": "boolean",
                    "nullable": False,
                    "null_rate": 0.0,
                    "distribution": "bernoulli",
                    "params": {"p_true": 0.6},
                },
            ],
            "relationships": [],
        }

    @pytest.mark.asyncio
    async def test_produces_correct_row_count(self):
        agent = DataGeneratorAgent()
        model = self._build_simple_model()
        records = agent.run(model, 50)
        assert len(records) == 50

    @pytest.mark.asyncio
    async def test_output_matches_schema_columns(self):
        agent = DataGeneratorAgent()
        model = self._build_simple_model()
        records = agent.run(model, 10)
        assert len(records) > 0
        for record in records:
            assert "age" in record
            assert "status" in record
            assert "active" in record

    @pytest.mark.asyncio
    async def test_numeric_values_within_bounds(self):
        agent = DataGeneratorAgent()
        model = self._build_simple_model()
        records = agent.run(model, 100)
        ages = [r["age"] for r in records if r["age"] is not None]
        assert all(18.0 <= a <= 80.0 for a in ages)

    @pytest.mark.asyncio
    async def test_categorical_values_from_distribution(self):
        agent = DataGeneratorAgent()
        model = self._build_simple_model()
        records = agent.run(model, 100)
        statuses = {r["status"] for r in records if r["status"] is not None}
        assert statuses.issubset({"active", "inactive"})

    @pytest.mark.asyncio
    async def test_boolean_values_are_bool(self):
        agent = DataGeneratorAgent()
        model = self._build_simple_model()
        records = agent.run(model, 20)
        actives = [r["active"] for r in records if r["active"] is not None]
        assert all(isinstance(a, bool) for a in actives)

    @pytest.mark.asyncio
    async def test_datetime_generation(self):
        model = {
            "column_models": [
                {
                    "name": "created_at",
                    "type": "datetime",
                    "nullable": False,
                    "null_rate": 0.0,
                    "distribution": "datetime_uniform",
                    "params": {
                        "min": "2020-01-01T00:00:00",
                        "max": "2024-12-31T23:59:59",
                    },
                }
            ],
            "relationships": [],
        }
        agent = DataGeneratorAgent()
        records = agent.run(model, 20)
        assert len(records) == 20
        assert all("created_at" in r for r in records)

    @pytest.mark.asyncio
    async def test_nullable_column_produces_nulls(self):
        model = {
            "column_models": [
                {
                    "name": "optional_score",
                    "type": "numeric",
                    "nullable": True,
                    "null_rate": 0.5,
                    "distribution": "normal",
                    "params": {"mean": 50.0, "std": 10.0},
                    "min": 0.0,
                    "max": 100.0,
                }
            ],
            "relationships": [],
        }
        agent = DataGeneratorAgent()
        records = agent.run(model, 200)
        nulls = sum(1 for r in records if r["optional_score"] is None)
        assert nulls > 0

    @pytest.mark.asyncio
    async def test_respects_max_row_count_cap(self):
        agent = DataGeneratorAgent()
        model = self._build_simple_model()
        records = agent.run(model, 15000)
        assert len(records) == 10_000

    @pytest.mark.asyncio
    async def test_empty_model_returns_empty(self):
        agent = DataGeneratorAgent()
        records = agent.run({"column_models": [], "relationships": []}, 100)
        assert records == []

    @pytest.mark.asyncio
    async def test_text_template_generation(self):
        model = {
            "column_models": [
                {
                    "name": "notes",
                    "type": "text",
                    "nullable": False,
                    "null_rate": 0.0,
                    "distribution": "text_template",
                    "params": {"template": "notes_{i}", "avg_length": 20},
                }
            ],
            "relationships": [],
        }
        agent = DataGeneratorAgent()
        records = agent.run(model, 5)
        assert len(records) == 5
        assert all("notes" in r for r in records)


class TestQualityValidatorAgent:
    def _run_validation(self, sample_size: int = 50, gen_size: int = 100) -> dict:
        import numpy as np
        rng = np.random.default_rng(42)
        sample = [{"age": float(a), "score": float(s)} for a, s in
                  zip(rng.normal(35, 10, sample_size).clip(18, 80),
                      rng.normal(75, 15, sample_size).clip(0, 100))]
        generated = [{"age": float(a), "score": float(s)} for a, s in
                     zip(rng.normal(35, 10, gen_size).clip(18, 80),
                         rng.normal(75, 15, gen_size).clip(0, 100))]
        schema = {
            "columns": [
                {"name": "age", "type": "numeric"},
                {"name": "score", "type": "numeric"},
            ]
        }
        agent = QualityValidatorAgent()
        return agent.run(sample, generated, schema)

    @pytest.mark.asyncio
    async def test_fidelity_score_between_0_and_100(self):
        report = self._run_validation()
        assert 0.0 <= report["overall_fidelity_score"] <= 100.0

    @pytest.mark.asyncio
    async def test_column_scores_between_0_and_100(self):
        report = self._run_validation()
        for cs in report["column_scores"]:
            assert 0.0 <= cs["score"] <= 100.0

    @pytest.mark.asyncio
    async def test_flags_columns_below_threshold(self):
        import numpy as np
        rng = np.random.default_rng(42)
        sample = [{"age": float(a)} for a in rng.normal(35, 2, 50)]
        generated = [{"age": float(a)} for a in rng.uniform(200, 300, 100)]
        schema = {"columns": [{"name": "age", "type": "numeric"}]}
        agent = QualityValidatorAgent()
        report = agent.run(sample, generated, schema)
        assert len(report["flagged_columns"]) > 0

    @pytest.mark.asyncio
    async def test_categorical_fidelity(self):
        sample = [{"status": "active"} if i < 7 else {"status": "inactive"} for i in range(10)]
        generated = [{"status": "active"} if i < 14 else {"status": "inactive"} for i in range(20)]
        schema = {"columns": [{"name": "status", "type": "categorical"}]}
        agent = QualityValidatorAgent()
        report = agent.run(sample, generated, schema)
        assert report["overall_fidelity_score"] > 70.0

    @pytest.mark.asyncio
    async def test_empty_sample_returns_zero_fidelity(self):
        agent = QualityValidatorAgent()
        report = agent.run([], [], {})
        assert report["overall_fidelity_score"] == 0.0

    @pytest.mark.asyncio
    async def test_report_contains_required_keys(self):
        report = self._run_validation()
        assert "column_scores" in report
        assert "overall_fidelity_score" in report
        assert "threshold" in report
        assert "flagged_columns" in report
        assert "summary" in report

    @pytest.mark.asyncio
    async def test_no_flags_for_matching_distribution(self):
        report = self._run_validation(sample_size=50, gen_size=50)
        scores = [cs["score"] for cs in report["column_scores"]]
        assert any(s >= 50.0 for s in scores)

    @pytest.mark.asyncio
    async def test_below_threshold_flag_consistent_with_scores(self):
        report = self._run_validation()
        threshold = report["threshold"]
        for cs in report["column_scores"]:
            if cs["below_threshold"]:
                assert cs["score"] < threshold
            else:
                assert cs["score"] >= threshold


class TestPrivacyAuditorAgent:
    @pytest.mark.asyncio
    async def test_detects_exact_match_risk(self):
        records = [{"name": "Alice", "age": 30, "zip": "12345"}]
        agent = PrivacyAuditorAgent()
        report = agent.run(records, records)
        risk_types = [r["type"] for r in report["risks"]]
        assert "exact_match" in risk_types

    @pytest.mark.asyncio
    async def test_returns_valid_risk_level(self):
        agent = PrivacyAuditorAgent()
        report = agent.run([], [{"age": 30, "gender": "M"}])
        assert report["risk_level"] in ("safe", "low", "medium", "high")

    @pytest.mark.asyncio
    async def test_safe_for_no_risks(self):
        sample = [{"value": i} for i in range(10)]
        generated = [{"value": i + 100} for i in range(10)]
        agent = PrivacyAuditorAgent()
        report = agent.run(sample, generated)
        assert report["risk_level"] in ("safe", "low")

    @pytest.mark.asyncio
    async def test_detects_sensitive_field(self):
        generated = [{"ssn": "123-45-6789", "age": 30}]
        agent = PrivacyAuditorAgent()
        report = agent.run([], generated)
        risk_types = [r["type"] for r in report["risks"]]
        assert "sensitive_field_exposure" in risk_types

    @pytest.mark.asyncio
    async def test_detects_direct_identifier(self):
        generated = [{"name": "Alice Smith", "email": "alice@example.com"}]
        agent = PrivacyAuditorAgent()
        report = agent.run([], generated)
        risk_types = [r["type"] for r in report["risks"]]
        assert "direct_identifier_present" in risk_types

    @pytest.mark.asyncio
    async def test_report_has_recommendations(self):
        records = [{"ssn": "111-22-3333", "age": 25}]
        agent = PrivacyAuditorAgent()
        report = agent.run([], records)
        assert "recommendations" in report
        assert isinstance(report["recommendations"], list)

    @pytest.mark.asyncio
    async def test_empty_generated_returns_safe(self):
        agent = PrivacyAuditorAgent()
        report = agent.run([], [])
        assert report["risk_level"] == "safe"

    @pytest.mark.asyncio
    async def test_report_contains_summary(self):
        agent = PrivacyAuditorAgent()
        report = agent.run([], [{"age": 30}])
        assert "summary" in report
        assert len(report["summary"]) > 0

    @pytest.mark.asyncio
    async def test_quasi_identifier_detection(self):
        generated = [
            {"age": i, "zip": f"{10000 + i}", "gender": "M" if i % 2 == 0 else "F"}
            for i in range(200)
        ]
        agent = PrivacyAuditorAgent()
        report = agent.run([], generated)
        risk_types = [r["type"] for r in report["risks"]]
        assert "quasi_identifier_linkage" in risk_types

    @pytest.mark.asyncio
    async def test_high_exact_match_is_high_risk(self):
        records = [{"age": 25, "score": 80.0}] * 50
        agent = PrivacyAuditorAgent()
        report = agent.run(records, records)
        exact_risks = [r for r in report["risks"] if r["type"] == "exact_match"]
        assert any(r["severity"] == "high" for r in exact_risks)
