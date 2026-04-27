import io
import logging
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.agents.data_generator_agent import DataGeneratorAgent
from app.agents.distribution_modeler_agent import DistributionModelerAgent
from app.agents.privacy_auditor_agent import PrivacyAuditorAgent
from app.agents.quality_validator_agent import QualityValidatorAgent
from app.agents.schema_analyzer_agent import SchemaAnalyzerAgent

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    raw_input: dict
    sample_records: list[dict]
    domain: str
    row_count: int
    schema_analysis: dict | None
    distribution_model: dict | None
    generated_records: list[dict] | None
    validation_report: dict | None
    privacy_report: dict | None
    error: str | None


def _analyze_schema_node(state: WorkflowState) -> dict:
    try:
        agent = SchemaAnalyzerAgent()
        result = agent.run(state["raw_input"], state["sample_records"])
        return {"schema_analysis": result}
    except Exception as err:
        logger.exception("schema_analyzer_agent failed")
        return {"error": f"Schema analysis failed: {err}", "schema_analysis": None}


def _model_distributions_node(state: WorkflowState) -> dict:
    if state.get("error"):
        return {}
    try:
        agent = DistributionModelerAgent()
        result = agent.run(state["schema_analysis"] or {})
        return {"distribution_model": result}
    except Exception as err:
        logger.exception("distribution_modeler_agent failed")
        return {"error": f"Distribution modelling failed: {err}", "distribution_model": None}


def _generate_data_node(state: WorkflowState) -> dict:
    if state.get("error"):
        return {}
    try:
        agent = DataGeneratorAgent()
        records = agent.run(state["distribution_model"] or {}, state["row_count"])
        return {"generated_records": records}
    except Exception as err:
        logger.exception("data_generator_agent failed")
        return {"error": f"Data generation failed: {err}", "generated_records": None}


def _validate_quality_node(state: WorkflowState) -> dict:
    if state.get("error"):
        return {}
    try:
        agent = QualityValidatorAgent()
        report = agent.run(
            state["sample_records"],
            state["generated_records"] or [],
            state["schema_analysis"] or {},
        )
        return {"validation_report": report}
    except Exception as err:
        logger.exception("quality_validator_agent failed")
        return {"validation_report": None, "error": f"Quality validation failed: {err}"}


def _audit_privacy_node(state: WorkflowState) -> dict:
    if state.get("error"):
        return {}
    try:
        agent = PrivacyAuditorAgent()
        report = agent.run(
            state["sample_records"],
            state["generated_records"] or [],
        )
        return {"privacy_report": report}
    except Exception as err:
        logger.exception("privacy_auditor_agent failed")
        return {"privacy_report": None, "error": f"Privacy audit failed: {err}"}


def _build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)
    graph.add_node("analyze_schema", _analyze_schema_node)
    graph.add_node("model_distributions", _model_distributions_node)
    graph.add_node("generate_data", _generate_data_node)
    graph.add_node("validate_quality", _validate_quality_node)
    graph.add_node("audit_privacy", _audit_privacy_node)
    graph.add_edge(START, "analyze_schema")
    graph.add_edge("analyze_schema", "model_distributions")
    graph.add_edge("model_distributions", "generate_data")
    graph.add_edge("generate_data", "validate_quality")
    graph.add_edge("validate_quality", "audit_privacy")
    graph.add_edge("audit_privacy", END)
    return graph.compile()


_workflow_graph = _build_graph()


def parse_csv_to_records(csv_bytes: bytes) -> list[dict]:
    df = pd.read_csv(io.BytesIO(csv_bytes))
    return df.where(pd.notna(df), other=None).to_dict(orient="records")


async def run_generation_workflow(
    raw_input: dict,
    sample_records: list[dict],
    domain: str,
    row_count: int,
) -> WorkflowState:
    initial_state: WorkflowState = {
        "raw_input": raw_input,
        "sample_records": sample_records,
        "domain": domain,
        "row_count": row_count,
        "schema_analysis": None,
        "distribution_model": None,
        "generated_records": None,
        "validation_report": None,
        "privacy_report": None,
        "error": None,
    }
    result = await _workflow_graph.ainvoke(initial_state)
    return result
