import io
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Synthetic Data Generation Platform",
    layout="wide",
)

_DEFAULT_API_URL = "http://localhost:8010"


def _get_headers() -> dict:
    token = st.session_state.get("access_token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _api(method: str, path: str, **kwargs) -> requests.Response:
    base = st.session_state.get("api_url", _DEFAULT_API_URL).rstrip("/")
    url = f"{base}{path}"
    try:
        return requests.request(method, url, timeout=60, headers=_get_headers(), **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Verify the API base URL and that the server is running.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
        st.stop()


def _sidebar():
    with st.sidebar:
        st.title("Synthetic Data Platform")
        st.divider()

        st.subheader("Connection")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_url", _DEFAULT_API_URL),
            key="api_url_input",
        )
        st.session_state["api_url"] = api_url

        st.subheader("Authentication")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            resp = _api(
                "POST",
                "/auth/login",
                json={"username": username, "password": password},
            )
            if resp.status_code == 200:
                st.session_state["access_token"] = resp.json()["access_token"]
                st.session_state["logged_in_user"] = username
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")

        if st.button("Register", use_container_width=True):
            if not username or not password:
                st.error("Enter username and password to register")
            else:
                email = f"{username}@example.com"
                resp = _api(
                    "POST",
                    "/auth/register",
                    json={"username": username, "email": email, "password": password},
                )
                if resp.status_code == 201:
                    st.success("Registered. You can now log in.")
                else:
                    detail = resp.json().get("detail", "Registration failed")
                    st.error(str(detail))

        if st.session_state.get("access_token"):
            st.success(f"Signed in as: {st.session_state.get('logged_in_user', '')}")
            if st.button("Logout", use_container_width=True):
                st.session_state.pop("access_token", None)
                st.session_state.pop("logged_in_user", None)
                st.rerun()

        st.divider()
        st.subheader("Groq API Key (optional)")
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            key="groq_key_input",
            help="Used for this session only. Never stored or transmitted beyond the API.",
        )
        if groq_key:
            st.session_state["groq_api_key"] = groq_key

        st.divider()
        page = st.selectbox(
            "Navigate",
            ["Generate Data", "Job History", "Analytics", "Data Preview"],
            key="page_selector",
        )
    return page


def _page_generate():
    st.header("Generate Synthetic Data")

    if not st.session_state.get("access_token"):
        st.warning("Please log in to use this feature.")
        return

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Schema Definition")
        input_mode = st.radio("Input mode", ["JSON Schema", "Upload CSV"], horizontal=True)

        schema_json_str = None
        csv_bytes = None

        if input_mode == "JSON Schema":
            default_schema = json.dumps(
                {
                    "columns": [
                        {"name": "age", "type": "integer", "min": 18, "max": 90},
                        {"name": "income", "type": "float", "min": 20000, "max": 150000},
                        {"name": "gender", "type": "categorical", "categories": ["M", "F", "Other"]},
                        {"name": "active", "type": "boolean"},
                    ]
                },
                indent=2,
            )
            schema_text = st.text_area("JSON Schema", value=default_schema, height=250)
            try:
                json.loads(schema_text)
                schema_json_str = schema_text
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
        else:
            uploaded = st.file_uploader("Upload sample CSV", type=["csv"])
            if uploaded:
                csv_bytes = uploaded.read()
                preview_df = pd.read_csv(io.BytesIO(csv_bytes))
                st.write("Preview (first 5 rows):")
                st.dataframe(preview_df.head())

    with col_right:
        st.subheader("Generation Settings")
        job_name = st.text_input("Job Name", value=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        domain = st.selectbox("Domain", ["healthcare", "finance", "retail", "hr", "iot", "custom"])
        row_count = st.number_input("Row Count", min_value=1, max_value=10000, value=100, step=50)

    if st.button("Run Generation", type="primary", use_container_width=True):
        if not job_name:
            st.error("Job name is required")
            return
        if input_mode == "JSON Schema" and not schema_json_str:
            st.error("Valid JSON schema is required")
            return
        if input_mode == "Upload CSV" and not csv_bytes:
            st.error("CSV file is required")
            return

        with st.spinner("Creating job..."):
            form_data: dict = {
                "job_name": job_name,
                "domain": domain,
                "row_count": str(row_count),
            }
            files = None
            if input_mode == "JSON Schema":
                form_data["schema_json"] = schema_json_str
                resp = _api("POST", "/jobs", data=form_data)
            else:
                files = {"sample_file": ("sample.csv", csv_bytes, "text/csv")}
                resp = _api("POST", "/jobs", data=form_data, files=files)

        if resp.status_code != 201:
            st.error(f"Failed to create job: {resp.json().get('detail', resp.text)}")
            return

        job_id = resp.json()["id"]
        st.info(f"Job created: {job_id}")

        with st.spinner("Running workflow (schema analysis, distribution modelling, generation, validation, privacy audit)..."):
            run_resp = _api("POST", f"/jobs/{job_id}/run")

        if run_resp.status_code != 200:
            st.error(f"Workflow failed: {run_resp.json().get('detail', run_resp.text)}")
            return

        job = run_resp.json()
        if job["status"] == "completed":
            st.success("Generation complete")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows Generated", job.get("row_count_generated", 0))
            m2.metric("Fidelity Score", f"{job.get('fidelity_score') or 0:.1f}")
            m3.metric("Privacy Risk", job.get("privacy_risk_level", "N/A").upper())
            m4.metric("Domain", job["domain"].upper())

            ds_resp = _api("GET", f"/jobs/{job_id}/dataset")
            if ds_resp.status_code == 200:
                ds = ds_resp.json()
                records = ds.get("data", [])
                if records:
                    df = pd.DataFrame(records)
                    st.subheader("Data Preview")
                    st.dataframe(df.head(20), use_container_width=True)

                    csv_dl = _api("GET", f"/jobs/{job_id}/dataset/csv")
                    if csv_dl.status_code == 200:
                        st.download_button(
                            "Download CSV",
                            data=csv_dl.content,
                            file_name=f"synthetic_{job_id}.csv",
                            mime="text/csv",
                        )

                val_report = ds.get("validation_report", {})
                priv_report = ds.get("privacy_report", {})

                with st.expander("Validation Report"):
                    st.json(val_report)
                with st.expander("Privacy Report"):
                    st.json(priv_report)
        else:
            st.error(f"Job failed: {job.get('error_message', 'Unknown error')}")


def _page_job_history():
    st.header("Job History")

    if not st.session_state.get("access_token"):
        st.warning("Please log in to view job history.")
        return

    col1, col2 = st.columns(2)
    with col1:
        domain_filter = st.selectbox(
            "Filter by Domain",
            ["All", "healthcare", "finance", "retail", "hr", "iot", "custom"],
        )
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "pending", "processing", "completed", "failed"],
        )

    params: dict = {}
    if domain_filter != "All":
        params["domain"] = domain_filter
    if status_filter != "All":
        params["status"] = status_filter

    resp = _api("GET", "/jobs", params=params)
    if resp.status_code != 200:
        st.error("Failed to load jobs")
        return

    jobs = resp.json()
    if not jobs:
        st.info("No jobs found.")
        return

    rows = []
    for j in jobs:
        rows.append(
            {
                "Job ID": j["id"][:8] + "...",
                "Name": j["job_name"],
                "Domain": j["domain"],
                "Status": j["status"],
                "Rows Requested": j["row_count_requested"],
                "Rows Generated": j.get("row_count_generated", ""),
                "Fidelity": f"{j['fidelity_score']:.1f}" if j.get("fidelity_score") is not None else "",
                "Privacy Risk": j.get("privacy_risk_level", ""),
                "Created": j["created_at"][:19].replace("T", " "),
                "_id": j["id"],
            }
        )

    df = pd.DataFrame(rows)
    display_cols = [c for c in df.columns if c != "_id"]
    st.dataframe(df[display_cols], use_container_width=True)
    st.caption(f"Total: {len(jobs)} job(s)")


def _page_analytics():
    st.header("Analytics")

    if not st.session_state.get("access_token"):
        st.warning("Please log in to view analytics.")
        return

    stats_resp = _api("GET", "/dashboard/stats")
    jobs_resp = _api("GET", "/jobs")

    if stats_resp.status_code != 200 or jobs_resp.status_code != 200:
        st.error("Failed to load analytics data")
        return

    stats = stats_resp.json()
    jobs = jobs_resp.json()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Jobs", stats["total_jobs"])
    m2.metric("Completed", stats["completed_jobs"])
    m3.metric("Failed", stats["failed_jobs"])
    avg_f = stats.get("avg_fidelity_score")
    m4.metric("Avg Fidelity", f"{avg_f:.1f}" if avg_f is not None else "N/A")

    if not jobs:
        st.info("No data yet. Generate some datasets first.")
        return

    plotly_template = "plotly_dark"

    col1, col2 = st.columns(2)

    with col1:
        domain_data = stats.get("jobs_by_domain", {})
        if domain_data:
            fig = px.bar(
                x=list(domain_data.keys()),
                y=list(domain_data.values()),
                labels={"x": "Domain", "y": "Count"},
                title="Jobs by Domain",
                template=plotly_template,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_data = stats.get("jobs_by_risk_level", {})
        if risk_data:
            fig = px.pie(
                names=list(risk_data.keys()),
                values=list(risk_data.values()),
                title="Privacy Risk Breakdown",
                template=plotly_template,
            )
            st.plotly_chart(fig, use_container_width=True)

    fidelity_scores = [j["fidelity_score"] for j in jobs if j.get("fidelity_score") is not None]
    if fidelity_scores:
        fig = px.histogram(
            x=fidelity_scores,
            nbins=20,
            labels={"x": "Fidelity Score", "y": "Count"},
            title="Fidelity Score Distribution",
            template=plotly_template,
        )
        st.plotly_chart(fig, use_container_width=True)

    completed_jobs = [j for j in jobs if j.get("created_at")]
    if completed_jobs:
        time_data = pd.DataFrame(
            {
                "date": [j["created_at"][:10] for j in completed_jobs],
                "count": 1,
            }
        ).groupby("date").sum().reset_index()
        fig = px.line(
            time_data,
            x="date",
            y="count",
            labels={"date": "Date", "count": "Jobs"},
            title="Generation Volume Over Time",
            template=plotly_template,
        )
        st.plotly_chart(fig, use_container_width=True)


def _page_data_preview():
    st.header("Data Preview")

    if not st.session_state.get("access_token"):
        st.warning("Please log in to preview data.")
        return

    resp = _api("GET", "/jobs", params={"status": "completed"})
    if resp.status_code != 200:
        st.error("Failed to load completed jobs")
        return

    jobs = resp.json()
    completed = [j for j in jobs if j["status"] == "completed"]

    if not completed:
        st.info("No completed jobs yet. Run a generation job first.")
        return

    job_options = {f"{j['job_name']} ({j['id'][:8]}...)": j["id"] for j in completed}
    selected_label = st.selectbox("Select Job", list(job_options.keys()))
    job_id = job_options[selected_label]

    ds_resp = _api("GET", f"/jobs/{job_id}/dataset")
    if ds_resp.status_code != 200:
        st.error("Failed to load dataset")
        return

    ds = ds_resp.json()
    records = ds.get("data", [])

    if records:
        df = pd.DataFrame(records)
        st.subheader(f"Generated Data ({len(records)} rows)")
        st.dataframe(df, use_container_width=True)

        csv_dl = _api("GET", f"/jobs/{job_id}/dataset/csv")
        if csv_dl.status_code == 200:
            st.download_button(
                "Download as CSV",
                data=csv_dl.content,
                file_name=f"synthetic_{job_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.warning("No records found in this dataset.")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Validation Report", expanded=True):
            val_report = ds.get("validation_report", {})
            if val_report:
                overall = val_report.get("overall_fidelity_score", 0)
                st.metric("Overall Fidelity Score", f"{overall:.1f}")
                scores = val_report.get("column_scores", [])
                if scores:
                    scores_df = pd.DataFrame(scores)
                    if "score" in scores_df.columns:
                        st.dataframe(scores_df[["column", "type", "score", "below_threshold"]], use_container_width=True)
                flagged = val_report.get("flagged_columns", [])
                if flagged:
                    st.warning(f"Flagged columns: {', '.join(flagged)}")
            else:
                st.info("No validation report available.")

    with col2:
        with st.expander("Privacy Report", expanded=True):
            priv_report = ds.get("privacy_report", {})
            if priv_report:
                risk_level = priv_report.get("risk_level", "N/A").upper()
                st.metric("Risk Level", risk_level)
                risks = priv_report.get("risks", [])
                if risks:
                    for risk in risks:
                        st.warning(f"{risk.get('type', '')}: {risk.get('recommendation', '')}")
                else:
                    st.success("No privacy risks detected.")
            else:
                st.info("No privacy report available.")


def main():
    page = _sidebar()
    if page == "Generate Data":
        _page_generate()
    elif page == "Job History":
        _page_job_history()
    elif page == "Analytics":
        _page_analytics()
    elif page == "Data Preview":
        _page_data_preview()


if __name__ == "__main__":
    main()
