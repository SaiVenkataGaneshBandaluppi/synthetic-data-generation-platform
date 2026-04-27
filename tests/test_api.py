import json

import pytest

_SIMPLE_SCHEMA = json.dumps({
    "columns": [
        {"name": "age", "type": "integer", "min": 18, "max": 90, "mean": 40, "std": 15},
        {"name": "income", "type": "float", "min": 20000, "max": 200000, "mean": 60000, "std": 25000},
        {"name": "category", "type": "categorical", "categories": ["A", "B", "C"]},
    ]
})


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_ok_status(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        response = await client.get("/health")
        assert response.status_code != 401


class TestAuthEndpoints:
    @pytest.mark.asyncio
    async def test_register_creates_user(self, client):
        response = await client.post(
            "/auth/register",
            json={
                "username": "newreguser",
                "email": "newreguser@example.com",
                "password": "SecurePass123!",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "newreguser"
        assert "id" in data
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_returns_user_fields(self, client):
        response = await client.post(
            "/auth/register",
            json={
                "username": "fieldtestuser",
                "email": "fieldtest@example.com",
                "password": "SecurePass123!",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "username" in data
        assert "email" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_register_duplicate_fails(self, client, registered_user):
        response = await client.post(
            "/auth/register",
            json={
                "username": "testuser",
                "email": "testuser@example.com",
                "password": "TestPassword123!",
            },
        )
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_login_valid_credentials_returns_jwt(self, client, registered_user):
        response = await client.post(
            "/auth/login",
            json={"username": "testuser", "password": "TestPassword123!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_password_returns_401(self, client, registered_user):
        response = await client.post(
            "/auth/login",
            json={"username": "testuser", "password": "WrongPassword!"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_unknown_user_returns_401(self, client):
        response = await client.post(
            "/auth/login",
            json={"username": "nobody_exists_xyz", "password": "SomePass123!"},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_register_invalid_email_rejected(self, client):
        response = await client.post(
            "/auth/register",
            json={
                "username": "bademail_user",
                "email": "not-an-email",
                "password": "SecurePass123!",
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_register_short_password_rejected(self, client):
        response = await client.post(
            "/auth/register",
            json={
                "username": "shortpassuser",
                "email": "shortpass@example.com",
                "password": "abc",
            },
        )
        assert response.status_code == 422


class TestJobsEndpoints:
    @pytest.mark.asyncio
    async def test_create_job_without_auth_returns_401(self, client):
        response = await client.post(
            "/jobs",
            data={
                "job_name": "test",
                "domain": "healthcare",
                "row_count": "100",
                "schema_json": _SIMPLE_SCHEMA,
            },
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_job_with_auth_succeeds(self, client, auth_headers):
        response = await client.post(
            "/jobs",
            data={
                "job_name": "my_test_job",
                "domain": "healthcare",
                "row_count": "50",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["job_name"] == "my_test_job"
        assert data["domain"] == "healthcare"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_job_invalid_domain_rejected(self, client, auth_headers):
        response = await client.post(
            "/jobs",
            data={
                "job_name": "bad_domain_job",
                "domain": "invalid_domain_xyz",
                "row_count": "50",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_job_row_count_too_high_rejected(self, client, auth_headers):
        response = await client.post(
            "/jobs",
            data={
                "job_name": "huge_job",
                "domain": "healthcare",
                "row_count": "99999",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_create_job_no_schema_rejected(self, client, auth_headers):
        response = await client.post(
            "/jobs",
            data={
                "job_name": "no_schema_job",
                "domain": "healthcare",
                "row_count": "50",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_jobs_without_auth_returns_401(self, client):
        response = await client.get("/jobs")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_jobs_returns_only_user_jobs(self, client, auth_headers):
        response = await client.post(
            "/auth/register",
            json={
                "username": "list_bola_user",
                "email": "listbola@example.com",
                "password": "SecurePass123!",
            },
        )
        assert response.status_code == 201
        login_resp = await client.post(
            "/auth/login",
            json={"username": "list_bola_user", "password": "SecurePass123!"},
        )
        other_token = login_resp.json()["access_token"]
        other_headers = {"Authorization": f"Bearer {other_token}"}

        await client.post(
            "/jobs",
            data={
                "job_name": "other_user_job",
                "domain": "finance",
                "row_count": "10",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=other_headers,
        )

        response = await client.get("/jobs", headers=auth_headers)
        assert response.status_code == 200
        jobs = response.json()
        job_names = [j["job_name"] for j in jobs]
        assert "other_user_job" not in job_names

    @pytest.mark.asyncio
    async def test_get_job_returns_404_for_another_users_job(self, client, auth_headers):
        reg = await client.post(
            "/auth/register",
            json={
                "username": "bola_user2",
                "email": "bola2@example.com",
                "password": "SecurePass123!",
            },
        )
        assert reg.status_code == 201
        login = await client.post(
            "/auth/login",
            json={"username": "bola_user2", "password": "SecurePass123!"},
        )
        other_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        create = await client.post(
            "/jobs",
            data={
                "job_name": "bola_victim_job",
                "domain": "retail",
                "row_count": "10",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=other_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]

        response = await client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_own_job_succeeds(self, client, auth_headers):
        create = await client.post(
            "/jobs",
            data={
                "job_name": "own_job_test",
                "domain": "hr",
                "row_count": "20",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]

        response = await client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    @pytest.mark.asyncio
    async def test_delete_own_job_succeeds(self, client, auth_headers):
        create = await client.post(
            "/jobs",
            data={
                "job_name": "delete_test_job",
                "domain": "iot",
                "row_count": "10",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]

        response = await client.delete(f"/jobs/{job_id}", headers=auth_headers)
        assert response.status_code == 204

        get_resp = await client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_run_job_returns_completed(self, client, auth_headers):
        create = await client.post(
            "/jobs",
            data={
                "job_name": "run_test_job",
                "domain": "healthcare",
                "row_count": "10",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]

        run = await client.post(f"/jobs/{job_id}/run", headers=auth_headers)
        assert run.status_code == 200
        data = run.json()
        assert data["status"] == "completed"
        assert data["row_count_generated"] is not None

    @pytest.mark.asyncio
    async def test_get_dataset_after_run(self, client, auth_headers):
        create = await client.post(
            "/jobs",
            data={
                "job_name": "dataset_fetch_job",
                "domain": "finance",
                "row_count": "15",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]
        await client.post(f"/jobs/{job_id}/run", headers=auth_headers)

        response = await client.get(f"/jobs/{job_id}/dataset", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 15

    @pytest.mark.asyncio
    async def test_get_dataset_csv_download(self, client, auth_headers):
        create = await client.post(
            "/jobs",
            data={
                "job_name": "csv_download_job",
                "domain": "retail",
                "row_count": "5",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]
        await client.post(f"/jobs/{job_id}/run", headers=auth_headers)

        response = await client.get(f"/jobs/{job_id}/dataset/csv", headers=auth_headers)
        assert response.status_code == 200
        assert "text/csv" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_dashboard_stats_returns_aggregated_metrics(self, client, auth_headers):
        response = await client.get("/dashboard/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_jobs" in data
        assert "completed_jobs" in data
        assert "failed_jobs" in data
        assert "jobs_by_domain" in data
        assert "jobs_by_risk_level" in data

    @pytest.mark.asyncio
    async def test_dashboard_stats_without_auth_returns_401(self, client):
        response = await client.get("/dashboard/stats")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_dataset_before_run_returns_409(self, client, auth_headers):
        create = await client.post(
            "/jobs",
            data={
                "job_name": "not_run_yet_job",
                "domain": "hr",
                "row_count": "10",
                "schema_json": _SIMPLE_SCHEMA,
            },
            headers=auth_headers,
        )
        assert create.status_code == 201
        job_id = create.json()["id"]

        response = await client.get(f"/jobs/{job_id}/dataset", headers=auth_headers)
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_invalid_bearer_token_returns_401(self, client):
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = await client.get("/jobs", headers=headers)
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_security_headers_present(self, client):
        response = await client.get("/health")
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
