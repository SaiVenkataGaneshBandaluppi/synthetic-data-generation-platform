import csv
import io
import json
import logging
import re
import uuid
from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import GeneratedDataset, GenerationJob, User
from app.services.rate_limiter import limiter
from app.services.utils import (
    contains_injection,
    create_access_token,
    decode_access_token,
    hash_password,
    sanitize_text,
    validate_column_count,
    validate_domain,
    validate_row_count,
    verify_password,
)
from app.services.workflow import parse_csv_to_records, run_generation_workflow

logger = logging.getLogger(__name__)
router = APIRouter()

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_MAX_FILE_BYTES = 5 * 1024 * 1024


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        v = sanitize_text(v)
        if not v or len(v) < 3 or len(v) > 50:
            raise ValueError("username must be 3 to 50 characters")
        if contains_injection(v):
            raise ValueError("invalid username")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError("invalid email format")
        if len(v) > 255:
            raise ValueError("email too long")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        if len(v) > 128:
            raise ValueError("password too long")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class JobResponse(BaseModel):
    id: str
    job_name: str
    domain: str
    row_count_requested: int
    row_count_generated: int | None
    fidelity_score: float | None
    privacy_risk_level: str | None
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DashboardStats(BaseModel):
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_fidelity_score: float | None
    jobs_by_domain: dict
    jobs_by_risk_level: dict


async def _get_current_user(
    request: Request, db: AsyncSession = Depends(get_db)
) -> User:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header[7:]
    try:
        payload = decode_access_token(token)
    except ValueError as err:
        raise HTTPException(status_code=401, detail=str(err)) from err
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    result = await db.execute(select(User).where(User.username == subject))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _job_to_response(job: GenerationJob) -> JobResponse:
    return JobResponse(
        id=str(job.id),
        job_name=job.job_name,
        domain=job.domain,
        row_count_requested=job.row_count_requested,
        row_count_generated=job.row_count_generated,
        fidelity_score=job.fidelity_score,
        privacy_risk_level=job.privacy_risk_level,
        status=job.status,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    return {"status": "ok", "service": "synthetic-data-generation-platform"}


@router.post("/auth/register", status_code=201)
@limiter.limit(lambda: f"{settings.AUTH_RATE_LIMIT_PER_MINUTE}/minute")
async def register(
    request: Request,
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    existing = await db.execute(
        select(User).where(
            (User.username == body.username) | (User.email == body.email)
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="Username or email already registered")
    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        created_at=user.created_at,
    )


@router.post("/auth/login")
@limiter.limit(lambda: f"{settings.AUTH_RATE_LIMIT_PER_MINUTE}/minute")
async def login(
    request: Request,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(subject=user.username)
    return TokenResponse(access_token=token)


@router.post("/jobs", status_code=201)
@limiter.limit("60/minute")
async def create_job(
    request: Request,
    job_name: str = Form(...),
    domain: str = Form(...),
    row_count: int = Form(...),
    schema_json: str | None = Form(None),
    sample_file: UploadFile | None = File(None),
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    job_name = sanitize_text(job_name)
    if not job_name or len(job_name) > 255:
        raise HTTPException(status_code=422, detail="job_name must be 1 to 255 characters")
    if contains_injection(job_name):
        raise HTTPException(status_code=422, detail="invalid job_name")
    try:
        domain = validate_domain(domain)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    try:
        row_count = validate_row_count(row_count)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    source_schema: dict | None = None
    if schema_json:
        if contains_injection(schema_json):
            raise HTTPException(status_code=422, detail="invalid schema_json")
        try:
            parsed = json.loads(schema_json)
            if "columns" in parsed:
                validate_column_count(parsed["columns"])
            source_schema = parsed
        except (json.JSONDecodeError, ValueError) as err:
            raise HTTPException(status_code=422, detail=f"invalid schema_json: {err}") from err

    if sample_file:
        content_type = sample_file.content_type or ""
        if "csv" not in content_type and "text" not in content_type:
            raise HTTPException(status_code=422, detail="sample_file must be a CSV")
        raw_bytes = await sample_file.read()
        if len(raw_bytes) > _MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 5MB)")
        source_schema = source_schema or {}
        source_schema["_csv_bytes"] = raw_bytes.decode("utf-8", errors="replace")

    if not source_schema:
        raise HTTPException(
            status_code=422, detail="Either schema_json or sample_file is required"
        )

    job = GenerationJob(
        user_id=current_user.id,
        job_name=job_name,
        domain=domain,
        source_schema=source_schema,
        row_count_requested=row_count,
        status="pending",
    )
    db.add(job)
    await db.flush()
    return _job_to_response(job)


@router.get("/jobs")
@limiter.limit("60/minute")
async def list_jobs(
    request: Request,
    domain: str | None = None,
    status: str | None = None,
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[JobResponse]:
    query = select(GenerationJob).where(GenerationJob.user_id == current_user.id)
    if domain:
        query = query.where(GenerationJob.domain == domain)
    if status:
        query = query.where(GenerationJob.status == status)
    query = query.order_by(GenerationJob.created_at.desc())
    result = await db.execute(query)
    jobs = result.scalars().all()
    return [_job_to_response(j) for j in jobs]


@router.get("/jobs/{job_id}")
@limiter.limit("60/minute")
async def get_job(
    request: Request,
    job_id: str,
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    try:
        uid = uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail="invalid job_id") from err
    result = await db.execute(
        select(GenerationJob).where(
            GenerationJob.id == uid, GenerationJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.delete("/jobs/{job_id}", status_code=204)
@limiter.limit("60/minute")
async def delete_job(
    request: Request,
    job_id: str,
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        uid = uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail="invalid job_id") from err
    result = await db.execute(
        select(GenerationJob).where(
            GenerationJob.id == uid, GenerationJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    await db.delete(job)


@router.post("/jobs/{job_id}/run")
@limiter.limit("10/minute")
async def run_job(
    request: Request,
    job_id: str,
    x_groq_key: str = Header(default=""),
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    try:
        uid = uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail="invalid job_id") from err
    result = await db.execute(
        select(GenerationJob).where(
            GenerationJob.id == uid, GenerationJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == "processing":
        raise HTTPException(status_code=409, detail="Job is already running")

    job.status = "processing"
    job.updated_at = datetime.utcnow()
    await db.flush()

    source = job.source_schema or {}
    csv_text = source.pop("_csv_bytes", None)
    sample_records: list[dict] = []
    if csv_text:
        try:
            sample_records = parse_csv_to_records(csv_text.encode("utf-8"))
        except Exception as err:
            logger.warning("CSV parse error: %s", err)

    try:
        workflow_result = await run_generation_workflow(
            raw_input=source,
            sample_records=sample_records,
            domain=job.domain,
            row_count=job.row_count_requested,
            groq_api_key=x_groq_key,
        )
    except Exception as err:
        logger.exception("Workflow execution error for job %s", job_id)
        job.status = "failed"
        job.error_message = "Internal workflow error"
        job.updated_at = datetime.utcnow()
        raise HTTPException(status_code=500, detail="Workflow execution failed") from err

    if workflow_result.get("error"):
        job.status = "failed"
        job.error_message = str(workflow_result["error"])[:1000]
        job.updated_at = datetime.utcnow()
        return _job_to_response(job)

    generated = workflow_result.get("generated_records") or []
    validation_report = workflow_result.get("validation_report") or {}
    privacy_report = workflow_result.get("privacy_report") or {}

    job.status = "completed"
    job.row_count_generated = len(generated)
    job.fidelity_score = validation_report.get("overall_fidelity_score")
    job.privacy_risk_level = privacy_report.get("risk_level")
    job.source_schema = workflow_result.get("schema_analysis")
    job.distribution_model = workflow_result.get("distribution_model")
    job.error_message = None
    job.updated_at = datetime.utcnow()

    stored_records = generated[:1000]
    existing_ds = await db.execute(
        select(GeneratedDataset).where(GeneratedDataset.job_id == uid)
    )
    dataset = existing_ds.scalar_one_or_none()
    if dataset is None:
        dataset = GeneratedDataset(
            job_id=uid,
            data=stored_records,
            validation_report=validation_report,
            privacy_report=privacy_report,
        )
        db.add(dataset)
    else:
        dataset.data = stored_records
        dataset.validation_report = validation_report
        dataset.privacy_report = privacy_report

    await db.flush()
    return _job_to_response(job)


@router.get("/jobs/{job_id}/dataset")
@limiter.limit("60/minute")
async def get_dataset(
    request: Request,
    job_id: str,
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        uid = uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail="invalid job_id") from err
    result = await db.execute(
        select(GenerationJob).where(
            GenerationJob.id == uid, GenerationJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job has not completed successfully")

    ds_result = await db.execute(
        select(GeneratedDataset).where(GeneratedDataset.job_id == uid)
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "job_id": job_id,
        "row_count": len(dataset.data or []),
        "data": dataset.data,
        "validation_report": dataset.validation_report,
        "privacy_report": dataset.privacy_report,
    }


@router.get("/jobs/{job_id}/dataset/csv")
@limiter.limit("60/minute")
async def download_dataset_csv(
    request: Request,
    job_id: str,
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        uid = uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=422, detail="invalid job_id") from err
    result = await db.execute(
        select(GenerationJob).where(
            GenerationJob.id == uid, GenerationJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job has not completed successfully")

    ds_result = await db.execute(
        select(GeneratedDataset).where(GeneratedDataset.job_id == uid)
    )
    dataset = ds_result.scalar_one_or_none()
    if dataset is None or not dataset.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    records = dataset.data
    buf = io.StringIO()
    if records:
        writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="dataset_{job_id}.csv"'},
    )


@router.get("/dashboard/stats")
@limiter.limit("60/minute")
async def dashboard_stats(
    request: Request,
    current_user: User = Depends(_get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DashboardStats:
    result = await db.execute(
        select(GenerationJob).where(GenerationJob.user_id == current_user.id)
    )
    jobs = result.scalars().all()

    total = len(jobs)
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")

    fidelity_scores = [j.fidelity_score for j in jobs if j.fidelity_score is not None]
    avg_fidelity = round(sum(fidelity_scores) / len(fidelity_scores), 2) if fidelity_scores else None

    by_domain: dict = {}
    by_risk: dict = {}
    for j in jobs:
        by_domain[j.domain] = by_domain.get(j.domain, 0) + 1
        if j.privacy_risk_level:
            by_risk[j.privacy_risk_level] = by_risk.get(j.privacy_risk_level, 0) + 1

    return DashboardStats(
        total_jobs=total,
        completed_jobs=completed,
        failed_jobs=failed,
        avg_fidelity_score=avg_fidelity,
        jobs_by_domain=by_domain,
        jobs_by_risk_level=by_risk,
    )
