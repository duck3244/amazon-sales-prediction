"""학습 잡 라우터."""

from __future__ import annotations
import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.schemas import (
    TrainRequest,
    TrainStartResponse,
    TrainStatusResponse,
    EpochPointResponse,
    JobLogResponse,
)
from app.services.job_store import job_store
from app.services.training_service import run_training

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/train', tags=['train'])


def _job_to_response(job) -> TrainStatusResponse:
    return TrainStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        params=job.params,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        history=[EpochPointResponse(**vars(p)) for p in job.history],
        best_val_loss=job.best_val_loss,
        early_stopped=job.early_stopped,
        test_loss=job.test_loss,
        error=job.error,
    )


@router.post('', response_model=TrainStartResponse)
async def start_training(req: TrainRequest) -> TrainStartResponse:
    try:
        job = job_store.acquire(req.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    asyncio.create_task(asyncio.to_thread(run_training, job))
    return TrainStartResponse(job_id=job.job_id, status=job.status)


@router.get('/active', response_model=TrainStatusResponse | None)
async def active_job() -> TrainStatusResponse | None:
    """현재/마지막 활성 잡 (페이지 새로고침 시 복원용)."""
    jobs = job_store.list()
    return _job_to_response(jobs[0]) if jobs else None


@router.get('/{job_id}', response_model=TrainStatusResponse)
async def get_job(job_id: str) -> TrainStatusResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'job {job_id} 없음')
    return _job_to_response(job)


@router.get('/{job_id}/log', response_model=JobLogResponse)
async def get_job_log(job_id: str, since: int = 0) -> JobLogResponse:
    """잡 로그를 incremental polling으로 노출.

    프런트는 마지막 받은 ``next_offset``을 ``since`` 인자로 전달해 새 라인만 가져온다.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f'job {job_id} 없음')
    total = len(job.log_lines)
    since = max(0, min(since, total))
    return JobLogResponse(
        job_id=job.job_id,
        total=total,
        next_offset=total,
        lines=job.log_lines[since:],
    )
