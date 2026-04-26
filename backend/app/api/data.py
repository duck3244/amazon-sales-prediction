"""데이터셋 관련 라우터."""

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.schemas import UploadResponse, DatasetSummary
from app.services.dataset_service import (
    save_uploaded_csv,
    summarize_csv,
    UploadTooLarge,
)
from app.settings import DATA_PATH

router = APIRouter(prefix='/api/data', tags=['data'])


@router.post('/upload', response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    """CSV 파일을 업로드하고 요약 통계를 반환."""
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='CSV 파일만 업로드 가능합니다.')

    try:
        path, size = save_uploaded_csv(file.file, file.filename)
    except UploadTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e))
    finally:
        await file.close()

    try:
        summary = summarize_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'CSV 파싱 실패: {e}')

    return UploadResponse(filename=file.filename, size_bytes=size, summary=summary)


@router.get('/summary', response_model=DatasetSummary)
async def current_summary() -> DatasetSummary:
    """현재 활성 데이터셋(`Amazon Sale Report.csv`) 요약."""
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail='활성 데이터셋이 없습니다. 먼저 업로드하세요.')
    return summarize_csv(DATA_PATH)
