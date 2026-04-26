"""예측/평가 라우터."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas import (
    CompareEntry,
    CompareResponse,
    EvaluationResponse,
    SinglePredictRequest,
    SinglePredictResponse,
    TrainedModelInfo,
)
from app.services.batch_predict_service import (
    batch_predict_csv,
    df_to_csv_string,
)
from app.services.dataset_service import save_uploaded_csv, UploadTooLarge
from app.services.predict_service import (
    ModelNotFound,
    evaluate_test_set,
    predict_single,
)
from app.settings import MODELS_DIR, UPLOADS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/api/predict', tags=['predict'])
models_router = APIRouter(prefix='/api/models', tags=['models'])


@models_router.get('', response_model=List[TrainedModelInfo])
async def list_models() -> List[TrainedModelInfo]:
    out: List[TrainedModelInfo] = []
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        cfg_path = sub / 'config.json'
        ckpt_path = sub / 'best_model.pth'
        if not cfg_path.exists():
            continue
        with cfg_path.open() as f:
            cfg = json.load(f)
        out.append(TrainedModelInfo(
            model_type=cfg.get('model_type', sub.name),
            input_dim=cfg.get('input_dim', 0),
            hidden_dims=cfg.get('hidden_dims', []),
            dropout=cfg.get('dropout', 0.0),
            best_val_loss=cfg.get('best_val_loss', 0.0),
            test_loss=cfg.get('test_loss'),
            available=ckpt_path.exists(),
        ))
    return out


@router.post('/compare', response_model=CompareResponse)
async def compare_models() -> CompareResponse:
    """학습된 모든 모델을 동일 X_test/y_test로 평가하여 한 번에 비교."""
    entries: List[CompareEntry] = []
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        cfg_path = sub / 'config.json'
        ckpt_path = sub / 'best_model.pth'
        if not cfg_path.exists() or not ckpt_path.exists():
            continue
        model_type = sub.name
        try:
            r = evaluate_test_set(model_type)
            entries.append(CompareEntry(
                model_type=model_type,
                mae=r['mae'], mse=r['mse'], rmse=r['rmse'], r2=r['r2'],
                n_samples=r['n_samples'],
            ))
        except Exception as e:
            logger.exception('compare: %s 실패', model_type)
            entries.append(CompareEntry(model_type=model_type, error=str(e)))
    return CompareResponse(entries=entries)


@router.post('/batch', response_model=EvaluationResponse)
async def batch_evaluate(model_type: str = 'basic') -> EvaluationResponse:
    try:
        result = evaluate_test_set(model_type)
    except ModelNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{type(e).__name__}: {e}')
    return EvaluationResponse(**result)


@router.post('/single', response_model=SinglePredictResponse)
async def single(req: SinglePredictRequest, model_type: str = 'basic') -> SinglePredictResponse:
    try:
        pred = predict_single(model_type, req.features)
    except ModelNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return SinglePredictResponse(prediction=pred)


@router.post('/batch_csv', response_class=StreamingResponse)
async def batch_csv(
    file: UploadFile = File(...),
    model_type: str = 'basic',
):
    """업로드된 CSV 전체에 대해 학습된 preprocessor + 모델로 예측 후 CSV 다운로드."""
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='CSV 파일만 업로드 가능합니다.')

    # activate=False — 학습용 활성 CSV는 덮어쓰지 않는다.
    try:
        path, _size = save_uploaded_csv(file.file, file.filename, activate=False)
    except UploadTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e))
    finally:
        await file.close()

    try:
        out_df, meta = batch_predict_csv(path, model_type)
    except ModelNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception('Batch predict failed')
        raise HTTPException(status_code=500, detail=f'{type(e).__name__}: {e}')

    csv_text = df_to_csv_string(out_df)
    out_name = f'{Path(file.filename).stem}__predicted_{model_type}.csv'
    headers = {
        'Content-Disposition': f'attachment; filename="{out_name}"',
        'X-Prediction-Rows': str(meta['rows']),
        'X-Prediction-Elapsed': str(meta['elapsed_seconds']),
        'X-Prediction-Model': meta['model_type'],
    }
    return StreamingResponse(
        iter([csv_text]),
        media_type='text/csv',
        headers=headers,
    )
