"""배치 예측 서비스 — 업로드된 CSV를 학습된 preprocessor로 변환 후 예측."""

from __future__ import annotations
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.services.predict_service import load_trained_model, ModelNotFound
from app.settings import PREPROC_PATH

# backend/ 모듈 경로
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data_preprocessing import DataPreprocessor  # noqa: E402

logger = logging.getLogger(__name__)


def _load_preprocessor() -> DataPreprocessor:
    if not PREPROC_PATH.exists():
        raise ModelNotFound('전처리기가 없습니다 — 먼저 전처리/학습을 수행하세요.')
    pp = DataPreprocessor()
    pp.load_preprocessor(str(PREPROC_PATH))
    return pp


@torch.no_grad()
def batch_predict_csv(csv_path: Path, model_type: str) -> tuple[pd.DataFrame, dict]:
    """raw CSV → 예측. (결과 DataFrame, 메타) 반환."""
    pp = _load_preprocessor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_trained_model(model_type, device)

    raw = pd.read_csv(csv_path, low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]

    # transform이 column 누락 시 친절한 에러를 던지도록 사전 검증
    required = set(pp.feature_names)
    # 'Amount' 같은 타겟은 raw에 있어도 무방 (transform이 무시)
    raw_cols = set(raw.columns)
    # date_columns 처리 후 새 feature가 만들어지므로 정확한 누락 검증은 어려움.
    # 대신 transform 후 feature_names가 모두 채워지는지 확인.
    started = time.time()
    try:
        X = pp.transform(raw, date_columns=['Date'])
    except KeyError as e:
        raise ValueError(f'필수 컬럼 누락: {e}')

    missing = required - set(X.columns)
    if missing:
        raise ValueError(f'학습 시 사용된 특성이 누락되었습니다: {sorted(missing)}')

    X = X[pp.feature_names].astype(np.float32).values

    # 배치 예측
    preds = []
    bs = 1024
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        preds.append(model(xb).cpu().numpy().reshape(-1))
    pred = np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)
    elapsed = time.time() - started

    # 결과 DataFrame: 원본의 핵심 식별자(있으면) + Predicted 추가
    out = pd.DataFrame({'Predicted': pred})
    for key_col in ('Order ID', 'SKU', 'index'):
        if key_col in raw.columns:
            out.insert(0, key_col, raw[key_col].iloc[: len(pred)].values)

    meta = {
        'model_type': model_type,
        'input_dim': cfg['input_dim'],
        'rows': int(len(pred)),
        'elapsed_seconds': round(elapsed, 3),
    }
    return out, meta


def df_to_csv_string(df: pd.DataFrame) -> str:
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
