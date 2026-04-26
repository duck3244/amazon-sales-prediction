"""학습된 모델 로드 + 배치/단건 예측 서비스."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.settings import (
    MODELS_DIR,
    X_TEST_PATH, Y_TEST_PATH,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model import get_model  # noqa: E402


class ModelNotFound(Exception):
    pass


def _load_config(model_type: str) -> dict:
    cfg_path = MODELS_DIR / model_type / 'config.json'
    if not cfg_path.exists():
        raise ModelNotFound(f'학습된 {model_type} 모델이 없습니다 ({cfg_path}).')
    with cfg_path.open() as f:
        return json.load(f)


def load_trained_model(model_type: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = _load_config(model_type)
    ckpt_path = MODELS_DIR / model_type / 'best_model.pth'
    if not ckpt_path.exists():
        raise ModelNotFound(f'체크포인트가 없습니다 ({ckpt_path}).')

    model = get_model(
        model_type=cfg['model_type'],
        input_dim=cfg['input_dim'],
        hidden_dims=cfg['hidden_dims'],
        dropout_rate=cfg['dropout'],
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def _predict_array(model: torch.nn.Module, X: np.ndarray, device: torch.device,
                   batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(device)
        out.append(model(xb).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.empty((0,), dtype=np.float32)


def evaluate_test_set(model_type: str) -> dict:
    """저장된 X_test/y_test로 모델을 평가하고 메트릭을 반환."""
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        raise ModelNotFound('전처리된 X_test/y_test가 없습니다. 전처리를 먼저 실행하세요.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _cfg = load_trained_model(model_type, device)

    X = np.load(X_TEST_PATH)
    y = np.load(Y_TEST_PATH)
    pred = _predict_array(model, X, device)

    mae = float(mean_absolute_error(y, pred))
    mse = float(mean_squared_error(y, pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y, pred))
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
        'n_samples': int(len(y)),
        'predictions_sample': pred[:50].tolist(),
        'actuals_sample': y[:50].tolist(),
    }


@torch.no_grad()
def predict_single(model_type: str, features: list[float]) -> float:
    cfg_path = MODELS_DIR / model_type / 'config.json'
    if not cfg_path.exists():
        raise ModelNotFound(f'학습된 {model_type} 모델이 없습니다.')
    with cfg_path.open() as f:
        cfg = json.load(f)
    if len(features) != cfg['input_dim']:
        raise ValueError(
            f'입력 차원이 맞지 않습니다: 받은 값 {len(features)}, 모델 기대 {cfg["input_dim"]}.'
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_trained_model(model_type, device)
    x = torch.tensor([features], dtype=torch.float32, device=device)
    return float(model(x).cpu().item())
