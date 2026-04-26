"""학습 잡 실행 서비스 — 별도 스레드에서 train.Trainer를 호출."""

from __future__ import annotations
import gc
import json
import logging
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from app.services.job_store import EpochPoint, TrainingJob, job_store
from app.settings import (
    MODELS_DIR,
    X_TRAIN_PATH, X_VAL_PATH, X_TEST_PATH,
    Y_TRAIN_PATH, Y_VAL_PATH, Y_TEST_PATH,
)

# 기존 ML 코드 재사용
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model import get_model        # noqa: E402
from train import EarlyStopping, SalesDataset, Trainer  # noqa: E402
from utils import set_seed         # noqa: E402

logger = logging.getLogger(__name__)


def _log(job: TrainingJob, line: str) -> None:
    """잡 로그에 한 줄 추가 (폴링으로 노출)."""
    ts = time.strftime('%H:%M:%S')
    job.log_lines.append(f'[{ts}] {line}')


def _load_arrays() -> tuple:
    for p in (X_TRAIN_PATH, X_VAL_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_VAL_PATH, Y_TEST_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f'전처리된 데이터를 찾을 수 없습니다: {p.name}. 먼저 전처리를 실행하세요.'
            )
    return (
        np.load(X_TRAIN_PATH), np.load(X_VAL_PATH), np.load(X_TEST_PATH),
        np.load(Y_TRAIN_PATH), np.load(Y_VAL_PATH), np.load(Y_TEST_PATH),
    )


def run_training(job: TrainingJob) -> None:
    """블로킹 학습 함수. asyncio.to_thread로 호출되어 FastAPI 이벤트 루프를 안 막음.

    주요 진행 상태는 명시적으로 ``job.log_lines``에 누적되어
    ``GET /api/train/{id}/log``로 노출된다.
    """
    job_store.update(job.job_id, status='running', started_at=time.time())
    p = job.params
    _log(job, f'Job {job.job_id} 시작 (model_type={p.get("model_type")}, '
              f'epochs={p.get("epochs")}, batch_size={p.get("batch_size")}, lr={p.get("lr")})')

    model = None
    trainer = None
    try:
        set_seed(int(p.get('seed', 42)), deterministic=bool(p.get('deterministic', False)))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _log(job, f'Device: {device} '
                  f'({torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})')

        X_train, X_val, X_test, y_train, y_val, y_test = _load_arrays()
        _log(job, f'데이터 로드 완료: train={len(X_train)}, val={len(X_val)}, '
                  f'test={len(X_test)}, n_features={X_train.shape[1]}')

        train_ds = SalesDataset(X_train, y_train)
        val_ds = SalesDataset(X_val, y_val)
        test_ds = SalesDataset(X_test, y_test)

        loader_kwargs = dict(
            num_workers=int(p.get('num_workers', 0)),
            pin_memory=device.type == 'cuda',
        )
        bs = int(p.get('batch_size', 256))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs)

        model_type = p.get('model_type', 'basic')
        save_dir = MODELS_DIR / model_type
        save_dir.mkdir(parents=True, exist_ok=True)

        model = get_model(
            model_type=model_type,
            input_dim=X_train.shape[1],
            hidden_dims=p.get('hidden_dims', [256, 128, 64]),
            dropout_rate=float(p.get('dropout', 0.3)),
        )
        n_params = sum(par.numel() for par in model.parameters())
        _log(job, f'모델: {model_type} (params={n_params:,})')

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=float(p.get('lr', 0.001)),
                               weight_decay=float(p.get('weight_decay', 1e-5)))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=int(p.get('patience', 15)), mode='min')

        def on_epoch_end(epoch, total, train_loss, val_loss, best, stopped):
            job.history.append(EpochPoint(epoch=epoch,
                                          train_loss=float(train_loss),
                                          val_loss=float(val_loss)))
            job_store.update(
                job.job_id,
                current_epoch=epoch,
                total_epochs=total,
                best_val_loss=float(best),
                early_stopped=bool(stopped),
            )
            _log(job, f'Epoch {epoch}/{total} · train={train_loss:.6f} '
                      f'val={val_loss:.6f} best={best:.6f}')
            if stopped:
                _log(job, f'Early stopping triggered (epoch {epoch})')

        trainer = Trainer(model, device, save_dir=save_dir, quiet=True)
        trainer.fit(
            train_loader, val_loader, criterion, optimizer,
            scheduler=scheduler,
            epochs=int(p.get('epochs', 30)),
            early_stopping=early_stopping,
            on_epoch_end=on_epoch_end,
        )

        # best 가중치 복원 후 test 평가
        trainer.load_checkpoint('best_model.pth')
        test_loss = trainer.validate(test_loader, criterion)
        _log(job, f'Test loss (best ckpt): {test_loss:.6f}')

        config = {
            'model_type': model_type,
            'input_dim': int(X_train.shape[1]),
            'hidden_dims': p.get('hidden_dims', [256, 128, 64]),
            'dropout': float(p.get('dropout', 0.3)),
            'batch_size': bs,
            'learning_rate': float(p.get('lr', 0.001)),
            'epochs': int(p.get('epochs', 30)),
            'best_val_loss': trainer.best_val_loss,
            'test_loss': float(test_loss),
        }
        with (save_dir / 'config.json').open('w') as f:
            json.dump(config, f, indent=2)

        job_store.update(
            job.job_id,
            status='completed',
            test_loss=float(test_loss),
            finished_at=time.time(),
        )
        _log(job, '학습 완료')
        logger.info('Job %s completed: test_loss=%.6f', job.job_id, test_loss)

    except Exception as e:
        logger.exception('Job %s failed', job.job_id)
        _log(job, f'오류: {type(e).__name__}: {e}')
        job_store.update(
            job.job_id,
            status='failed',
            error=f'{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}',
            finished_at=time.time(),
        )
    finally:
        # GPU 메모리 회수
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        job_store.release()
