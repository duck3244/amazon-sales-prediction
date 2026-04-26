"""학습 잡 상태를 관리하는 in-memory 저장소.

MVP 전제:
  - 동시 학습 잡 1개로 제한 (GPU 메모리, matplotlib 전역 상태, 파일 충돌 방지).
  - 프로세스 재시작 시 잡 정보 소멸 (DB 미사용).
"""

from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EpochPoint:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass
class TrainingJob:
    job_id: str
    status: str = 'pending'              # pending | running | completed | failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # 입력 파라미터
    params: dict = field(default_factory=dict)

    # 진행 상태
    current_epoch: int = 0
    total_epochs: int = 0
    history: list[EpochPoint] = field(default_factory=list)
    best_val_loss: Optional[float] = None
    early_stopped: bool = False

    # 종료 후
    test_loss: Optional[float] = None
    error: Optional[str] = None

    # 캡처된 stdout 로그 (학습 중 실시간 누적). 잠금 없이 append만 함.
    log_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'job_id': self.job_id,
            'status': self.status,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'params': self.params,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'history': [vars(p) for p in self.history],
            'best_val_loss': self.best_val_loss,
            'early_stopped': self.early_stopped,
            'test_loss': self.test_loss,
            'error': self.error,
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, TrainingJob] = {}
        self._active: Optional[str] = None

    # ---------- 동시성 ----------
    def has_active(self) -> bool:
        with self._lock:
            if self._active is None:
                return False
            job = self._jobs.get(self._active)
            return job is not None and job.status == 'running'

    def acquire(self, params: dict) -> TrainingJob:
        """새 잡 생성 + active 슬롯 확보. 이미 active면 RuntimeError."""
        with self._lock:
            if self._active is not None:
                cur = self._jobs.get(self._active)
                if cur and cur.status == 'running':
                    raise RuntimeError('이미 진행 중인 학습이 있습니다.')
            job = TrainingJob(job_id=uuid.uuid4().hex[:12], params=params,
                              total_epochs=int(params.get('epochs', 0)))
            self._jobs[job.job_id] = job
            self._active = job.job_id
            return job

    def release(self) -> None:
        with self._lock:
            self._active = None

    # ---------- 상태 갱신 ----------
    def get(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TrainingJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in changes.items():
                setattr(job, k, v)


# 모듈 단일 인스턴스
job_store = JobStore()
