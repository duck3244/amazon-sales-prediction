"""Pydantic 요청/응답 스키마."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    missing: int
    missing_pct: float


class DatasetSummary(BaseModel):
    rows: int
    cols: int
    columns: List[ColumnSummary]
    has_target: bool = Field(description="기본 타겟 컬럼(Amount) 존재 여부")


class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    summary: DatasetSummary


class ApiError(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class TrainRequest(BaseModel):
    model_type: str = Field(default='basic', pattern='^(basic|advanced|attention)$')
    epochs: int = Field(default=30, ge=1, le=500)
    batch_size: int = Field(default=256, ge=1, le=4096)
    lr: float = Field(default=1e-3, gt=0, le=1.0)
    weight_decay: float = Field(default=1e-5, ge=0, le=1.0)
    dropout: float = Field(default=0.3, ge=0, le=0.9)
    hidden_dims: List[int] = Field(default_factory=lambda: [256, 128, 64])
    patience: int = Field(default=15, ge=1, le=200)
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)


class EpochPointResponse(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    params: dict
    current_epoch: int
    total_epochs: int
    history: List[EpochPointResponse]
    best_val_loss: Optional[float]
    early_stopped: bool
    test_loss: Optional[float]
    error: Optional[str]


class TrainStartResponse(BaseModel):
    job_id: str
    status: str


class PreprocessResponse(BaseModel):
    train: int
    val: int
    test: int
    n_features: int
    feature_names: List[str]


class ModelInfo(BaseModel):
    model_type: str
    input_dim: int
    hidden_dims: List[int]
    dropout: float
    best_val_loss: float
    test_loss: Optional[float] = None


class EvaluationResponse(BaseModel):
    mae: float
    mse: float
    rmse: float
    r2: float
    n_samples: int
    predictions_sample: List[float]
    actuals_sample: List[float]


class TrainedModelInfo(BaseModel):
    model_type: str
    input_dim: int
    hidden_dims: List[int]
    dropout: float
    best_val_loss: float
    test_loss: Optional[float] = None
    available: bool


class CompareEntry(BaseModel):
    model_type: str
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    n_samples: Optional[int] = None
    error: Optional[str] = None


class CompareResponse(BaseModel):
    entries: List[CompareEntry]


class SinglePredictRequest(BaseModel):
    features: List[float]


class SinglePredictResponse(BaseModel):
    prediction: float


class JobLogResponse(BaseModel):
    job_id: str
    total: int               # 누적 라인 수
    next_offset: int         # 다음 폴링에 사용할 offset
    lines: List[str]         # since..total 사이의 라인들
