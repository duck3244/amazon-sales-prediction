# Architecture — Amazon Sales Prediction

## 1. 개요

Amazon 판매 데이터 기반 딥러닝 매출 예측 시스템. 단일 리포지토리 안에 두 개의 독립 애플리케이션으로 구성된다.

| 영역 | 스택 | 역할 |
|------|------|------|
| **backend/** | Python 3.9+, PyTorch 2.x, FastAPI 0.115+, scikit-learn, pandas | 데이터 전처리 · 모델 학습 · 예측 · REST API |
| **frontend/** | Vue 3 (Composition API) + TypeScript, Vite 5, Pinia, Tailwind 3, ECharts 5 | 데이터 업로드 · 학습 모니터링 · 평가/비교 · 예측 UI |

배포 시에는 FastAPI가 `frontend/dist`를 정적으로 서빙하므로 단일 프로세스로 동작 가능하다 (개발 시에는 Vite 5173 ↔ FastAPI 8000 분리, `/api` 프록시).

---

## 2. 시스템 아키텍처 (레이어드 뷰)

```
┌──────────────────────────────────────────────────────────────┐
│                    Browser (SPA)                             │
│  Upload  │  Train  │  Evaluate  │  Compare  │  Predict       │
└──────────┬───────────────────────────────────────────────────┘
           │ HTTP/JSON (axios, /api/*)
           │
┌──────────▼───────────────────────────────────────────────────┐
│                  FastAPI Application                         │
│  app/main.py — CORS · SPA fallback · health                  │
│                                                              │
│  ┌──────────────── API Layer (app/api) ─────────────────┐   │
│  │  data.py · preprocess.py · train.py · predict.py     │   │
│  │  Pydantic 스키마 검증 (app/schemas.py)                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────── Service Layer (app/services) ────────┐   │
│  │  dataset_service · training_service ·                │   │
│  │  predict_service · batch_predict_service ·           │   │
│  │  job_store (in-memory)                               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────── ML Core (backend/*.py) ──────────────┐   │
│  │  data_preprocessing.py · model.py ·                  │   │
│  │  train.py · predict.py · utils.py                    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────┬───────────────────────────────────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────────┐
│                Filesystem 영속 계층                           │
│  Amazon Sale Report.csv · X_*.npy / y_*.npy ·                │
│  preprocessor.pkl · models/{type}/best_model.pth · config.json│
│  results/* (시각화/평가 산출물)                                │
└──────────────────────────────────────────────────────────────┘
```

설계 의도:
- **API ↔ Service ↔ ML Core 3-tier**: 라우터는 검증만, 비즈니스 로직은 서비스, 모델/전처리는 순수 Python 모듈로 격리.
- **데이터베이스 없음**: 학습 잡 상태는 `JobStore`(인메모리), 데이터/모델은 파일 시스템에만 저장. MVP 단순성을 우선.
- **CLI/REST 양립**: `train.py`, `predict.py` 등 ML 코어 모듈은 단독 실행 가능하면서 동시에 API 서비스에서도 호출됨.

---

## 3. 백엔드 모듈 구성

### 3.1 ML 코어 (`backend/`)

| 파일 | 책임 |
|------|------|
| `data_analysis.py` | EDA용 독립 스크립트 (분포·상관관계·이상치). API에는 직접 연결되지 않음 |
| `data_preprocessing.py` | `DataPreprocessor` — sklearn 스타일 fit/transform 파이프라인 (학습 분할에서만 fit → 누수 방지) |
| `model.py` | 3종 PyTorch 모델 (`SalesPredictor`, `AdvancedSalesPredictor`, `AttentionSalesPredictor`) + `get_model()` 팩토리 |
| `train.py` | `SalesDataset`, `EarlyStopping`, `Trainer` (학습 루프·체크포인트·시각화) |
| `predict.py` | `Predictor` 클래스 + `evaluate_model`, `predict_new_data` 등 헬퍼 |
| `utils.py` | 시드 고정, 파라미터 카운트, 디바이스 선택, 결과 분석 등 |

### 3.2 API 레이어 (`backend/app/api/`)

| 라우터 | 엔드포인트 | 설명 |
|--------|-----------|------|
| `data.py` | `POST /api/data/upload`, `GET /api/data/summary` | CSV 업로드 + 데이터셋 요약 |
| `preprocess.py` | `POST /api/preprocess` | `DataPreprocessor` 실행 → train/val/test `.npy` 산출 |
| `train.py` | `POST /api/train`, `GET /api/train/active`, `GET /api/train/{job_id}`, `GET /api/train/{job_id}/log` | 비동기 학습 잡 + 폴링 |
| `predict.py` | `GET /api/models`, `POST /api/predict/batch`, `POST /api/predict/compare`, `POST /api/predict/single`, `POST /api/predict/batch_csv` | 모델 목록·평가·비교·단일/배치 예측 |

요청/응답 타입은 모두 `app/schemas.py`의 Pydantic 모델로 정의되어 OpenAPI 스키마가 자동 생성된다.

### 3.3 서비스 레이어 (`backend/app/services/`)

| 모듈 | 핵심 |
|------|------|
| `dataset_service.py` | 청크 단위 업로드 저장, `summarize_csv()` |
| `job_store.py` | `EpochPoint`, `TrainingJob`, `JobStore` — 동시 학습 잡 1개로 제한 |
| `training_service.py` | `run_training(job)` — `asyncio.to_thread`로 학습 루프 실행, on_epoch 콜백으로 진행률·로그 갱신 |
| `predict_service.py` | `load_trained_model`, `evaluate_test_set`, `predict_single` |
| `batch_predict_service.py` | 업로드 CSV → 학습된 preprocessor 적용 → 배치 예측 → CSV 응답 |

### 3.4 설정 (`backend/app/settings.py`)

`BASE_DIR`, `DATA_PATH`, `PREPROC_PATH`, `MODELS_DIR`, `RESULTS_DIR`, `UPLOADS_DIR`, `CORS_ORIGINS`, `MAX_UPLOAD_BYTES`(100MB), `FRONTEND_DIST_DIR`. 경로는 절대경로 상수로 중앙집중.

---

## 4. 프론트엔드 모듈 구성

```
frontend/src/
├── api/        # axios 인스턴스(/api), 도메인별 호출 함수, OpenAPI 타입
├── stores/     # Pinia: dataset, training (jobId/status persisted)
├── views/      # Upload · Train · Evaluate · Compare · Predict
├── components/ # StatCard · HealthBadge · LossChart · ScatterChart · MetricBarChart
├── router/     # vue-router (history mode)
├── App.vue     # 네비 + RouterView
└── main.ts     # Pinia + persistedstate 플러그인 + 라우터
```

핵심 설계:
- **API 클라이언트 분리**: `api/data.ts`, `api/train.ts`, `api/predict.ts`로 도메인별 호출. `client.ts`는 axios 인스턴스 + 에러 인터셉터.
- **상태 영속화**: `dataset` 스토어는 요약을 통째로, `training` 스토어는 `jobId`만 localStorage에 저장 → 새로고침 후 활성 잡 복구.
- **실시간 진행률**: `Train.vue`가 1500ms 주기로 `/api/train/{id}` + `/api/train/{id}/log?since=offset`을 폴링 (WebSocket 미사용).
- **시각화**: ECharts(`vue-echarts`)로 손실 곡선·산점도·메트릭 막대 차트 렌더링.

---

## 5. 데이터 플로우 (E2E)

```
[1] 업로드
  Upload.vue → POST /api/data/upload (multipart)
   → dataset_service.save_uploaded_csv() → UPLOADS_DIR + DATA_PATH 복사
   → dataset_service.summarize_csv() → DatasetSummary

[2] 전처리
  Train.vue → POST /api/preprocess
   → DataPreprocessor: 컬럼 드롭 → 결측 행 제거 → 상태 필터 → 날짜 피처 추출
   → train/val/test 분할 (0.8/0.1/0.1)
   → fit on TRAIN: 결측 통계 / 카테고리 인코더 / StandardScaler
   → transform on ALL → X_*.npy / y_*.npy + preprocessor.pkl

[3] 학습 (비동기)
  Train.vue → POST /api/train (TrainRequest)
   → JobStore.acquire() → asyncio.to_thread(run_training)
   → 폴링: GET /api/train/{id} + /log?since=offset
   → run_training: DataLoader → get_model(type) → Trainer.fit()
       · 매 에폭 on_epoch_end 콜백 → JobStore.update(history, epoch, log)
       · best_val_loss 갱신 시 best_model.pth 저장
       · EarlyStopping.patience 초과 시 중단
   → load best ckpt → test_loss → models/{type}/config.json 저장
   → status = completed

[4] 평가/비교
  Evaluate.vue → POST /api/predict/batch?model_type=...
  Compare.vue  → POST /api/predict/compare
   → predict_service: load config → load checkpoint → predict X_test
   → MAE/MSE/RMSE/R² + 샘플 반환

[5] 예측
  Predict.vue [Single] → POST /api/predict/single
   → predict_service.predict_single(features)
  Predict.vue [Batch]  → POST /api/predict/batch_csv (multipart)
   → batch_predict_service: preprocessor.transform → batch infer → CSV 스트리밍
```

---

## 6. 핵심 파일 산출물 (Filesystem)

| 경로 | 생성 시점 | 용도 |
|------|----------|------|
| `backend/Amazon Sale Report.csv` | 업로드 시 활성화 | 활성 데이터셋 |
| `backend/uploads/*.csv` | 매 업로드 | 원본 보존 |
| `backend/X_train.npy` `y_train.npy` 등 | 전처리 | 학습/검증/테스트 분할 |
| `backend/preprocessor.pkl` | 전처리 | 신규 CSV 추론 시 동일 변환 적용 |
| `backend/models/{basic,advanced,attention}/best_model.pth` | 학습 | 최적 체크포인트 |
| `backend/models/{type}/config.json` | 학습 | 모델 하이퍼파라미터 (재로딩용) |
| `backend/models/{type}/training_history.png` | 학습 종료 | 손실 곡선 |
| `backend/results/evaluation_results.json` | 평가 | 지표 |
| `backend/results/prediction_visualization.png` | 평가 | 산점도/잔차 |

---

## 7. 동시성과 비동기 모델

- FastAPI는 단일 워커, 학습 잡은 **`asyncio.to_thread`** 로 실행 → 블로킹 PyTorch 코드를 이벤트 루프 외부로 분리.
- `JobStore`가 `_active` 슬롯을 잠가 **동시 학습 잡 1개** 보장 (GPU·matplotlib 전역 상태 충돌 회피).
- 로그는 `TrainingJob.log_lines`에 누적되고 클라이언트는 offset 기반 증분 폴링으로 가져감.

---

## 8. 주요 설계 결정과 트레이드오프

| 결정 | 이유 | 트레이드오프 |
|------|------|------------|
| In-memory `JobStore` | MVP 단순화, 외부 인프라 불필요 | 프로세스 재시작 시 잡 이력 손실 → 향후 Redis/DB 이관 여지 |
| 폴링 기반 진행률 | WebSocket 인프라 없이 구현 가능 | 1.5초 지연, 트래픽 약간 증가 |
| 전처리 산출물을 `.npy`로 직렬화 | numpy I/O가 빠르고 결정적 | 메모리에 한 번에 적재 → 대규모 데이터셋엔 부적합 |
| 3종 모델 변형 (basic/advanced/attention) | 비교 학습/리포팅 학습용 | 모델 수만큼 디스크 사용 증가 |
| sklearn fit/transform 컨벤션 | 누수 방지 + 추론 시 재사용 단순 | preprocessor.pkl과 학습 분포 결합 → 데이터 변경 시 재학습 필수 |
| 단일 워커 권고 (`--workers 1`) | matplotlib·PyTorch 전역 상태 보호 | 수평 스케일링 불가 → 학습/추론 분리 시 별도 설계 필요 |

---

## 9. 배포 토폴로지

**개발 모드**
```
Vite :5173  ──proxy /api──▶  Uvicorn :8000 (FastAPI)
                                       └─ ML core, filesystem
```

**프로덕션 모드 (단일 프로세스)**
```
Browser ──▶  Uvicorn :8000
              ├─ /api/*  → FastAPI 라우터
              └─ /*      → frontend/dist 정적 + SPA fallback
```

권장 구성: 리버스 프록시(Nginx) 뒤에서 Uvicorn 실행, 정적 자산은 캐싱, 업로드 크기 100MB 제한 일치.

---

## 10. 참고: 핵심 코드 위치

- 모델 정의: `backend/model.py:9-251`
- 전처리 파이프라인: `backend/data_preprocessing.py:26-196`
- 학습 루프: `backend/train.py:77-216`
- 학습 잡 오케스트레이션: `backend/app/services/training_service.py`, `backend/app/services/job_store.py`
- API 진입점: `backend/app/main.py`
- 프론트 엔트리: `frontend/src/main.ts`, `frontend/src/App.vue`
- 학습 화면(폴링/차트): `frontend/src/views/Train.vue`
