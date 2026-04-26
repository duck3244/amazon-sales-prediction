# UML — Amazon Sales Prediction

본 문서는 시스템의 정적/동적 구조를 Mermaid 기반 다이어그램으로 정리한다. (GitHub·VS Code·IntelliJ 모두 Mermaid 렌더링 지원)

- 클래스 다이어그램: ML 코어, 서비스, 프론트 스토어
- 컴포넌트 다이어그램: 백엔드/프론트 모듈 의존
- 시퀀스 다이어그램: 업로드, 전처리, 학습(폴링), 평가/예측
- 상태 다이어그램: TrainingJob 생명주기
- 유스케이스 다이어그램: 사용자 흐름

---

## 1. 클래스 다이어그램 — ML 코어 (`backend/`)

```mermaid
classDiagram
    class nn_Module {
        <<PyTorch>>
        +forward(x) Tensor
    }

    class SalesPredictor {
        -input_dim: int
        -hidden_dims: List[int]
        -dropout_rate: float
        -network: nn.Sequential
        +forward(x) Tensor
        -_initialize_weights()
    }

    class ResidualBlock {
        -fc1, fc2: nn.Linear
        -bn1, bn2: nn.BatchNorm1d
        -dropout: nn.Dropout
        +forward(x) Tensor
    }

    class AdvancedSalesPredictor {
        -input_layer: nn.Sequential
        -residual_blocks: nn.ModuleList
        -output_layer: nn.Sequential
        +forward(x) Tensor
    }

    class AttentionSalesPredictor {
        -feature_embedding: nn.Sequential
        -multihead_attn: nn.MultiheadAttention
        -ffn: nn.Sequential
        -norm1, norm2: nn.LayerNorm
        -output_layer: nn.Sequential
        +forward(x) Tensor
    }

    nn_Module <|-- SalesPredictor
    nn_Module <|-- ResidualBlock
    nn_Module <|-- AdvancedSalesPredictor
    nn_Module <|-- AttentionSalesPredictor
    AdvancedSalesPredictor o-- ResidualBlock : composes N개

    class get_model {
        <<factory>>
        +get_model(model_type, input_dim, **kwargs) nn_Module
    }
    get_model ..> SalesPredictor : creates
    get_model ..> AdvancedSalesPredictor : creates
    get_model ..> AttentionSalesPredictor : creates

    class DataPreprocessor {
        -scaler: StandardScaler
        -label_encoders: dict
        -category_whitelist: dict
        -numeric_medians: dict
        -categorical_modes: dict
        -feature_names: List[str]
        -target_column: str
        +load_data(path) DataFrame
        +drop_unwanted_columns(df) DataFrame
        +filter_status(df) DataFrame
        +extract_date_features(df, date_cols) DataFrame
        +fit_missing(df_train)
        +transform_missing(df) DataFrame
        +fit_categorical(df_train, top_n)
        +transform_categorical(df) DataFrame
        +fit_scaler(df_train)
        +transform_scaler(df) DataFrame
        +save_preprocessor(path)
        +load_preprocessor(path)
    }

    class SalesDataset {
        -X: Tensor
        -y: Tensor
        +__len__()
        +__getitem__(idx)
    }

    class EarlyStopping {
        -patience: int
        -counter: int
        -best_score: float
        -early_stop: bool
        +__call__(score) bool
    }

    class Trainer {
        -model: nn_Module
        -device: torch.device
        -save_dir: Path
        -train_losses: List[float]
        -val_losses: List[float]
        -best_val_loss: float
        +train_epoch(loader, loss, opt) float
        +validate(loader, loss) float
        +fit(train_l, val_l, loss, opt, sched, epochs, early_stop, on_epoch_end)
        +save_checkpoint(name, epoch, val_loss)
        +load_checkpoint(name) dict
        +plot_losses(path)
    }

    class Predictor {
        -model: nn_Module
        -device: torch.device
        +predict(X) ndarray
        +predict_dataloader(loader) tuple
    }

    Trainer o-- nn_Module
    Trainer ..> EarlyStopping : uses
    Trainer ..> SalesDataset : iterates via DataLoader
    Predictor o-- nn_Module
    DataPreprocessor ..> StandardScaler : uses
    DataPreprocessor ..> LabelEncoder : uses
```

---

## 2. 클래스 다이어그램 — 서비스/잡 관리 (`backend/app/`)

```mermaid
classDiagram
    class EpochPoint {
        +epoch: int
        +train_loss: float
        +val_loss: float
    }

    class TrainingJob {
        +job_id: str
        +status: str  // pending|running|completed|failed
        +created_at, started_at, finished_at: float
        +params: dict
        +current_epoch, total_epochs: int
        +history: List[EpochPoint]
        +best_val_loss: float
        +test_loss: float
        +early_stopped: bool
        +error: str
        +log_lines: List[str]
        +to_dict() dict
    }

    class JobStore {
        -_jobs: dict[str, TrainingJob]
        -_active: str
        +acquire(params) TrainingJob
        +release()
        +get(job_id) TrainingJob
        +list() List[TrainingJob]
        +update(job_id, **changes)
    }

    class TrainingService {
        <<module>>
        +run_training(job: TrainingJob)
        -_log(job, message)
    }

    class PredictService {
        <<module>>
        +load_trained_model(model_type, device)
        +evaluate_test_set(model_type) dict
        +predict_single(model_type, features) float
        -_predict_array(model, X, device, batch)
    }

    class DatasetService {
        <<module>>
        +save_uploaded_csv(stream, filename, activate) Path
        +summarize_csv(path) DatasetSummary
    }

    class BatchPredictService {
        <<module>>
        +batch_predict_csv(csv_path, model_type) tuple
        +df_to_csv_string(df) str
        -_load_preprocessor() DataPreprocessor
    }

    JobStore "1" o-- "*" TrainingJob
    TrainingJob "1" o-- "*" EpochPoint
    TrainingService ..> JobStore : updates
    TrainingService ..> TrainingJob : drives
    TrainingService ..> Trainer : invokes
    PredictService ..> nn_Module : loads
    BatchPredictService ..> DataPreprocessor : loads
    BatchPredictService ..> PredictService : reuses
    DatasetService ..> DatasetSummary : returns
```

---

## 3. 클래스 다이어그램 — 프론트엔드 (Pinia/Views)

```mermaid
classDiagram
    class DatasetStore {
        <<Pinia, persisted>>
        +summary: DatasetSummary | null
        +filename: string | null
        +sizeBytes: number | null
        +setUploaded(payload)
        +clear()
    }

    class TrainingStore {
        <<Pinia, jobId persisted>>
        +jobId: string | null
        +status: TrainStatusResponse | null
        +setJob(id)
        +setStatus(s)
        +clear()
    }

    class ApiClient {
        <<axios>>
        +api: AxiosInstance
        +fetchHealth()
    }

    class DataApi {
        +uploadDataset(file)
        +fetchSummary()
    }
    class TrainApi {
        +startTraining(req)
        +getActiveJob()
        +getJob(jobId)
        +getJobLog(jobId, since)
        +runPreprocess()
    }
    class PredictApi {
        +listModels()
        +evaluateBatch(modelType)
        +compareModels()
        +predictSingle(modelType, features)
        +predictBatch(modelType, file)
    }

    class UploadView
    class TrainView {
        -logLines: string[]
        -logOffset: number
        -isRunning: ComputedRef
        +pollOnce()
        +startPolling()
        +stopPolling()
    }
    class EvaluateView
    class CompareView
    class PredictView

    DataApi ..> ApiClient
    TrainApi ..> ApiClient
    PredictApi ..> ApiClient

    UploadView ..> DataApi : uses
    UploadView ..> DatasetStore : reads/writes
    TrainView ..> TrainApi : uses
    TrainView ..> TrainingStore : reads/writes
    EvaluateView ..> PredictApi
    CompareView ..> PredictApi
    PredictView ..> PredictApi
```

---

## 4. 컴포넌트 다이어그램 — 백엔드

```mermaid
flowchart TB
    subgraph FE[Frontend SPA]
        FE_UI[Views & Components]
    end

    subgraph API[FastAPI app/api]
        DA[data.py]
        PR[preprocess.py]
        TR[train.py]
        PD[predict.py]
    end

    subgraph SVC[app/services]
        DSV[dataset_service]
        TSV[training_service]
        PSV[predict_service]
        BSV[batch_predict_service]
        JS[(JobStore)]
    end

    subgraph CORE[ML Core]
        DPP[DataPreprocessor]
        MDL[model.py]
        TRN[Trainer]
        PRD[Predictor]
        UTL[utils.py]
    end

    subgraph FS[Filesystem]
        CSV[CSV / uploads]
        NPY["X_*.npy / y_*.npy"]
        PKL[preprocessor.pkl]
        MODELS["models/{type}/*"]
        RES["results/*"]
    end

    FE_UI -- /api --> DA
    FE_UI --> PR
    FE_UI --> TR
    FE_UI --> PD

    DA --> DSV
    PR --> DPP
    TR --> JS
    TR --> TSV
    PD --> PSV
    PD --> BSV

    TSV --> TRN
    TSV --> MDL
    TSV --> JS
    PSV --> MDL
    BSV --> DPP
    BSV --> PSV

    DSV --> CSV
    DPP --> NPY
    DPP --> PKL
    TRN --> MODELS
    PRD --> RES
```

---

## 5. 컴포넌트 다이어그램 — 프론트엔드

```mermaid
flowchart LR
    subgraph App[main.ts / App.vue]
        ROUTER[vue-router]
        PINIA[Pinia + persistedstate]
    end

    subgraph Views
        U[Upload.vue]
        T[Train.vue]
        E[Evaluate.vue]
        C[Compare.vue]
        P[Predict.vue]
    end

    subgraph Components
        SC[StatCard]
        HB[HealthBadge]
        LC[LossChart]
        SCH[ScatterChart]
        MBC[MetricBarChart]
    end

    subgraph Stores
        DS[useDatasetStore]
        TS[useTrainingStore]
    end

    subgraph API[api/*]
        CL[client.ts]
        DA[data.ts]
        TA[train.ts]
        PA[predict.ts]
    end

    ROUTER --> U & T & E & C & P
    U --> DS
    T --> TS
    T --> LC
    E --> SCH
    C --> MBC
    U --> SC & HB
    T --> SC
    E --> SC
    DA --> CL
    TA --> CL
    PA --> CL
    U --> DA
    T --> TA
    E --> PA
    C --> PA
    P --> PA
```

---

## 6. 시퀀스 다이어그램 — 데이터셋 업로드

```mermaid
sequenceDiagram
    actor User
    participant UV as Upload.vue
    participant DAxios as api/data.ts
    participant API as FastAPI /data/upload
    participant DSV as dataset_service
    participant FS as Filesystem

    User->>UV: CSV 선택
    UV->>DAxios: uploadDataset(file)
    DAxios->>API: POST /api/data/upload (multipart)
    API->>DSV: save_uploaded_csv(stream, filename)
    DSV->>FS: chunked write → uploads/, DATA_PATH 복사
    DSV-->>API: Path
    API->>DSV: summarize_csv(path)
    DSV-->>API: DatasetSummary
    API-->>DAxios: UploadResponse
    DAxios-->>UV: payload
    UV->>UV: datasetStore.setUploaded(payload)
    UV-->>User: 요약(행/열/결측) 표시
```

---

## 7. 시퀀스 다이어그램 — 전처리

```mermaid
sequenceDiagram
    actor User
    participant TV as Train.vue
    participant API as FastAPI /preprocess
    participant DPP as DataPreprocessor
    participant FS as Filesystem

    User->>TV: "전처리" 클릭
    TV->>API: POST /api/preprocess
    API->>DPP: load_data(DATA_PATH)
    DPP->>FS: read CSV
    API->>DPP: drop/filter/extract_date_features
    API->>DPP: train_test_split (0.8/0.1/0.1)
    API->>DPP: fit_missing/categorical/scaler (TRAIN only)
    API->>DPP: transform_* (모든 분할)
    API->>FS: save X_*.npy / y_*.npy
    API->>FS: save preprocessor.pkl
    API-->>TV: PreprocessResponse {train, val, test, n_features}
    TV-->>User: 분할 통계 표시
```

---

## 8. 시퀀스 다이어그램 — 학습 (비동기 + 폴링)

```mermaid
sequenceDiagram
    actor User
    participant TV as Train.vue
    participant API as FastAPI /train
    participant JS as JobStore
    participant TS as training_service
    participant TR as Trainer
    participant FS as Filesystem

    User->>TV: 하이퍼파라미터 제출
    TV->>API: POST /api/train (TrainRequest)
    API->>JS: acquire(params) -> job
    API-->>TV: { job_id, status: pending }
    API->>TS: asyncio.create_task(run_training(job))

    loop 1.5s 폴링
        TV->>API: GET /api/train/{id}
        API->>JS: get(id)
        API-->>TV: TrainStatusResponse(history, current_epoch, ...)
        TV->>API: GET /api/train/{id}/log?since=offset
        API-->>TV: { lines, next_offset }
        TV-->>User: 손실차트 + 로그 갱신
    end

    par 학습 스레드
        TS->>TR: Trainer.fit(train_l, val_l, on_epoch_end)
        loop epoch
            TR->>TR: train_epoch / validate
            TR->>JS: on_epoch_end → update(history, epoch, log)
            alt best_val_loss 갱신
                TR->>FS: save best_model.pth
            end
            alt EarlyStopping.early_stop
                TR->>TR: break
            end
        end
        TS->>FS: save config.json
        TS->>JS: update(status=completed, test_loss)
    end

    TV->>API: GET /api/train/{id}
    API-->>TV: status=completed
    TV-->>User: 최종 결과 표시
```

---

## 9. 시퀀스 다이어그램 — 평가 / 비교

```mermaid
sequenceDiagram
    actor User
    participant EV as Evaluate.vue
    participant API as /predict/batch
    participant PSV as predict_service
    participant FS as Filesystem

    User->>EV: 모델 선택 + 평가
    EV->>API: POST /api/predict/batch?model_type=...
    API->>PSV: load_trained_model(model_type)
    PSV->>FS: read config.json + best_model.pth
    API->>PSV: evaluate_test_set(model_type)
    PSV->>FS: load X_test.npy, y_test.npy
    PSV-->>API: {mae, mse, rmse, r2, samples}
    API-->>EV: EvaluationResponse
    EV-->>User: 메트릭 카드 + 산점도
```

```mermaid
sequenceDiagram
    actor User
    participant CV as Compare.vue
    participant API as /predict/compare
    participant PSV as predict_service

    User->>CV: "비교 실행"
    CV->>API: POST /api/predict/compare
    loop 모델 타입별
        API->>PSV: evaluate_test_set(type)
        PSV-->>API: 메트릭
    end
    API-->>CV: CompareResponse[]
    CV-->>User: 막대 차트 + 베스트 모델 배지
```

---

## 10. 시퀀스 다이어그램 — 단일/배치 예측

```mermaid
sequenceDiagram
    actor User
    participant PV as Predict.vue
    participant API as /predict/single
    participant PSV as predict_service

    User->>PV: features 입력
    PV->>API: POST /api/predict/single?model_type=...
    API->>PSV: predict_single(model_type, features)
    PSV-->>API: prediction
    API-->>PV: { prediction }
    PV-->>User: 결과 표시
```

```mermaid
sequenceDiagram
    actor User
    participant PV as Predict.vue
    participant API as /predict/batch_csv
    participant DSV as dataset_service
    participant BSV as batch_predict_service
    participant DPP as DataPreprocessor
    participant PSV as predict_service

    User->>PV: CSV 업로드 + 모델 선택
    PV->>API: POST /api/predict/batch_csv (multipart)
    API->>DSV: save_uploaded_csv(stream, activate=False)
    API->>BSV: batch_predict_csv(path, model_type)
    BSV->>DPP: load_preprocessor()
    BSV->>DPP: transform(raw_df)
    BSV->>PSV: load_trained_model + predict
    BSV-->>API: (output_df, metadata)
    API-->>PV: CSV stream + headers (rows, elapsed, model)
    PV-->>User: 결과 CSV 다운로드
```

---

## 11. 상태 다이어그램 — TrainingJob

```mermaid
stateDiagram-v2
    [*] --> pending : JobStore.acquire()
    pending --> running : training_service 시작
    running --> running : 에폭 진행 (history 갱신)
    running --> completed : 모든 에폭 완료 / EarlyStopping
    running --> failed : 예외 발생
    completed --> [*] : JobStore.release()
    failed --> [*] : JobStore.release()
```

---

## 12. 유스케이스 다이어그램

```mermaid
flowchart LR
    User((사용자))
    User --> UC1[데이터셋 업로드/요약]
    User --> UC2[전처리 실행]
    User --> UC3[모델 학습 시작/모니터링]
    User --> UC4[학습된 모델 평가]
    User --> UC5[모델 간 비교]
    User --> UC6[단일 입력 예측]
    User --> UC7[CSV 배치 예측 다운로드]

    subgraph FastAPI
        UC1 --- E1["/api/data/*"]
        UC2 --- E2["/api/preprocess"]
        UC3 --- E3["/api/train, /api/train/{id}, /log"]
        UC4 --- E4["/api/predict/batch"]
        UC5 --- E5["/api/predict/compare"]
        UC6 --- E6["/api/predict/single"]
        UC7 --- E7["/api/predict/batch_csv"]
    end
```

---

## 13. 핵심 객체 의존 요약 (한눈에 보기)

| 객체 | 의존 | 호출 주체 |
|------|------|----------|
| `Trainer` | `nn.Module`, `EarlyStopping`, `DataLoader` | `training_service.run_training` |
| `Predictor` | `nn.Module`, `DataLoader` | `predict_service`, `batch_predict_service` |
| `DataPreprocessor` | `StandardScaler`, `LabelEncoder` | `/preprocess` 라우터, `batch_predict_service` |
| `JobStore` | `TrainingJob`, `EpochPoint` | `/train` 라우터, `training_service` |
| FE `Train.vue` | `useTrainingStore`, `api/train.ts` | 사용자 폴링 루프 |
| FE `api/client.ts` | axios | 모든 도메인 API 모듈 |
