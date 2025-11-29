# Amazon E-Commerce Sales Prediction Project

## 📋 프로젝트 개요
이 프로젝트는 Amazon 판매 데이터를 분석하고 PyTorch를 사용하여 판매량을 예측하는 딥러닝 모델을 구현합니다.

### 주요 특징
- 🔍 **포괄적인 데이터 분석**: 시각화 및 통계 분석
- 🛠️ **완전한 전처리 파이프라인**: 결측치 처리, 특성 추출, 정규화
- 🤖 **3가지 모델 아키텍처**: Basic, Advanced (Residual), Attention
- 📊 **고급 학습 기능**: Early Stopping, Learning Rate Scheduling, Dropout
- 📈 **종합적인 평가**: MAE, RMSE, R² 등 다양한 성능 지표
- 🎨 **풍부한 시각화**: 학습 곡선, 예측 결과, 상관관계 분석

## 📦 데이터셋
- **출처**: [Kaggle - E-Commerce Sales Dataset](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)
- **파일**: Amazon Sale Report.csv

## 📁 프로젝트 구조
```
amazon-sales-prediction/
│
├── README.md                          # 프로젝트 문서
├── requirements.txt                   # Python 패키지 목록
├── run_guide.sh                       # 실행 가이드 스크립트
├── amazon_sales_prediction.ipynb      # Jupyter 노트북
│
├── 📊 데이터 파일
│   └── Amazon Sale Report.csv         # Kaggle에서 다운로드 필요
│
├── 🔧 핵심 스크립트
│   ├── data_analysis.py               # 데이터 탐색 및 시각화
│   ├── data_preprocessing.py          # 데이터 전처리
│   ├── model.py                       # PyTorch 모델 정의
│   ├── train.py                       # 모델 학습
│   ├── predict.py                     # 예측 및 평가
│   └── utils.py                       # 유틸리티 함수
│
├── 💾 전처리된 데이터 (자동 생성)
│   ├── X_train.npy, y_train.npy      # 학습 데이터
│   ├── X_val.npy, y_val.npy          # 검증 데이터
│   ├── X_test.npy, y_test.npy        # 테스트 데이터
│   └── preprocessor.pkl               # 전처리 객체
│
├── 🤖 모델 디렉토리 (자동 생성)
│   └── models/
│       ├── best_model.pth             # 최고 성능 모델
│       ├── config.json                # 모델 설정
│       └── training_history.png       # 학습 곡선
│
└── 📈 결과 디렉토리 (자동 생성)
    └── results/
        ├── evaluation_results.json    # 성능 지표
        ├── predictions.csv            # 예측 결과
        └── prediction_visualization.png
```

---

## 🚀 시작하기

### Step 1: 환경 설정

#### 필수 요구사항
- Python 3.8 이상
- pip 패키지 매니저

#### 패키지 설치
```bash
pip install -r requirements.txt
```

**설치되는 주요 패키지:**
- `torch>=2.0.0` - PyTorch 딥러닝 프레임워크
- `numpy>=1.24.0` - 수치 계산
- `pandas>=2.0.0` - 데이터 처리
- `scikit-learn>=1.3.0` - 전처리 및 평가
- `matplotlib>=3.7.0` - 시각화
- `seaborn>=0.12.0` - 고급 시각화

---

### Step 2: 데이터 다운로드

1. Kaggle 웹사이트 접속
   ```
   https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data
   ```

2. `Amazon Sale Report.csv` 파일 다운로드

3. 프로젝트 루트 디렉토리에 저장
   ```
   amazon-sales-prediction/
   └── Amazon Sale Report.csv  ← 여기에 저장
   ```

---

### Step 3: 데이터 탐색 및 분석

#### 실행 명령어
```bash
python data_analysis.py
```

#### 수행 작업
- ✅ 데이터 기본 정보 출력 (shape, 컬럼, 데이터 타입)
- ✅ 기술 통계 계산
- ✅ 결측치 분석
- ✅ 수치형 변수 분포 시각화
- ✅ 범주형 변수 분포 분석
- ✅ 상관관계 히트맵 생성
- ✅ 이상치 탐지 (IQR 방법)

#### 생성되는 파일
```
📊 numerical_features_distribution.png  # 수치형 변수 분포
📊 correlation_heatmap.png              # 상관관계 히트맵
📊 [변수명]_distribution.png            # 범주형 변수 분포들
📄 data_summary.txt                     # 데이터 요약 정보
```

---

### Step 4: 데이터 전처리

#### 실행 명령어
```bash
python data_preprocessing.py
```

#### 수행 작업
1. **결측치 처리**
   - 수치형: 중앙값으로 대체
   - 범주형: 최빈값으로 대체

2. **날짜 특성 추출**
   - year, month, day, dayofweek, quarter

3. **범주형 변수 인코딩**
   - Label Encoding
   - 카테고리 수가 많은 경우 상위 50개만 유지

4. **추가 특성 생성**
   - 평균 가격 등 파생 변수

5. **이상치 제거**
   - IQR 방법 (threshold=1.5)

6. **특성 정규화**
   - StandardScaler 적용

7. **데이터 분할**
   - Train: 70%
   - Validation: 10%
   - Test: 20%

#### 생성되는 파일
```
💾 X_train.npy          # 학습 특성 데이터
💾 y_train.npy          # 학습 타겟 데이터
💾 X_val.npy            # 검증 특성 데이터
💾 y_val.npy            # 검증 타겟 데이터
💾 X_test.npy           # 테스트 특성 데이터
💾 y_test.npy           # 테스트 타겟 데이터
💾 preprocessor.pkl     # 전처리 객체 (스케일러, 인코더)
```

---

### Step 5: 모델 학습

#### 기본 실행
```bash
python train.py --epochs 100 --batch_size 32 --lr 0.001
```

#### 모델 타입별 학습

**1. Basic 모델 (기본 MLP)**
```bash
python train.py \
  --model_type basic \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001
```
- 3개의 완전 연결층 (256→128→64)
- BatchNorm + ReLU + Dropout

**2. Advanced 모델 (잔차 연결)**
```bash
python train.py \
  --model_type advanced \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001
```
- Residual Blocks
- Skip connections
- 깊은 네트워크 학습 용이

**3. Attention 모델**
```bash
python train.py \
  --model_type attention \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001
```
- Multi-head Self-Attention
- Feed-forward Network
- 특성 간 관계 학습

#### 커스텀 하이퍼파라미터
```bash
python train.py \
  --model_type basic \
  --hidden_dims 512 256 128 \
  --dropout 0.4 \
  --epochs 150 \
  --batch_size 64 \
  --lr 0.0005 \
  --weight_decay 1e-4 \
  --patience 20
```

#### 주요 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--model_type` | 모델 타입 (basic/advanced/attention) | basic |
| `--hidden_dims` | 은닉층 차원 리스트 | [256, 128, 64] |
| `--dropout` | 드롭아웃 비율 | 0.3 |
| `--epochs` | 학습 에포크 수 | 100 |
| `--batch_size` | 배치 크기 | 32 |
| `--lr` | 학습률 | 0.001 |
| `--weight_decay` | L2 정규화 가중치 | 1e-5 |
| `--patience` | Early Stopping patience | 15 |
| `--seed` | 랜덤 시드 | 42 |

#### 생성되는 파일
```
📁 models/
  ├── best_model.pth           # 최고 성능 모델 체크포인트
  ├── config.json              # 모델 설정 및 하이퍼파라미터
  └── training_history.png     # 학습/검증 손실 곡선
```

#### 학습 과정에서 제공되는 기능
- ✅ **Early Stopping**: 검증 손실이 개선되지 않으면 자동 종료
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau로 학습률 자동 조정
- ✅ **모델 체크포인팅**: 최고 성능 모델 자동 저장
- ✅ **Progress Bar**: 실시간 학습 진행 상황 표시
- ✅ **GPU 지원**: CUDA 자동 감지 및 사용

---

### Step 6: 예측 및 평가

#### 기본 실행
```bash
python predict.py --model_path models/best_model.pth
```

#### 상세 옵션
```bash
python predict.py \
  --model_path models/best_model.pth \
  --config_path models/config.json \
  --test_data X_test.npy \
  --test_labels y_test.npy \
  --batch_size 32 \
  --use_dataloader \
  --save_dir results
```

#### 수행 작업
1. **모델 로드**
   - 저장된 체크포인트 로드
   - 모델 설정 복원

2. **예측 수행**
   - 테스트 데이터에 대한 예측
   - 배치 처리로 효율적 계산

3. **성능 평가**
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R² Score

4. **결과 시각화**
   - Actual vs Predicted scatter plot
   - Residual plot
   - Residual distribution
   - Time series comparison

#### 생성되는 파일
```
📁 results/
  ├── evaluation_results.json         # 성능 지표 JSON
  ├── predictions.csv                 # 예측 결과 상세
  └── prediction_visualization.png    # 4가지 시각화 플롯
```

#### predictions.csv 구조
```csv
Actual,Predicted,Residual,Absolute_Error
100.5,98.3,2.2,2.2
200.1,205.7,-5.6,5.6
...
```

---

## 📊 모델 아키텍처

### 1. Basic Model (SalesPredictor)
```
Input (n_features)
    ↓
Linear(n_features → 256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(64 → 1)
    ↓
Output (prediction)
```

**특징:**
- 간단하고 효율적
- 대부분의 회귀 문제에 적합
- 빠른 학습 속도

### 2. Advanced Model (AdvancedSalesPredictor)
```
Input (n_features)
    ↓
Linear(n_features → 256) + BatchNorm + ReLU + Dropout
    ↓
Residual Block 1 (256 → 256)
    ↓
Residual Block 2 (256 → 256)
    ↓
Residual Block 3 (256 → 256)
    ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout
    ↓
Linear(128 → 1)
    ↓
Output (prediction)
```

**특징:**
- 깊은 네트워크 학습 가능
- Gradient vanishing 문제 완화
- 더 복잡한 패턴 학습

### 3. Attention Model (AttentionSalesPredictor)
```
Input (n_features)
    ↓
Feature Embedding (n_features → 256)
    ↓
Multi-head Self-Attention (4 heads)
    ↓
Residual + Layer Norm
    ↓
Feed-Forward Network
    ↓
Residual + Layer Norm
    ↓
Output Layer
    ↓
Output (prediction)
```

**특징:**
- 특성 간 관계 학습
- 중요한 특성에 집중
- 해석 가능성 향상

---

## 💡 사용 팁

### 1. GPU 사용
CUDA가 설치된 환경에서는 자동으로 GPU를 사용합니다.
```python
# 학습 시 자동으로 감지
Using device: cuda
GPU 0: NVIDIA GeForce RTX 3080
```

### 2. 모델 비교
여러 모델을 학습하여 성능 비교:
```bash
# Basic 모델
python train.py --model_type basic --save_dir models/basic

# Advanced 모델
python train.py --model_type advanced --save_dir models/advanced

# Attention 모델
python train.py --model_type attention --save_dir models/attention

# 각 모델 평가
python predict.py --model_path models/basic/best_model.pth --save_dir results/basic
python predict.py --model_path models/advanced/best_model.pth --save_dir results/advanced
python predict.py --model_path models/attention/best_model.pth --save_dir results/attention
```

### 3. 하이퍼파라미터 튜닝
실험해볼 수 있는 파라미터:
- **Learning Rate**: [0.0001, 0.0005, 0.001, 0.005]
- **Batch Size**: [16, 32, 64, 128]
- **Hidden Dimensions**: [128,64], [256,128,64], [512,256,128]
- **Dropout**: [0.2, 0.3, 0.4, 0.5]

### 4. Early Stopping
검증 손실이 개선되지 않으면 자동 종료:
```bash
# patience를 늘리면 더 오래 학습
python train.py --patience 20

# patience를 줄이면 빠르게 종료
python train.py --patience 5
```

### 5. 학습 재개
중단된 학습을 재개하려면 체크포인트 로드 기능 추가 필요 (현재는 처음부터 학습)

---

## 🎯 빠른 시작 (전체 파이프라인)

처음 프로젝트를 실행하는 경우:

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 데이터 다운로드 (Kaggle에서)
# Amazon Sale Report.csv를 현재 디렉토리에 저장

# 3. 데이터 분석
python data_analysis.py

# 4. 데이터 전처리
python data_preprocessing.py

# 5. 모델 학습
python train.py --epochs 100

# 6. 예측 및 평가
python predict.py
```

**예상 소요 시간:**
- 데이터 분석: 2-5분
- 데이터 전처리: 5-10분
- 모델 학습: 10-30분 (데이터 크기 및 하드웨어에 따라)
- 예측 및 평가: 1-2분

---

## 📈 성능 지표 해석

### MAE (Mean Absolute Error)
- 예측값과 실제값의 평균 절대 오차
- **낮을수록 좋음**
- 해석: 평균적으로 예측이 실제값에서 얼마나 벗어나는지

### RMSE (Root Mean Squared Error)
- 예측값과 실제값의 평균 제곱근 오차
- **낮을수록 좋음**
- MAE보다 큰 오차에 더 민감

### R² Score
- 모델의 설명력
- **1에 가까울수록 좋음**
- 0.8 이상: 매우 좋음
- 0.6-0.8: 좋음
- 0.4-0.6: 보통
- 0.4 미만: 개선 필요

---

## 🔧 문제 해결

### 1. CUDA Out of Memory
```bash
# 배치 크기 줄이기
python train.py --batch_size 16

# 또는 모델 크기 줄이기
python train.py --hidden_dims 128 64
```

### 2. 학습이 느린 경우
```bash
# 배치 크기 늘리기 (GPU 메모리가 충분한 경우)
python train.py --batch_size 128

# 에포크 수 줄이기
python train.py --epochs 50
```

### 3. Overfitting 발생
```bash
# Dropout 증가
python train.py --dropout 0.5

# Weight decay 증가
python train.py --weight_decay 1e-4

# Early stopping patience 줄이기
python train.py --patience 10
```

### 4. Underfitting 발생
```bash
# 모델 크기 증가
python train.py --hidden_dims 512 256 128 64

# Dropout 감소
python train.py --dropout 0.2

# 학습률 조정
python train.py --lr 0.0005
```

---

## 🎓 학습 자료

PyTorch 및 딥러닝 학습을 위한 추천 자료:
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai](https://www.fast.ai/)

---
